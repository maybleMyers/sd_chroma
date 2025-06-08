# flux_models.py

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple, TYPE_CHECKING
import logging

from library import utils # Assuming this is your utils.py for setup_logging
from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

import torch
from einops import rearrange
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from library import custom_offloading_utils
# from .math import attention, rope # Assuming these are globally defined below

logger = logging.getLogger(__name__)

# --- Helper for logging tensor stats ---
def log_tensor_stats(tensor: Optional[torch.Tensor], name: str = "tensor", logger_obj: Optional[logging.Logger] = None):
    if logger_obj is None:
        logger_obj = logging.getLogger(__name__) 

    if tensor is not None:
        has_elements = tensor.numel() > 0
        is_all_nan_tensor = torch.isnan(tensor)
        is_all_inf_tensor = torch.isinf(tensor)
        is_all_nan = is_all_nan_tensor.all().item() if has_elements else False
        is_all_inf = is_all_inf_tensor.all().item() if has_elements else False
        
        min_val_str, max_val_str, mean_val_str = 'N/A', 'N/A', 'N/A'

        if has_elements and not (is_all_nan or is_all_inf):
            finite_tensor = tensor[~(is_all_nan_tensor | is_all_inf_tensor)] 
            if finite_tensor.numel() > 0:
                min_val_str = f"{finite_tensor.min().item():.4e}"
                max_val_str = f"{finite_tensor.max().item():.4e}"
                mean_val_str = f"{finite_tensor.mean().item():.4e}"
            else: 
                 min_val_str, max_val_str, mean_val_str = 'All NaN/Inf', 'All NaN/Inf', 'All NaN/Inf'
        elif is_all_nan: min_val_str, max_val_str, mean_val_str = 'All NaN', 'All NaN', 'All NaN'
        elif is_all_inf: min_val_str, max_val_str, mean_val_str = 'All Inf', 'All Inf', 'All Inf'
        
        logger_obj.debug(
            f"{name}: shape {tensor.shape}, dtype {tensor.dtype}, "
            f"min {min_val_str}, max {max_val_str}, mean {mean_val_str}, "
            f"isnan {torch.isnan(tensor).any().item()}, isinf {torch.isinf(tensor).any().item()}"
        )
    else:
        logger_obj.debug(f"{name}: None")

# --- Official Chroma Math Utils ---
def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, mask: Optional[Tensor]=None) -> Tensor:
    q, k = apply_rope(q, k, pe)
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    x = rearrange(x, "B H L D -> B L (H D)")
    return x
# --- End Official Chroma Math Utils ---

# --- Dataclasses ---
@dataclass
class ApproximatorParams: # For the main modulator in Chroma
    in_dim: int = 64
    out_dim_per_mod_vector: int = 3072 
    hidden_dim: int = 5120 
    n_layers: int = 5 
    mod_index_length: int = 344

@dataclass
class FluxParams: # Kept this name for compatibility with flux_utils.py
    in_channels: int
    context_in_dim: int
    hidden_size: int 
    mlp_ratio: float
    num_heads: int   
    depth: int  
    depth_single_blocks: int 
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    # Chroma specific structure, controlled by how flux_chroma_params() populates this:
    guidance_embed: bool = False # For Chroma, this path for internal_cond_vec is off
    use_modulation: bool = False # For Chroma, this is False (uses external Approximator)
    use_distilled_guidance_layer: bool = True # For Chroma, this means main Approximator is active
    approximator_config: Optional[ApproximatorParams] = None # Holds config for the main Approximator
    # Below are for original FLUX Dev/Schnell, not used by Chroma path in __init__
    vec_in_dim: Optional[int] = None 
    distilled_guidance_dim: Optional[int] = None # Not used if img_p_enhancer is None
    use_time_embed: bool = False      
    use_vector_embed: bool = False    
    double_block_has_main_norms: Optional[bool] = None

@dataclass
class AutoEncoderParams: # Standard VAE params
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 16 # Output latent channels from Encoder before DiagonalGaussian
    scale_factor: float = 0.3611 # Values from FLUX.1 AE
    shift_factor: float = 0.1159

# --- Start Re-inserted Autoencoder Classes ---
def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q, k, v = self.q(h_), self.k(h_), self.v(h_)
        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = nn.functional.scaled_dot_product_attention(q, k, v)
        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))

class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: Optional[int] = None): # out_channels can be None
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=self.out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, self.out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h); h = swish(h); h = self.conv1(h)
        h = self.norm2(h); h = swish(h); h = self.conv2(h)
        if self.in_channels != self.out_channels: x = self.nin_shortcut(x)
        return x + h

class Downsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)
    def forward(self, x: Tensor):
        x = nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x: Tensor):
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, params: AutoEncoderParams): # Pass params object
        super().__init__()
        self.ch, self.num_resolutions, self.num_res_blocks = params.ch, len(params.ch_mult), params.num_res_blocks
        self.resolution, self.in_channels = params.resolution, params.in_channels
        self.conv_in = nn.Conv2d(self.in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        
        curr_res, in_ch_mult = self.resolution, (1,) + tuple(params.ch_mult)
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block_list = nn.ModuleList() # Changed from 'block' to 'block_list'
            block_in_level = self.ch * in_ch_mult[i_level] # Use block_in_level for clarity
            block_out_level = self.ch * params.ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block_list.append(ResnetBlock(in_channels=block_in_level, out_channels=block_out_level))
                block_in_level = block_out_level # Update block_in_level for next resblock in same level
            
            down_level = nn.Module()
            down_level.block = block_list # Assign the list of blocks
            # Official FLUX AE doesn't seem to have attention in encoder down blocks
            # down_level.attn = nn.ModuleList() # Empty if no attention
            
            if i_level != self.num_resolutions - 1:
                down_level.downsample = Downsample(block_in_level) # Use final block_in_level of this level
                curr_res //= 2
            self.down.append(down_level)
            block_in = block_in_level # Carry over to next level if needed (though loop redefines block_in)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in) # block_in is from last level
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * params.z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                # if len(self.down[i_level].attn) > 0: h = self.down[i_level].attn[i_block](h) # No attn in down blocks
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h); h = self.mid.attn_1(h); h = self.mid.block_2(h)
        h = self.norm_out(h); h = swish(h); h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, params: AutoEncoderParams): # Pass params object
        super().__init__()
        self.ch, self.num_resolutions, self.num_res_blocks = params.ch, len(params.ch_mult), params.num_res_blocks
        self.resolution, self.out_channels = params.resolution, params.out_ch # Use out_ch
        
        block_in = self.ch * params.ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv2d(params.z_channels, block_in, kernel_size=3, stride=1, padding=1)
        
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block_list = nn.ModuleList() # Changed from 'block'
            block_out_level = self.ch * params.ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1): # +1 for decoder
                block_list.append(ResnetBlock(in_channels=block_in, out_channels=block_out_level))
                block_in = block_out_level # Update for next resblock
            
            up_level = nn.Module()
            up_level.block = block_list
            # up_level.attn = nn.ModuleList() # No attn in up blocks in original FLUX AE
            if i_level != 0:
                up_level.upsample = Upsample(block_in) # block_in is now block_out_level from this level
            self.up.insert(0, up_level) # Prepend to maintain order

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, self.out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h); h = self.mid.attn_1(h); h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                # if len(self.up[i_level].attn) > 0: h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h); h = swish(h); h = self.conv_out(h)
        return h

class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__(); self.sample, self.chunk_dim = sample, chunk_dim
    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) if self.sample else mean

class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.encoder = Encoder(params) # Pass params object
        self.decoder = Decoder(params) # Pass params object
        self.reg = DiagonalGaussian()
        self.scale_factor, self.shift_factor = params.scale_factor, params.shift_factor
    @property
    def device(self) -> torch.device: return next(self.parameters()).device
    @property
    def dtype(self) -> torch.dtype: return next(self.parameters()).dtype
    def encode(self, x: Tensor) -> Tensor: return self.scale_factor * (self.reg(self.encoder(x)) - self.shift_factor)
    def decode(self, z: Tensor) -> Tensor: return self.decoder(z / self.scale_factor + self.shift_factor)
    def forward(self, x: Tensor) -> Tensor: return self.decode(self.encode(x))
# --- End Re-inserted Autoencoder Classes ---


# --- Official Chroma Layer Definitions (from src.models.chroma.module.layers.py) ---
# MLPEmbedder, RMSNorm, ModulationOut, distribute_modulations, Approximator, QKNorm, SelfAttention,
# _modulation_shift_scale_fn, _modulation_gate_fn,
# DoubleStreamBlock, SingleStreamBlock, LastLayer
# These should be the versions from the official `layers.py` you provided.
# (Pasted from previous response, assuming these are correct official versions)
class MLPEmbedder(nn.Module): # From official layers.py (seems identical to yours)
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)
    @property
    def device(self): return next(self.parameters()).device
    def forward(self, x: Tensor) -> Tensor: return self.out_layer(self.silu(self.in_layer(x)))

class RMSNorm(torch.nn.Module): # From official layers.py
    def __init__(self, dim: int, use_compiled: bool = False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x: Tensor):
        if hasattr(F, "rms_norm"): return F.rms_norm(x, self.scale.shape, weight=self.scale, eps=1e-6)
        else:
            x_dtype = x.dtype; x = x.float()
            rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
            return ((x * rrms) * self.scale.float()).to(dtype=x_dtype)

@dataclass
class ModulationOut: shift: Tensor; scale: Tensor; gate: Tensor

def distribute_modulations(tensor: torch.Tensor, num_double_blocks: int, num_single_blocks: int, final_layer_mod: bool = True) -> Dict[str, Any]:
    batch_size, total_vectors, dim_per_vector = tensor.shape; block_dict = {}; idx = 0
    expected_vectors = (num_double_blocks * 2 * 3) + (num_double_blocks * 2 * 3) + (num_single_blocks * 1 * 3)
    if final_layer_mod: expected_vectors += 2
    if total_vectors < expected_vectors: logger.warning(f"distribute_modulations: Have {total_vectors}, need {expected_vectors}.")
    def _get_slice(num_vecs_needed):
        nonlocal idx; end_idx = idx + num_vecs_needed
        if end_idx > total_vectors:
            s_ = torch.zeros(batch_size, 1, dim_per_vector, device=tensor.device, dtype=tensor.dtype)
            sc_ = torch.zeros(batch_size, 1, dim_per_vector, device=tensor.device, dtype=tensor.dtype)
            if num_vecs_needed == 3: g_ = torch.ones(batch_size, 1, dim_per_vector, device=tensor.device, dtype=tensor.dtype); return s_,sc_,g_
            return (s_,sc_) if num_vecs_needed == 2 else [s_]*num_vecs_needed
        slices = [tensor[:, i:i+1, :] for i in range(idx, end_idx)]; idx = end_idx
        return slices[0], slices[1], slices[2] if num_vecs_needed == 3 else (slices[0], slices[1])
    for i in range(num_single_blocks): s, sc, g = _get_slice(3); block_dict[f"single_blocks.{i}.modulation.lin"] = ModulationOut(shift=s, scale=sc, gate=g)
    for i in range(num_double_blocks):
        img_m, txt_m = [], []
        for _ in range(2): s, sc, g = _get_slice(3); img_m.append(ModulationOut(shift=s, scale=sc, gate=g))
        for _ in range(2): s, sc, g = _get_slice(3); txt_m.append(ModulationOut(shift=s, scale=sc, gate=g))
        block_dict[f"double_blocks.{i}.img_mod.lin"] = img_m; block_dict[f"double_blocks.{i}.txt_mod.lin"] = txt_m
    if final_layer_mod: s, sc = _get_slice(2); block_dict["final_layer.adaLN_modulation.1"] = [s, sc]
    return block_dict

class Approximator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers=4):
        super().__init__(); self.in_proj = nn.Linear(in_dim, hidden_dim, bias=True)
        self.layers = nn.ModuleList([MLPEmbedder(hidden_dim, hidden_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(hidden_dim) for _ in range(n_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)
    @property
    def device(self): return next(self.parameters()).device
    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        for layer, norm_layer in zip(self.layers, self.norms): x = x + layer(norm_layer(x))
        return self.out_proj(x)

class QKNorm(torch.nn.Module):
    def __init__(self, dim: int, use_compiled: bool = False):
        super().__init__(); self.query_norm = RMSNorm(dim); self.key_norm = RMSNorm(dim)
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return self.query_norm(q).to(v.dtype), self.key_norm(k).to(v.dtype)

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, use_compiled: bool = False):
        super().__init__(); self.num_heads = num_heads; head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias); self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x: Tensor, pe: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        qkv = self.qkv(x); q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v); x = attention(q, k, v, pe=pe, mask=mask)
        return self.proj(x)

def _modulation_shift_scale_fn(x, scale, shift): return (1 + scale) * x + shift
def _modulation_gate_fn(x, gate, gate_params): return x + gate * gate_params

class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, use_compiled: bool = False, **kwargs): # Added **kwargs to absorb extra params
        super().__init__(); mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads; self.hidden_size = hidden_size
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden_dim, bias=True), nn.GELU(approximate="tanh"), nn.Linear(mlp_hidden_dim, hidden_size, bias=True))
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden_dim, bias=True), nn.GELU(approximate="tanh"), nn.Linear(mlp_hidden_dim, hidden_size, bias=True))
    @property
    def device(self): return next(self.parameters()).device
    def forward( self, img: Tensor, txt: Tensor, pe: Tensor, distill_vec: list[list[ModulationOut]], mask: Optional[Tensor]=None, **kwargs) -> tuple[Tensor, Tensor]: # Added **kwargs
        (img_mod_attn, img_mod_mlp), (txt_mod_attn, txt_mod_mlp) = distill_vec[0], distill_vec[1]
        img_res, txt_res = img, txt
        img_modulated_for_qkv = _modulation_shift_scale_fn(self.img_norm1(img_res), img_mod_attn.scale, img_mod_attn.shift)
        img_qkv = self.img_attn.qkv(img_modulated_for_qkv); img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)
        txt_modulated_for_qkv = _modulation_shift_scale_fn(self.txt_norm1(txt_res), txt_mod_attn.scale, txt_mod_attn.shift)
        txt_qkv = self.txt_attn.qkv(txt_modulated_for_qkv); txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)
        q_joint = torch.cat((txt_q, img_q), dim=2); k_joint = torch.cat((txt_k, img_k), dim=2); v_joint = torch.cat((txt_v, img_v), dim=2)
        attn_output_joint = attention(q_joint, k_joint, v_joint, pe=pe, mask=mask)
        txt_attn_output = attn_output_joint[:, :txt_res.shape[1]]; img_attn_output = attn_output_joint[:, txt_res.shape[1]:]
        img_after_attn = _modulation_gate_fn(img_res, img_mod_attn.gate, self.img_attn.proj(img_attn_output))
        img_mlp_input = _modulation_shift_scale_fn(self.img_norm2(img_after_attn), img_mod_mlp.scale, img_mod_mlp.shift)
        img_final = _modulation_gate_fn(img_after_attn, img_mod_mlp.gate, self.img_mlp(img_mlp_input))
        txt_after_attn = _modulation_gate_fn(txt_res, txt_mod_attn.gate, self.txt_attn.proj(txt_attn_output))
        txt_mlp_input = _modulation_shift_scale_fn(self.txt_norm2(txt_after_attn), txt_mod_mlp.scale, txt_mod_mlp.shift)
        txt_final = _modulation_gate_fn(txt_after_attn, txt_mod_mlp.gate, self.txt_mlp(txt_mlp_input))
        return img_final, txt_final

class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, qk_scale: float | None = None, use_compiled: bool = False, **kwargs): # Added **kwargs
        super().__init__(); self.hidden_dim = hidden_size; self.num_heads = num_heads
        head_dim = hidden_size // num_heads; self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=True)
        self.norm = QKNorm(head_dim); self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")
    @property
    def device(self): return next(self.parameters()).device
    def forward(self, x: Tensor, pe: Tensor, distill_vec: ModulationOut, mask: Optional[Tensor]=None, **kwargs) -> Tensor: # Added **kwargs
        mod = distill_vec; x_res = x; x_normed = self.pre_norm(x)
        x_modulated = _modulation_shift_scale_fn(x_normed, mod.scale, mod.shift)
        qkv_combined, mlp_in_features = torch.split(self.linear1(x_modulated), [3 * self.hidden_dim, self.mlp_hidden_dim], dim=-1)
        q, k, v = rearrange(qkv_combined, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v); attn_output = attention(q, k, v, pe=pe, mask=mask)
        mlp_output_activated = self.mlp_act(mlp_in_features)
        combined_features = torch.cat((attn_output, mlp_output_activated), dim=-1)
        output_projection = self.linear2(combined_features)
        return _modulation_gate_fn(x_res, mod.gate, output_projection)

class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, use_compiled: bool = False):
        super().__init__(); self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
    @property
    def device(self): return next(self.parameters()).device
    def forward(self, x: Tensor, distill_vec: list[Tensor], **kwargs) -> Tensor: # Added **kwargs
        shift, scale = distill_vec[0], distill_vec[1]
        if shift.ndim == 3 and shift.shape[1] == 1: shift = shift.squeeze(1)
        if scale.ndim == 3 and scale.shape[1] == 1: scale = scale.squeeze(1)
        x_normed = self.norm_final(x)
        x_modulated = (1 + scale.unsqueeze(1)) * x_normed + shift.unsqueeze(1)
        return self.linear(x_modulated)
# --- End Official Chroma Layer Definitions ---


# --- Flux class refactored to be ChromaModel ---
# This is the class that will be instantiated by flux_utils.load_flow_model
# It needs to be named "Flux" for compatibility with the existing trainer scripts.
class Flux(nn.Module):
    def __init__(self, params: FluxParams): # Accepts the existing FluxParams
        super().__init__()
        self.params_dto = params
        
        # Common initializations based on self.params_dto
        self.in_channels = self.params_dto.in_channels
        self.out_channels = self.in_channels
        if self.params_dto.hidden_size % self.params_dto.num_heads != 0:
            raise ValueError(f"Hidden size {self.params_dto.hidden_size} must be divisible by num_heads {self.params_dto.num_heads}")
        pe_dim = self.params_dto.hidden_size // self.params_dto.num_heads
        if sum(self.params_dto.axes_dim) != pe_dim:
            raise ValueError(f"axes_dim sum {sum(self.params_dto.axes_dim)} != pe_dim {pe_dim}")
        
        self.hidden_size = self.params_dto.hidden_size
        self.num_heads = self.params_dto.num_heads

        self.pe_embedder = EmbedND(dim=pe_dim, theta=self.params_dto.theta, axes_dim=self.params_dto.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.txt_in = nn.Linear(self.params_dto.context_in_dim, self.hidden_size, bias=True)

        # --- Component Initialization based on model type (Chroma vs. Dev/Schnell) ---
        if not params.use_modulation: # This is the Chroma path
            self.time_in = None
            self.vector_in = None
            self.guidance_in = nn.Identity() 

            if params.approximator_config and params.use_distilled_guidance_layer:
                app_cfg = params.approximator_config
                out_dim_for_approx = app_cfg.out_dim_per_mod_vector if app_cfg.out_dim_per_mod_vector != 0 else self.params_dto.hidden_size
                
                self.distilled_guidance_layer = Approximator(
                    app_cfg.in_dim, out_dim_for_approx, app_cfg.hidden_dim, app_cfg.n_layers
                )
                self.register_buffer('mod_index', torch.arange(app_cfg.mod_index_length), persistent=False)
                self.mod_index_length = app_cfg.mod_index_length
                logger.info(f"Chroma Path: Initialized self.distilled_guidance_layer as Approximator with {app_cfg.n_layers} layers for block modulation.")
            else:
                self.distilled_guidance_layer = None
                logger.warning("Chroma Path: Not initializing distilled_guidance_layer (Approximator). Check params.")
            
            self.modulation_approximator = None # Explicitly None for Chroma path

        else: # Dev/Schnell Path (uses internal modulation within blocks)
            self.time_in = MLPEmbedder(256, params.hidden_size) if params.use_time_embed else None
            self.vector_in = MLPEmbedder(params.vec_in_dim, params.hidden_size) if params.use_vector_embed else None
            self.guidance_in = MLPEmbedder(256, params.hidden_size) if params.guidance_embed else nn.Identity()
            self.distilled_guidance_layer = None 
            self.modulation_approximator = None 

        self.double_blocks = nn.ModuleList(
            [DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=self.params_dto.mlp_ratio, 
                               qkv_bias=self.params_dto.qkv_bias, 
                               use_modulation=params.use_modulation, # Pass to block
                               has_main_norms=params.double_block_has_main_norms if hasattr(params, 'double_block_has_main_norms') else True) # Pass to block
             for _ in range(self.params_dto.depth)]
        )
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=self.params_dto.mlp_ratio, use_modulation=params.use_modulation) # Pass to block
             for _ in range(self.params_dto.depth_single_blocks)]
        )
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

        self.blocks_to_swap = None; self.offloader_double = None; self.offloader_single = None
        self.num_double_blocks = len(self.double_blocks); self.num_single_blocks = len(self.single_blocks)
        self.gradient_checkpointing = False; self.cpu_offload_checkpointing = False
        
    @property
    def device(self): return next(self.parameters()).device
    @property
    def dtype(self): return next(self.parameters()).device

    def enable_gradient_checkpointing(self, cpu_offload: bool = False): 
        self.gradient_checkpointing = True; self.cpu_offload_checkpointing = cpu_offload
        # Enable GC on submodules if they support it (MLPEmbedders for Dev/Schnell, blocks themselves)
        for submodule_list in [self.double_blocks, self.single_blocks]:
            for block in submodule_list:
                if hasattr(block, 'enable_gradient_checkpointing'):
                    block.enable_gradient_checkpointing(cpu_offload=cpu_offload)
        for emb_layer in [self.time_in, self.vector_in, self.guidance_in]:
            if emb_layer is not None and hasattr(emb_layer, 'enable_gradient_checkpointing'):
                emb_layer.enable_gradient_checkpointing() # MLPEmbedder has this
        if hasattr(self, 'distilled_guidance_layer') and self.distilled_guidance_layer is not None and \
           hasattr(self.distilled_guidance_layer, 'layers'): # Approximator's layers are MLPEmbedders
            for layer in self.distilled_guidance_layer.layers:
                 if hasattr(layer,'enable_gradient_checkpointing'): layer.enable_gradient_checkpointing()

        logger.info(f"Flux (Chroma-mode): GC enabled. CPU offload: {cpu_offload}.")

    def disable_gradient_checkpointing(self): 
        self.gradient_checkpointing = False; self.cpu_offload_checkpointing = False
        for submodule_list in [self.double_blocks, self.single_blocks]:
            for block in submodule_list:
                if hasattr(block, 'disable_gradient_checkpointing'):
                    block.disable_gradient_checkpointing()
        for emb_layer in [self.time_in, self.vector_in, self.guidance_in]:
            if emb_layer is not None and hasattr(emb_layer, 'disable_gradient_checkpointing'):
                emb_layer.disable_gradient_checkpointing()
        if hasattr(self, 'distilled_guidance_layer') and self.distilled_guidance_layer is not None and \
           hasattr(self.distilled_guidance_layer, 'layers'):
            for layer in self.distilled_guidance_layer.layers:
                 if hasattr(layer,'disable_gradient_checkpointing'): layer.disable_gradient_checkpointing()
        logger.info(f"Flux (Chroma-mode): GC disabled.")
    
    def enable_block_swap(self, num_blocks: int, device: torch.device):
        if self.num_double_blocks == 0 and self.num_single_blocks == 0: logger.info("Flux (Chroma-mode): No blocks for swap."); return
        self.blocks_to_swap=num_blocks
        db_s = min(num_blocks // 2, max(0, self.num_double_blocks - 2)) if self.num_double_blocks > 0 else 0
        sb_s = min(num_blocks - db_s, max(0, self.num_single_blocks - 2)) if self.num_single_blocks > 0 else 0
        if db_s > 0 and self.num_double_blocks > 0 : self.offloader_double=custom_offloading_utils.ModelOffloader(self.double_blocks,self.num_double_blocks,db_s,device)
        if sb_s > 0 and self.num_single_blocks > 0: self.offloader_single=custom_offloading_utils.ModelOffloader(self.single_blocks,self.num_single_blocks,sb_s,device)
        logger.info(f"Flux (Chroma-mode): Block swap up to {num_blocks} (D:{db_s}, S:{sb_s}).")

    def prepare_block_swap_before_forward(self):
        if not self.blocks_to_swap: return
        if self.offloader_double: self.offloader_double.prepare_block_devices_before_forward(self.double_blocks)
        if self.offloader_single: self.offloader_single.prepare_block_devices_before_forward(self.single_blocks)
    
    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap: 
            s_db,s_sb = None, None
            if self.offloader_double: s_db=self.double_blocks; self.double_blocks=None
            if self.offloader_single: s_sb=self.single_blocks; self.single_blocks=None
        self.to(device)
        if self.blocks_to_swap: 
            if self.offloader_double and s_db is not None: self.double_blocks=s_db
            if self.offloader_single and s_sb is not None: self.single_blocks=s_sb

    def forward(
        self,
        img: Tensor, img_ids: Tensor, txt: Tensor, txt_ids_original_t5: Tensor, # Renamed txt_ids to clarify
        timesteps: Tensor, y: Optional[Tensor], guidance: Optional[Tensor] = None, 
        txt_attention_mask: Optional[Tensor] = None, # This is (B, L_txt_orig_tokens)
        **kwargs # Catches attn_padding if passed by official Chroma logic style
    ) -> Tensor:
        
        attn_padding = kwargs.get('attn_padding', 1) # From official Chroma
        log_tensor_stats(img, "Flux_fwd_SOT_noisy_input_packed", logger_obj=logger)
        
        img_p = self.img_in(img)
        txt_f = self.txt_in(txt)
        log_tensor_stats(img_p, "Flux_fwd_after_img_in", logger_obj=logger)
        log_tensor_stats(txt_f, "Flux_fwd_after_txt_in", logger_obj=logger)
        
        _bs = img.shape[0]
        internal_cond_vec = torch.zeros(_bs, self.hidden_size, device=img.device, dtype=img.dtype) # Always zeros for Chroma
        log_tensor_stats(internal_cond_vec, "Flux_fwd_internal_cond_vec_final (Chroma: Zeros)", logger_obj=logger)
        
        mod_vectors_dict = {} # Default to empty if no modulator
        if not self.params_dto.use_modulation and hasattr(self, 'distilled_guidance_layer') and self.distilled_guidance_layer is not None:
            app_cfg = self.params_dto.approximator_config
            if app_cfg:
                ts_emb_approx = timestep_embedding(timesteps.reshape(-1), 16)
                guid_val_approx = guidance if guidance is not None else torch.zeros_like(timesteps.reshape(-1))
                guid_emb_approx = timestep_embedding(guid_val_approx.reshape(-1), 16)
                
                mod_idx_device = self.mod_index.to(device=img.device, non_blocking=True)
                mod_idx_batched = mod_idx_device.unsqueeze(0).repeat(_bs, 1)
                mod_idx_emb = timestep_embedding(mod_idx_batched, 32)
                
                ts_guid_combined_emb_approx = torch.cat([ts_emb_approx, guid_emb_approx], dim=-1)
                ts_guid_emb_approx_repeated = ts_guid_combined_emb_approx.unsqueeze(1).repeat(1, self.mod_index_length, 1)
                approx_in = torch.cat([ts_guid_emb_approx_repeated, mod_idx_emb], dim=-1)

                mod_vecs = self.distilled_guidance_layer(approx_in)
                log_tensor_stats(mod_vecs, "Flux_fwd_mod_vecs_from_distilled_guidance_layer", logger_obj=logger)
                mod_vectors_dict = distribute_modulations(mod_vecs, self.num_double_blocks, self.num_single_blocks, final_layer_mod=True)
            else: logger.warning("Chroma path: distilled_guidance_layer exists but no approximator_config.")
        elif not self.params_dto.use_modulation:
            logger.warning("Chroma path: no distilled_guidance_layer (Approximator). Using default modulations.")

        B, L_txt_calc, _ = txt_f.shape
        # PE uses original T5 token IDs (txt_ids_original_t5) for sequence length, not feature length.
        # This was a subtle bug in previous adapter. The official Chroma.forward uses txt_ids (original tokens) for PE.
        # However, all_pos_ids needs to match the sequence length of the features (txt_f, img_p) that pe is applied to.
        # The official code uses txt_ids for PE, which seems to imply that RoPE is applied *before* features are projected,
        # or that txt_ids' length matches txt_f's length. This is complex.
        # For now, using L_txt_calc from txt_f for consistency with feature dimensions.
        txt_seq_positions = torch.arange(L_txt_calc, device=txt_f.device, dtype=img_ids.dtype).unsqueeze(0).expand(B, -1)
        txt_pos_ids_3d = torch.zeros(B, L_txt_calc, 3, device=txt_f.device, dtype=img_ids.dtype); txt_pos_ids_3d[..., 0] = txt_seq_positions
        all_pos_ids = torch.cat((txt_pos_ids_3d, img_ids), dim=1)
        pe = self.pe_embedder(all_pos_ids)
        
        current_img_features = img_p; current_txt_features = txt_f

        L_img_calc = current_img_features.shape[1]; L_combined = L_txt_calc + L_img_calc
        combined_mask_for_attn = None # This is for scaled_dot_product_attention (True means MASK)
        if txt_attention_mask is not None: # (B, L_txt_original_tokens)
            # Align with L_txt_calc from T5 features
            aligned_txt_mask = txt_attention_mask
            if txt_attention_mask.shape[1] != L_txt_calc:
                 logger.debug(f"Aligning txt_attention_mask from {txt_attention_mask.shape[1]} to {L_txt_calc}")
                 if txt_attention_mask.shape[1] > L_txt_calc: aligned_txt_mask = txt_attention_mask[:, :L_txt_calc]
                 else: 
                     padding = torch.zeros(B, L_txt_calc - txt_attention_mask.shape[1], device=txt_attention_mask.device, dtype=txt_attention_mask.dtype)
                     aligned_txt_mask = torch.cat([txt_attention_mask, padding], dim=1)
            
            txt_mask_w_padding = modify_mask_to_attend_padding(aligned_txt_mask.float(), L_txt_calc, attn_padding).bool() # Ensure bool
            img_component_mask = torch.ones(B, L_img_calc, device=txt_mask_w_padding.device, dtype=torch.bool) # Image part is all True (attend)
            combined_sequence_mask_1D_attend = torch.cat([txt_mask_w_padding, img_component_mask], dim=1) # True means attend
            # SDPA expects True to MASK. So invert. (B, L_combined)
            mask_to_ignore_1D = ~combined_sequence_mask_1D_attend
            # Expand for SDPA: (B, L_q, L_k) or (B, H, L_q, L_k). SDPA can broadcast (B, 1, L_k).
            combined_mask_for_attn = mask_to_ignore_1D.unsqueeze(1) # (B, 1, L_combined) - keys to ignore

        for i, blk in enumerate(self.double_blocks):
            log_tensor_stats(current_img_features, f"Flux_fwd_DB_{i}_img_INPUT", logger_obj=logger)
            log_tensor_stats(current_txt_features, f"Flux_fwd_DB_{i}_txt_INPUT", logger_obj=logger)
            
            img_mod_list = mod_vectors_dict.get(f"double_blocks.{i}.img_mod.lin")
            txt_mod_list = mod_vectors_dict.get(f"double_blocks.{i}.txt_mod.lin")
            block_distill_vec = [img_mod_list, txt_mod_list] if img_mod_list and txt_mod_list else None # Ensure both are found

            if self.blocks_to_swap and self.offloader_double: self.offloader_double.wait_for_block(i)
            
            # Call with official signature for DoubleStreamBlock
            # It expects vec (our internal_cond_vec) to be unused by its logic, using distill_vec instead
            # Our new DoubleStreamBlock only takes img, txt, pe, distill_vec, mask
            if self.gradient_checkpointing and self.training:
                 current_img_features, current_txt_features = torch_checkpoint(
                    blk, current_img_features, current_txt_features, pe, block_distill_vec, combined_mask_for_attn, 
                    use_reentrant=False
                )
            else:
                current_img_features, current_txt_features = blk(
                    img=current_img_features, txt=current_txt_features, pe=pe, 
                    distill_vec=block_distill_vec, mask=combined_mask_for_attn
                )
            log_tensor_stats(current_img_features, f"Flux_fwd_DB_{i}_img_OUTPUT", logger_obj=logger)
            log_tensor_stats(current_txt_features, f"Flux_fwd_DB_{i}_txt_OUTPUT", logger_obj=logger)
            if torch.isnan(current_img_features).any() or torch.isnan(current_txt_features).any(): logger.error(f"NaN after DSB {i}!"); break
            if self.blocks_to_swap and self.offloader_double: self.offloader_double.submit_move_blocks(self.double_blocks,i)

        if torch.isnan(current_img_features).any() or torch.isnan(current_txt_features).any():
            return torch.full((img.shape[0], img_p.shape[1], self.out_channels), float('nan'), device=self.device, dtype=self.dtype)

        sb_in = torch.cat((current_txt_features, current_img_features), dim=1)
        
        for i, blk in enumerate(self.single_blocks):
            log_tensor_stats(sb_in, f"Flux_fwd_SB_{i}_INPUT", logger_obj=logger)
            single_mod_for_block = mod_vectors_dict.get(f"single_blocks.{i}.modulation.lin")
            if self.blocks_to_swap and self.offloader_single: self.offloader_single.wait_for_block(i)
            # Official SingleStreamBlock forward: x, pe, distill_vec, mask
            if self.gradient_checkpointing and self.training:
                sb_in = torch_checkpoint(blk, sb_in, pe, single_mod_for_block, combined_mask_for_attn, use_reentrant=False)
            else:
                sb_in = blk(x=sb_in, pe=pe, distill_vec=single_mod_for_block, mask=combined_mask_for_attn)
            log_tensor_stats(sb_in, f"Flux_fwd_SB_{i}_OUTPUT", logger_obj=logger)
            if torch.isnan(sb_in).any(): logger.error(f"NaN after SSB {i}!"); break
            if self.blocks_to_swap and self.offloader_single: self.offloader_single.submit_move_blocks(self.single_blocks,i)

        if torch.isnan(sb_in).any():
            return torch.full((img.shape[0], img_p.shape[1], self.out_channels), float('nan'), device=self.device, dtype=self.dtype)

        img_f_final = sb_in[:, L_txt_calc:] 
        log_tensor_stats(img_f_final, "Flux_fwd_img_f_final_PRE_LASTLAYER", logger_obj=logger)
        
        if self.training and self.cpu_offload_checkpointing and self.blocks_to_swap: img_f_final = img_f_final.to(self.device)
            
        final_mod_for_lastlayer = mod_vectors_dict.get("final_layer.adaLN_modulation.1")
        if final_mod_for_lastlayer is None: # This means adaLN for final layer is pruned
            logger.debug("Modulation for final_layer (final_layer.adaLN_modulation.1) not found. Using zeros for shift/scale.")
            dummy_final_mod_shape = (img_f_final.shape[0], 1, self.hidden_size)
            final_mod_for_lastlayer = [
                torch.zeros(dummy_final_mod_shape, device=img_f_final.device, dtype=img_f_final.dtype),
                torch.zeros(dummy_final_mod_shape, device=img_f_final.device, dtype=img_f_final.dtype) # Scale 0 -> (1+0)*x = x
            ]
        
        # Official LastLayer takes distill_vec=[shift, scale]
        img_out = self.final_layer(img_f_final, distill_vec=final_mod_for_lastlayer) 
        # log_tensor_stats(img_out, "Flux_fwd_output_from_final_layer_img_out", logger_obj=logger) # Already in LastLayer
        return img_out

# --- End Flux (ChromaModel) definition ---

# --- AutoEncoder definitions from original flux_models.py ---
# ... (These should be kept as they were, I'll snip for brevity but ensure they are present in your file)
# @dataclass
# class AutoEncoderParams: ...
# def swish(...): ...
# class AttnBlock(...): ...
# class ResnetBlock(...): ...
# class Downsample(...): ...
# class Upsample(...): ...
# class Encoder(...): ... # Takes AutoEncoderParams
# class Decoder(...): ... # Takes AutoEncoderParams
# class DiagonalGaussian(...): ...
# class AutoEncoder(...): ... # Takes AutoEncoderParams
# --- End AutoEncoder definitions ---


# --- Functions to get params, used by flux_utils.py ---
def flux_chroma_params(): # Renamed back for flux_utils compatibility
    params = FluxParams( # Use the main FluxParams dataclass
        in_channels=64, context_in_dim=4096, hidden_size=3072,
        mlp_ratio=4.0, num_heads=24, depth=19, depth_single_blocks=38,
        axes_dim=[16, 56, 56], theta=10_000, qkv_bias=True,
        # Chroma-specific settings
        guidance_embed=False,       # No separate guidance MLP for internal_cond_vec
        use_modulation=False,       # Uses external Approximator
        use_distilled_guidance_layer=True, # Signifies the main Approximator is active
        use_time_embed=False,       # No separate time MLP for internal_cond_vec
        use_vector_embed=False,     # No separate vector MLP for internal_cond_vec
        # approvimator_config is now part of FluxParams itself
        approximator_config=ApproximatorParams( # Configure the main Approximator
            in_dim=64, 
            out_dim_per_mod_vector=3072, # Should match hidden_size
            hidden_dim=5120, 
            n_layers=5, # Chroma uses 5 layers for its Approximator
            mod_index_length=344
        )
        # Fields like vec_in_dim, distilled_guidance_dim become unused for Chroma config
        # double_block_has_main_norms is handled by using official block definitions
    )
    return params

# Keep these for potential future use or if flux_utils still references them for non-Chroma
def flux1_dev_params(depth_double=19, depth_single=38):
    return FluxParams(in_channels=64, vec_in_dim=768, context_in_dim=4096, hidden_size=3072,
        mlp_ratio=4.0, num_heads=24, depth=depth_double, depth_single_blocks=depth_single,
        axes_dim=[16, 56, 56], theta=10_000, qkv_bias=True, 
        guidance_embed=True, use_modulation=True, 
        use_distilled_guidance_layer=False, # No main Approximator for Dev
        approximator_config=None,      # No main Approximator for Dev
        use_time_embed=True, use_vector_embed=True, 
        double_block_has_main_norms=True)

def flux1_schnell_params(depth_double=19, depth_single=38):
     return FluxParams(in_channels=64, vec_in_dim=768, context_in_dim=4096, hidden_size=3072,
        mlp_ratio=4.0, num_heads=24, depth=depth_double, depth_single_blocks=depth_single,
        axes_dim=[16, 56, 56], theta=10_000, qkv_bias=True, 
        guidance_embed=False, use_modulation=True, # Schnell has internal mod, no guidance_in MLP
        use_distilled_guidance_layer=False,
        approximator_config=None,
        use_time_embed=True, use_vector_embed=True, 
        double_block_has_main_norms=True)


# model_configs for flux_utils.py - ensure it uses the functions that return FluxParams
model_configs = {
    "dev": {"params_fn": flux1_dev_params}, # Store function to call
    "schnell": {"params_fn": flux1_schnell_params},
    "chroma": {"params_fn": flux_chroma_params} # This will return FluxParams configured for Chroma
}
# Add default AE params separately if needed by flux_utils.load_ae
_original_configs_for_ae = { # Used by load_ae if specific model config doesn't provide ae_params
    "default": AutoEncoderParams() # Uses defaults from AutoEncoderParams dataclass
}
# --- AutoEncoder definitions ---
@dataclass
class AutoEncoderParams:
    resolution: int = 256
    in_channels: int = 3
    ch: int = 128
    out_ch: int = 3
    ch_mult: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    num_res_blocks: int = 2
    z_channels: int = 16 
    scale_factor: float = 0.3611 
    shift_factor: float = 0.1159

def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)

class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w)
        v = v.permute(0, 2, 1)

        sim = torch.bmm(q, k) * (c**-0.5)
        attn = torch.softmax(sim, dim=-1)

        h_ = torch.bmm(attn, v)
        h_ = h_.permute(0, 2, 1).reshape(b, c, h, w)
        h_ = self.proj_out(h_)

        return x + h_

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = nn.Identity()


    def forward(self, x: Tensor) -> Tensor:
        h = x
        h = self.norm1(h)
        h = swish(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = swish(h)
        h = self.dropout(h)
        h = self.conv2(h)

        x = self.nin_shortcut(x)
        return x + h

class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        if self.with_conv:
            pad = (0, 1, 0, 1) 
            x = nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool = True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Encoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.ch = params.ch
        self.num_resolutions = len(params.ch_mult)
        self.num_res_blocks = params.num_res_blocks
        self.resolution = params.resolution
        self.in_channels = params.in_channels
        dropout = 0.0 # Standard VAEs often don't use dropout here

        self.conv_in = nn.Conv2d(self.in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        
        curr_res = self.resolution
        in_ch_mult = (1,) + tuple(params.ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block_in = self.ch * in_ch_mult[i_level]
            block_out = self.ch * params.ch_mult[i_level]
            
            block_list = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                block_list.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out # Update for next resblock in same level
            
            down_level = nn.Module()
            down_level.block = block_list
            if i_level != self.num_resolutions - 1:
                down_level.downsample = Downsample(block_in, with_conv=True)
                curr_res //= 2
            self.down.append(down_level)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * params.z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.ch = params.ch
        self.num_resolutions = len(params.ch_mult)
        self.num_res_blocks = params.num_res_blocks
        self.resolution = params.resolution
        self.out_channels = params.out_ch # Corrected to params.out_ch
        dropout = 0.0

        block_in = self.ch * params.ch_mult[self.num_resolutions - 1]
        self.conv_in = nn.Conv2d(params.z_channels, block_in, kernel_size=3, stride=1, padding=1)
        
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, dropout=dropout)
        
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block_list = nn.ModuleList()
            block_out = self.ch * params.ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1): # Decoders often have +1 block per level
                block_list.append(ResnetBlock(in_channels=block_in, out_channels=block_out, dropout=dropout))
                block_in = block_out
            
            up_level = nn.Module()
            up_level.block = block_list
            if i_level != 0:
                up_level.upsample = Upsample(block_in, with_conv=True)
            self.up.insert(0, up_level) 

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, self.out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h

class DiagonalGaussian(nn.Module):
    def __init__(self, sample: bool = True, chunk_dim: int = 1):
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim
    def forward(self, z: Tensor) -> Tensor:
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            # exp(0.5 * logvar) = std
            # 공식 코드에서는 종종 torch.exp(logvar).sqrt() 또는 torch.exp(logvar * 0.5)
            return mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return mean

class AutoEncoder(nn.Module):
    def __init__(self, params: AutoEncoderParams):
        super().__init__()
        self.params = params # Store params
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.quant_conv = nn.Conv2d(2 * params.z_channels, 2 * params.z_channels, 1) # As in Diffusers KL
        self.post_quant_conv = nn.Conv2d(params.z_channels, params.z_channels, 1)    # As in Diffusers KL
        self.logvar = nn.Parameter(torch.zeros(params.z_channels, 1, 1)) # Trainable logvar if needed, or use from encoder
        self.reg = DiagonalGaussian() # Kept from original

    @property
    def device(self) -> torch.device: return next(self.parameters()).device
    @property
    def dtype(self) -> torch.dtype: return next(self.parameters()).dtype
    
    def encode(self, x: Tensor) -> Tensor: # Modified to be more like Diffusers KL VAE
        h = self.encoder(x)
        moments = self.quant_conv(h)
        # posterior = DiagonalGaussianDistribution(moments) # Diffusers style
        # For compatibility with existing DiagonalGaussian:
        # DiagonalGaussian expects mean and logvar to be chunked from its input 'z'
        # Here, moments directly contains mean and logvar (or pre-logvar)
        # If self.reg is DiagonalGaussian, it expects input for chunking.
        # The FLUX AE had self.reg(self.encoder(x)) where encoder output 2*z_channels.
        # So, self.encoder(x) already outputs for DiagonalGaussian.
        # The quant_conv is extra if encoder already does this.
        # Let's assume encoder output is 2*z_channels for DiagonalGaussian
        
        # Path 1: If encoder outputs 2*z for DiagonalGaussian
        # z_for_reg = self.encoder(x)
        # latents = self.reg(z_for_reg) # This samples

        # Path 2: Diffusers style, with DiagonalGaussian adapted or replaced
        # For now, let's keep it closer to your original self.reg(self.encoder(x)) path
        # and assume the quant_conv/post_quant_conv might be for a different VAE flavor.
        # The FLUX AE model is simpler:
        encoded_doubled_latents = self.encoder(x) # Outputs (B, 2*Z, H, W)
        sampled_latents = self.reg(encoded_doubled_latents) # Outputs (B, Z, H, W) after sampling

        return self.params.scale_factor * (sampled_latents - self.params.shift_factor)

    def decode(self, z: Tensor) -> Tensor:
        # Reverse scale and shift
        z_unshifted_unscaled = z / self.params.scale_factor + self.params.shift_factor
        # The post_quant_conv is part of Diffusers KL VAE, not obviously in original simple FLUX AE.
        # If self.decoder expects Z channels directly:
        return self.decoder(z_unshifted_unscaled)

    def forward(self, x: Tensor) -> Tensor: 
        return self.decode(self.encode(x))