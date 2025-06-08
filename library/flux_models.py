# modified for chroma

# copy from FLUX repo: https://github.com/black-forest-labs/flux
# license: Apache-2.0 License


import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field # Added field for default_factory
from typing import Any, Dict, List, Optional, Union, Tuple, TYPE_CHECKING
import logging

from library import utils # Assuming this is your utils.py for setup_logging
from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

import torch
from einops import rearrange
from torch import Tensor, nn
import torch.nn.functional as F # Added for F.rms_norm
from torch.utils.checkpoint import checkpoint as torch_checkpoint # Renamed to avoid conflict

from library import custom_offloading_utils # Keep if block swapping is still desired
from .math import attention, rope

logger = logging.getLogger(__name__)

# from library import flux_models # Removed self-import
from library.utils import load_safetensors

if TYPE_CHECKING:
    pass


# region autoencoder - Kept from original, assumed compatible


@dataclass
class AutoEncoderParams:
    resolution: int
    in_channels: int
    ch: int
    out_ch: int
    ch_mult: list[int]
    num_res_blocks: int
    z_channels: int
    scale_factor: float
    shift_factor: float


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
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

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
    def __init__(self, resolution: int, in_channels: int, ch: int, ch_mult: list[int], num_res_blocks: int, z_channels: int):
        super().__init__()
        self.ch, self.num_resolutions, self.num_res_blocks = ch, len(ch_mult), num_res_blocks
        self.resolution, self.in_channels = resolution, in_channels
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)
        curr_res, in_ch_mult = resolution, (1,) + tuple(ch_mult)
        self.in_ch_mult, self.down = in_ch_mult, nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block, attn = nn.ModuleList(), nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]; block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out)); block_in = block_out
            down = nn.Module(); down.block, down.attn = block, attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in); curr_res //= 2
            self.down.append(down)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1])
                if len(self.down[i_level].attn) > 0: h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1: hs.append(self.down[i_level].downsample(hs[-1]))
        h = hs[-1]
        h = self.mid.block_1(h); h = self.mid.attn_1(h); h = self.mid.block_2(h)
        h = self.norm_out(h); h = swish(h); h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, ch: int, out_ch: int, ch_mult: list[int], num_res_blocks: int, in_channels: int, resolution: int, z_channels: int):
        super().__init__()
        self.ch, self.num_resolutions, self.num_res_blocks = ch, len(ch_mult), num_res_blocks
        self.resolution, self.in_channels = resolution, in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block, attn = nn.ModuleList(), nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out)); block_in = block_out
            up = nn.Module(); up.block, up.attn = block, attn
            if i_level != 0: up.upsample = Upsample(block_in); curr_res *= 2
            self.up.insert(0, up)
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h); h = self.mid.attn_1(h); h = self.mid.block_2(h)
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0: h = self.up[i_level].attn[i_block](h)
            if i_level != 0: h = self.up[i_level].upsample(h)
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
        self.encoder = Encoder(params.resolution, params.in_channels, params.ch, params.ch_mult, params.num_res_blocks, params.z_channels)
        self.decoder = Decoder(params.ch, params.out_ch, params.ch_mult, params.num_res_blocks, params.in_channels, params.resolution, params.z_channels)
        self.reg = DiagonalGaussian()
        self.scale_factor, self.shift_factor = params.scale_factor, params.shift_factor
    @property
    def device(self) -> torch.device: return next(self.parameters()).device
    @property
    def dtype(self) -> torch.dtype: return next(self.parameters()).dtype
    def encode(self, x: Tensor) -> Tensor: return self.scale_factor * (self.reg(self.encoder(x)) - self.shift_factor)
    def decode(self, z: Tensor) -> Tensor: return self.decoder(z / self.scale_factor + self.shift_factor)
    def forward(self, x: Tensor) -> Tensor: return self.decode(self.encode(x))

# endregion


# --- Helper for logging tensor stats (from original file) ---
def log_tensor_stats(tensor: Optional[torch.Tensor], name: str = "tensor", logger_obj: Optional[logging.Logger] = None):
    if logger_obj is None:
        logger_obj = logging.getLogger(__name__)

    if tensor is not None:
        has_elements = tensor.numel() > 0
        min_val = tensor.min().item() if has_elements else 'N/A (0 elements)'
        max_val = tensor.max().item() if has_elements else 'N/A (0 elements)'
        mean_val = tensor.mean().item() if has_elements else 'N/A (0 elements)'
        
        logger_obj.debug(
            f"{name}: shape {tensor.shape}, dtype {tensor.dtype}, "
            f"min {min_val}, "
            f"max {max_val}, "
            f"mean {mean_val}, "
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


# --- Official Chroma Layer Definitions ---
class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )
        return emb.unsqueeze(1)

def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    original_shape = t.shape
    if t.ndim == 0:
        t = t.unsqueeze(0)
    elif t.ndim > 1 and t.numel() == t.shape[0]*t.shape[1]:
        t = t.reshape(-1)
    elif t.ndim > 1:
        t = t.reshape(-1)


    t_scaled = time_factor * t
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(t_scaled.device)

    args = t_scaled[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

    if len(original_shape) > 1 and embedding.shape[0] == original_shape[0] * original_shape[1] and original_shape[0] != 0 and original_shape[1] != 0:
        embedding = embedding.view(*original_shape, dim)
    elif len(original_shape) == 1 and original_shape[0] == 0:
        embedding = torch.empty(*original_shape, dim, device=embedding.device, dtype=embedding.dtype)

    return embedding.float()


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    @property
    def device(self): return next(self.parameters()).device
    def forward(self, x: Tensor) -> Tensor: return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, use_compiled: bool = False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: Tensor):
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, self.scale.shape, weight=self.scale, eps=1e-6)
        else:
            x_dtype = x.dtype
            x = x.float()
            rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
            return ((x * rrms) * self.scale.float()).to(dtype=x_dtype)


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor

def distribute_modulations(tensor: torch.Tensor, num_double_blocks: int, num_single_blocks: int, final_layer_mod: bool = True) -> Dict[str, Any]:
    batch_size, total_vectors, dim_per_vector = tensor.shape
    block_dict = {}
    idx = 0

    expected_vectors = (num_double_blocks * 2 * 3) + (num_double_blocks * 2 * 3) + \
                       (num_single_blocks * 1 * 3)
    if final_layer_mod:
        expected_vectors += 2

    if total_vectors < expected_vectors:
        logger.warning(f"distribute_modulations: Not enough vectors. Have {total_vectors}, need {expected_vectors}. Modulation might be incomplete.")

    def _get_slice(num_vecs_needed):
        nonlocal idx
        end_idx = idx + num_vecs_needed
        if end_idx > total_vectors:
            if num_vecs_needed == 3:
                s = torch.zeros(batch_size, 1, dim_per_vector, device=tensor.device, dtype=tensor.dtype)
                sc = torch.zeros(batch_size, 1, dim_per_vector, device=tensor.device, dtype=tensor.dtype)
                g = torch.ones(batch_size, 1, dim_per_vector, device=tensor.device, dtype=tensor.dtype)
                return s, sc, g
            elif num_vecs_needed == 2:
                s = torch.zeros(batch_size, 1, dim_per_vector, device=tensor.device, dtype=tensor.dtype)
                sc = torch.zeros(batch_size, 1, dim_per_vector, device=tensor.device, dtype=tensor.dtype)
                return s, sc
            else:
                return [torch.zeros(batch_size, 1, dim_per_vector, device=tensor.device, dtype=tensor.dtype)] * num_vecs_needed
        
        slices = [tensor[:, i:i+1, :] for i in range(idx, end_idx)]
        idx = end_idx
        return slices[0], slices[1], slices[2] if num_vecs_needed == 3 else (slices[0], slices[1])


    for i in range(num_single_blocks):
        key = f"single_blocks.{i}.modulation.lin"
        s, sc, g = _get_slice(3)
        block_dict[key] = ModulationOut(shift=s, scale=sc, gate=g)

    for i in range(num_double_blocks):
        key_img = f"double_blocks.{i}.img_mod.lin"
        key_txt = f"double_blocks.{i}.txt_mod.lin"
        
        img_mods_block = []
        for _ in range(2):
            s, sc, g = _get_slice(3)
            img_mods_block.append(ModulationOut(shift=s, scale=sc, gate=g))
        block_dict[key_img] = img_mods_block

        txt_mods_block = []
        for _ in range(2):
            s, sc, g = _get_slice(3)
            txt_mods_block.append(ModulationOut(shift=s, scale=sc, gate=g))
        block_dict[key_txt] = txt_mods_block
        
    if final_layer_mod:
        key_final = "final_layer.adaLN_modulation.1"
        s, sc = _get_slice(2)
        block_dict[key_final] = [s, sc]

    if idx > total_vectors:
        logger.error(f"distribute_modulations: Overran vectors. Index {idx}, total {total_vectors}")
    elif idx < expected_vectors and total_vectors >= expected_vectors :
        logger.warning(f"distribute_modulations: Underran expected vectors. Index {idx}, expected {expected_vectors}")


    return block_dict


class Approximator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers=4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim, bias=True)
        self.layers = nn.ModuleList(
            [MLPEmbedder(hidden_dim, hidden_dim) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList([RMSNorm(hidden_dim) for _ in range(n_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    @property
    def device(self): return next(self.parameters()).device
    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        for layer, norm_layer in zip(self.layers, self.norms):
            x = x + layer(norm_layer(x))
        x = self.out_proj(x)
        return x


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int, use_compiled: bool = False):
        super().__init__()
        self.query_norm = RMSNorm(dim, use_compiled=use_compiled)
        self.key_norm = RMSNorm(dim, use_compiled=use_compiled)
        
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q_normed = self.query_norm(q)
        k_normed = self.key_norm(k)
        return q_normed.to(v.dtype), k_normed.to(v.dtype)


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, use_compiled: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim, use_compiled=use_compiled)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x: Tensor, pe: Tensor, mask: Optional[Tensor]=None) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe, mask=mask)
        x = self.proj(x)
        return x


def _modulation_shift_scale_fn(x, scale, shift):
    return (1 + scale) * x + shift

def _modulation_gate_fn(x, gate, gate_params):
    return x + gate * gate_params


class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False, use_compiled: bool = False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, use_compiled=use_compiled)
        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias, use_compiled=use_compiled)
        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        self.use_compiled = use_compiled

    @property
    def device(self): return next(self.parameters()).device

    def _compiled_modulation_shift_scale_fn(self, x, scale, shift):
        return torch.compile(_modulation_shift_scale_fn)(x, scale, shift)
    
    def _compiled_modulation_gate_fn(self, x, gate, gate_params):
        return torch.compile(_modulation_gate_fn)(x, gate, gate_params)

    def forward( self, img: Tensor, txt: Tensor, pe: Tensor,
                 distill_vec: list[list[ModulationOut]],
                 mask: Optional[Tensor]=None
                ) -> tuple[Tensor, Tensor]:
        
        (img_mod_attn, img_mod_mlp), (txt_mod_attn, txt_mod_mlp) = distill_vec[0], distill_vec[1]

        img_res, txt_res = img, txt


        img_modulated_for_qkv = _modulation_shift_scale_fn(self.img_norm1(img_res), img_mod_attn.scale, img_mod_attn.shift)
        img_qkv = self.img_attn.qkv(img_modulated_for_qkv)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        txt_modulated_for_qkv = _modulation_shift_scale_fn(self.txt_norm1(txt_res), txt_mod_attn.scale, txt_mod_attn.shift)
        txt_qkv = self.txt_attn.qkv(txt_modulated_for_qkv)
        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q_joint = torch.cat((txt_q, img_q), dim=2)
        k_joint = torch.cat((txt_k, img_k), dim=2)
        v_joint = torch.cat((txt_v, img_v), dim=2)
        
        attn_output_joint = attention(q_joint, k_joint, v_joint, pe=pe, mask=mask)
        
        txt_attn_output = attn_output_joint[:, :txt_res.shape[1]]
        img_attn_output = attn_output_joint[:, txt_res.shape[1]:]

        img_after_attn = _modulation_gate_fn(img_res, img_mod_attn.gate, self.img_attn.proj(img_attn_output))
        img_mlp_input = _modulation_shift_scale_fn(self.img_norm2(img_after_attn), img_mod_mlp.scale, img_mod_mlp.shift)
        img_final = _modulation_gate_fn(img_after_attn, img_mod_mlp.gate, self.img_mlp(img_mlp_input))

        txt_after_attn = _modulation_gate_fn(txt_res, txt_mod_attn.gate, self.txt_attn.proj(txt_attn_output))
        txt_mlp_input = _modulation_shift_scale_fn(self.txt_norm2(txt_after_attn), txt_mod_mlp.scale, txt_mod_mlp.shift)
        txt_final = _modulation_gate_fn(txt_after_attn, txt_mod_mlp.gate, self.txt_mlp(txt_mlp_input))

        return img_final, txt_final


class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, qk_scale: float | None = None, use_compiled: bool = False):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        
        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim, bias=True)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size, bias=True)

        self.norm = QKNorm(head_dim, use_compiled=use_compiled)
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.use_compiled = use_compiled

    @property
    def device(self): return next(self.parameters()).device
    
    def forward(self, x: Tensor, pe: Tensor, distill_vec: ModulationOut, mask: Optional[Tensor]=None) -> Tensor:
        mod = distill_vec
        x_res = x

        x_normed = self.pre_norm(x)
        x_modulated = _modulation_shift_scale_fn(x_normed, mod.scale, mod.shift)
        
        qkv_combined, mlp_in_features = torch.split(
            self.linear1(x_modulated), [3 * self.hidden_dim, self.mlp_hidden_dim], dim=-1
        )
        
        q, k, v = rearrange(qkv_combined, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        
        attn_output = attention(q, k, v, pe=pe, mask=mask)
        mlp_output_activated = self.mlp_act(mlp_in_features)
        
        combined_features = torch.cat((attn_output, mlp_output_activated), dim=-1)
        output_projection = self.linear2(combined_features)
        
        return _modulation_gate_fn(x_res, mod.gate, output_projection)


class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, use_compiled: bool = False):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        
    @property
    def device(self): return next(self.parameters()).device

    def forward(self, x: Tensor, distill_vec: list[Tensor]) -> Tensor:
        shift, scale = distill_vec[0], distill_vec[1]
        
        if shift.ndim == 3 and shift.shape[1] == 1: shift = shift.squeeze(1)
        if scale.ndim == 3 and scale.shape[1] == 1: scale = scale.squeeze(1)

        x_normed = self.norm_final(x)
        x_modulated = (1 + scale.unsqueeze(1)) * x_normed + shift.unsqueeze(1)
        x_out = self.linear(x_modulated)
        return x_out

# --- End Official Chroma Layer Definitions ---


# --- ChromaModel definition ---
@dataclass
class ChromaModelParams:
    in_channels: int = 64
    context_in_dim: int = 4096
    hidden_size: int = 3072
    mlp_ratio: float = 4.0
    num_heads: int = 24
    depth: int = 19
    depth_single_blocks: int = 38
    axes_dim: list[int] = field(default_factory=lambda: [16, 56, 56])
    theta: int = 10_000
    qkv_bias: bool = True
    approximator_in_dim: int = 64
    approximator_depth: int = 5
    approximator_hidden_size: int = 5120
    approximator_out_dim_per_mod_vector: int = 3072
    mod_index_length: int = 344

def get_chroma_model_params():
    return ChromaModelParams()

def modify_mask_to_attend_padding(mask: Tensor, max_len: int, num_to_attend: int) -> Tensor:
    mask = mask.bool()
    valid_lengths = mask.long().sum(dim=1)
    new_mask = mask.clone()
    for i in range(mask.shape[0]):
        current_len = valid_lengths[i]
        attend_until = min(current_len + num_to_attend, max_len)
        if current_len < max_len:
            new_mask[i, current_len:attend_until] = True
    return new_mask

class ChromaModel(nn.Module):
    def __init__(self, params: ChromaModelParams):
        super().__init__()
        self.params_dto = params
        self.in_channels = params.in_channels
        self.out_channels = self.in_channels

        if params.hidden_size % params.num_heads != 0:
            raise ValueError(f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}")
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"axes_dim sum {sum(params.axes_dim)} != pe_dim {pe_dim}")
        
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads

        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size, bias=True)

        out_dim_approx = params.approximator_out_dim_per_mod_vector if params.approximator_out_dim_per_mod_vector != 0 else params.hidden_size
        self.distilled_guidance_layer = Approximator(
            params.approximator_in_dim,
            out_dim_approx,
            params.approximator_hidden_size,
            params.approximator_depth,
        )
        self.register_buffer('mod_index', torch.arange(params.mod_index_length), persistent=False)
        self.mod_index_length = params.mod_index_length
        logger.info(f"ChromaModel: Initialized self.distilled_guidance_layer as Approximator with {params.approximator_depth} layers.")

        self.double_blocks = nn.ModuleList(
            [DoubleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio, qkv_bias=params.qkv_bias)
             for _ in range(params.depth)]
        )
        self.single_blocks = nn.ModuleList(
            [SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
             for _ in range(params.depth_single_blocks)]
        )
        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

        self.blocks_to_swap = None
        self.offloader_double = None
        self.offloader_single = None
        self.num_double_blocks = len(self.double_blocks)
        self.num_single_blocks = len(self.single_blocks)
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False # Added for compatibility with existing log
        
    @property
    def device(self): return next(self.parameters()).device
    @property
    def dtype(self): return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, cpu_offload: bool = False): 
        self.gradient_checkpointing = True
        self.cpu_offload_checkpointing = cpu_offload # Store for logging or other logic
        logger.info(f"ChromaModel: Gradient Checkpointing enabled. CPU offload: {cpu_offload}")
    def disable_gradient_checkpointing(self): 
        self.gradient_checkpointing = False
        self.cpu_offload_checkpointing = False
        logger.info(f"ChromaModel: Gradient Checkpointing disabled.")
    
    def enable_block_swap(self, num_blocks: int, device: torch.device):
        if self.num_double_blocks == 0 and self.num_single_blocks == 0:
            logger.info("ChromaModel: No blocks to enable swap for.")
            return
        self.blocks_to_swap=num_blocks
        db_s = min(num_blocks // 2, max(0, self.num_double_blocks - 2)) if self.num_double_blocks > 0 else 0
        sb_s = min(num_blocks - db_s, max(0, self.num_single_blocks - 2)) if self.num_single_blocks > 0 else 0
        
        if db_s > 0 and self.num_double_blocks > 0 : # ensure blocks exist
            self.offloader_double=custom_offloading_utils.ModelOffloader(self.double_blocks,self.num_double_blocks,db_s,device)
        if sb_s > 0 and self.num_single_blocks > 0: # ensure blocks exist
            self.offloader_single=custom_offloading_utils.ModelOffloader(self.single_blocks,self.num_single_blocks,sb_s,device)
        logger.info(f"ChromaModel: Block swap enabled. Swapping up to {num_blocks} blocks (D:{db_s}, S:{sb_s}).")

    def prepare_block_swap_before_forward(self):
        if not self.blocks_to_swap: return
        if self.offloader_double: self.offloader_double.prepare_block_devices_before_forward(self.double_blocks)
        if self.offloader_single: self.offloader_single.prepare_block_devices_before_forward(self.single_blocks)
    
    def move_to_device_except_swap_blocks(self, device: torch.device): # For compatibility
        if self.blocks_to_swap: 
            s_db,s_sb = None, None
            if self.offloader_double: 
                s_db=self.double_blocks
                self.double_blocks=None
            if self.offloader_single:
                s_sb=self.single_blocks
                self.single_blocks=None
        self.to(device)
        if self.blocks_to_swap: 
            if self.offloader_double and s_db is not None: self.double_blocks=s_db
            if self.offloader_single and s_sb is not None: self.single_blocks=s_sb


    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Optional[Tensor], 
        guidance: Optional[Tensor] = None,
        txt_attention_mask: Optional[Tensor] = None,
        attn_padding: int = 1
    ) -> Tensor:
        
        log_tensor_stats(img, "ChromaModel_fwd_SOT_noisy_input_packed", logger_obj=logger)
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError(f"Inputs img/txt must be 3D. Got img: {img.ndim}D, txt: {txt.ndim}D")
            
        img_p = self.img_in(img)
        log_tensor_stats(img_p, "ChromaModel_fwd_after_img_in", logger_obj=logger)
        
        txt_f = self.txt_in(txt)
        log_tensor_stats(txt_f, "ChromaModel_fwd_after_txt_in", logger_obj=logger)

        _bs = img.shape[0]
        ts_emb_approx = timestep_embedding(timesteps.reshape(-1), 16)
        guid_val_approx = guidance if guidance is not None else torch.zeros_like(timesteps.reshape(-1))
        guid_emb_approx = timestep_embedding(guid_val_approx.reshape(-1), 16)
        
        mod_idx_device = self.mod_index.to(device=img.device, non_blocking=True)
        mod_idx_batched = mod_idx_device.unsqueeze(0).repeat(_bs, 1)
        mod_idx_emb = timestep_embedding(mod_idx_batched, 32)
        
        ts_guid_combined_emb_approx = torch.cat([ts_emb_approx, guid_emb_approx], dim=-1)
        ts_guid_emb_approx_repeated = ts_guid_combined_emb_approx.unsqueeze(1).repeat(1, self.mod_index_length, 1)
        
        approx_in = torch.cat([ts_guid_emb_approx_repeated, mod_idx_emb], dim=-1)
        log_tensor_stats(approx_in, "ChromaModel_fwd_approx_in_to_distilled_guidance_layer", logger_obj=logger)

        mod_vectors = self.distilled_guidance_layer(approx_in)
        log_tensor_stats(mod_vectors, "ChromaModel_fwd_mod_vectors_from_distilled_guidance_layer", logger_obj=logger)

        mod_vectors_dict = distribute_modulations(
            mod_vectors, self.num_double_blocks, self.num_single_blocks, final_layer_mod=True
        )
        if mod_vectors_dict:
            logger.debug("ChromaModel_fwd: Successfully distributed modulations.")

        B, L_txt_calc, _ = txt_f.shape
        txt_seq_positions = torch.arange(L_txt_calc, device=txt_f.device, dtype=img_ids.dtype) 
        txt_seq_positions = txt_seq_positions.unsqueeze(0).expand(B, -1)
        txt_pos_ids_3d = torch.zeros(B, L_txt_calc, 3, device=txt_f.device, dtype=img_ids.dtype)
        txt_pos_ids_3d[..., 0] = txt_seq_positions
        all_pos_ids = torch.cat((txt_pos_ids_3d, img_ids), dim=1)
        pe = self.pe_embedder(all_pos_ids)
        
        current_img_features = img_p
        current_txt_features = txt_f

        L_img_calc = current_img_features.shape[1]
        L_combined = L_txt_calc + L_img_calc
        combined_mask_for_attn = None
        if txt_attention_mask is not None:
            if txt_attention_mask.shape[1] != L_txt_calc:
                 logger.warning(f"txt_attention_mask length {txt_attention_mask.shape[1]} != txt_f feature length {L_txt_calc}. Adjusting mask.")
                 if txt_attention_mask.shape[1] > L_txt_calc:
                     txt_attention_mask = txt_attention_mask[:, :L_txt_calc]
                 else:
                     padding = torch.zeros(B, L_txt_calc - txt_attention_mask.shape[1], device=txt_attention_mask.device, dtype=txt_attention_mask.dtype)
                     txt_attention_mask = torch.cat([txt_attention_mask, padding], dim=1)

            txt_mask_w_padding = modify_mask_to_attend_padding(txt_attention_mask, L_txt_calc, attn_padding)
            img_component_mask = torch.ones(B, L_img_calc, device=txt_mask_w_padding.device, dtype=txt_mask_w_padding.dtype)
            combined_sequence_mask_1D = torch.cat([txt_mask_w_padding, img_component_mask], dim=1)
            combined_mask_for_attn = ~(combined_sequence_mask_1D.bool().unsqueeze(1).expand(-1, L_combined, -1))

        for i, blk in enumerate(self.double_blocks):
            double_mod_for_block = [
                mod_vectors_dict.get(f"double_blocks.{i}.img_mod.lin"),
                mod_vectors_dict.get(f"double_blocks.{i}.txt_mod.lin")
            ]
            if self.blocks_to_swap and self.offloader_double: self.offloader_double.wait_for_block(i)
            
            if self.gradient_checkpointing and self.training:
                 current_img_features, current_txt_features = torch_checkpoint(
                    blk, current_img_features, current_txt_features, pe, double_mod_for_block, combined_mask_for_attn, 
                    use_reentrant=False
                )
            else:
                current_img_features, current_txt_features = blk(
                    img=current_img_features, txt=current_txt_features, pe=pe, 
                    distill_vec=double_mod_for_block, mask=combined_mask_for_attn
                )
            if torch.isnan(current_img_features).any() or torch.isnan(current_txt_features).any():
                logger.error(f"NaN detected after DoubleStreamBlock {i}!"); break
            if self.blocks_to_swap and self.offloader_double: self.offloader_double.submit_move_blocks(self.double_blocks,i)

        if torch.isnan(current_img_features).any() or torch.isnan(current_txt_features).any():
            nan_output_shape = (img.shape[0], img_p.shape[1], self.out_channels)
            return torch.full(nan_output_shape, float('nan'), device=self.device, dtype=self.dtype)

        sb_in = torch.cat((current_txt_features, current_img_features), dim=1)
        log_tensor_stats(sb_in, "ChromaModel_fwd_sb_in_PRE_SB_LOOP", logger_obj=logger)
        
        for i, blk in enumerate(self.single_blocks):
            single_mod_for_block = mod_vectors_dict.get(f"single_blocks.{i}.modulation.lin")
            
            if self.blocks_to_swap and self.offloader_single: self.offloader_single.wait_for_block(i)
            if self.gradient_checkpointing and self.training:
                sb_in = torch_checkpoint(
                    blk, sb_in, pe, single_mod_for_block, combined_mask_for_attn,
                    use_reentrant=False
                )
            else:
                sb_in = blk(x=sb_in, pe=pe, distill_vec=single_mod_for_block, mask=combined_mask_for_attn)
            if torch.isnan(sb_in).any():
                logger.error(f"NaN detected after SingleStreamBlock {i}!"); break
            if self.blocks_to_swap and self.offloader_single: self.offloader_single.submit_move_blocks(self.single_blocks,i)

        if torch.isnan(sb_in).any():
            nan_output_shape = (img.shape[0], img_p.shape[1], self.out_channels)
            return torch.full(nan_output_shape, float('nan'), device=self.device, dtype=self.dtype)

        img_f_final = sb_in[:, L_txt_calc:] 
        log_tensor_stats(img_f_final, "ChromaModel_fwd_img_f_final_PRE_LASTLAYER", logger_obj=logger)
        
        if self.training and self.cpu_offload_checkpointing and self.blocks_to_swap:
            img_f_final = img_f_final.to(self.device)
            
        final_mod_for_lastlayer = mod_vectors_dict.get("final_layer.adaLN_modulation.1")
        if final_mod_for_lastlayer is None:
            logger.warning("Modulation for final_layer not found in mod_vectors_dict. Using zeros.")
            dummy_final_mod_shape = (img_f_final.shape[0], 1, self.hidden_size)
            final_mod_for_lastlayer = [
                torch.zeros(dummy_final_mod_shape, device=img_f_final.device, dtype=img_f_final.dtype),
                torch.zeros(dummy_final_mod_shape, device=img_f_final.device, dtype=img_f_final.dtype)
            ]

        img_out = self.final_layer(img_f_final, distill_vec=final_mod_for_lastlayer)
        log_tensor_stats(img_out, "ChromaModel_fwd_output_from_final_layer_img_out", logger_obj=logger)
        return img_out

# --- End ChromaModel definition ---