# modified for chroma

# copy from FLUX repo: https://github.com/black-forest-labs/flux
# license: Apache-2.0 License


import math
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple, TYPE_CHECKING 
import logging 

from library import utils
from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from library import custom_offloading_utils


logger = logging.getLogger(__name__) 

# from library import flux_models # Removed self-import
from library.utils import load_safetensors 

if TYPE_CHECKING:
    pass


@dataclass
class ApproximatorParams:
    in_dim: int = 64
    out_dim_per_mod_vector: int = 0 
    hidden_dim: int = 5120 
    n_layers: int = 4
    mod_index_length: int = 344 

@dataclass
class FluxParams:
    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int 
    mlp_ratio: float
    num_heads: int   
    depth: int  
    depth_single_blocks: int 
    axes_dim: list[int]
    theta: int
    qkv_bias: bool         
    guidance_embed: bool   
    use_modulation: bool   # True for Schnell/Dev, False for Chroma (external mod)
    use_distilled_guidance_layer: bool # For Chroma's feature_fusion_distiller layer
    distilled_guidance_dim: int = 5120 
    use_time_embed: bool = True      
    use_vector_embed: bool = True    
    # If True, DoubleStreamBlocks have their main LayerNorms with learnable params if use_modulation=False
    # If False, these norms might be nn.Identity() or non-affine.
    double_block_has_main_norms: bool = True 
    approximator_config: Optional[ApproximatorParams] = None


# region autoencoder


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
# region config

@dataclass
class ModelSpec:
    params: Union[FluxParams, Dict]
    ae_params: AutoEncoderParams
    ckpt_path: str | None
    ae_path: str | None

_original_configs_for_ae = {
    "dev": ModelSpec(ckpt_path=None, params={}, ae_path=None, ae_params=AutoEncoderParams(
        resolution=256, in_channels=3, ch=128, out_ch=3, ch_mult=[1, 2, 4, 4],
        num_res_blocks=2, z_channels=16, scale_factor=0.3611, shift_factor=0.1159)),
    "schnell": ModelSpec(ckpt_path=None, params={}, ae_path=None, ae_params=AutoEncoderParams(
        resolution=256, in_channels=3, ch=128, out_ch=3, ch_mult=[1, 2, 4, 4],
        num_res_blocks=2, z_channels=16, scale_factor=0.3611, shift_factor=0.1159)),
}

def flux1_dev_params(depth_double=19, depth_single=38):
    return FluxParams(in_channels=64, vec_in_dim=768, context_in_dim=4096, hidden_size=3072,
        mlp_ratio=4.0, num_heads=24, depth=depth_double, depth_single_blocks=depth_single,
        axes_dim=[16, 56, 56], theta=10_000, qkv_bias=True, guidance_embed=True, 
        use_modulation=True, use_distilled_guidance_layer=False, use_time_embed=True, 
        use_vector_embed=True, double_block_has_main_norms=True)

def flux1_schnell_params(depth_double=19, depth_single=38):
    return FluxParams(in_channels=64, vec_in_dim=768, context_in_dim=4096, hidden_size=3072,
        mlp_ratio=4.0, num_heads=24, depth=depth_double, depth_single_blocks=depth_single,
        axes_dim=[16, 56, 56], theta=10_000, qkv_bias=True, guidance_embed=False, 
        use_modulation=True, use_distilled_guidance_layer=False, use_time_embed=True, 
        use_vector_embed=True, double_block_has_main_norms=True)

def flux_chroma_params(depth_double=19, depth_single=38):
    return FluxParams(in_channels=64, vec_in_dim=768, context_in_dim=4096, hidden_size=3072,
        mlp_ratio=4.0, num_heads=24, depth=depth_double, depth_single_blocks=depth_single,
        axes_dim=[16, 56, 56], theta=10_000, qkv_bias=True, guidance_embed=True,
        use_modulation=False, use_distilled_guidance_layer=True, distilled_guidance_dim=5120,
        use_time_embed=False, use_vector_embed=False, 
        double_block_has_main_norms=False, # Chroma blocks do NOT have their own learnable norm parameters
        approximator_config=ApproximatorParams(in_dim=64, out_dim_per_mod_vector=3072,
            hidden_dim=5120, n_layers=4, mod_index_length=344))

model_configs = {
    "dev": ModelSpec(params=flux1_dev_params(), ae_params=_original_configs_for_ae["dev"].ae_params, ckpt_path=None, ae_path=None),
    "schnell": ModelSpec(params=flux1_schnell_params(), ae_params=_original_configs_for_ae["schnell"].ae_params, ckpt_path=None, ae_path=None),
    "chroma": ModelSpec(params=flux_chroma_params(), ae_params=_original_configs_for_ae["schnell"].ae_params, ckpt_path=None, ae_path=None)
}

# endregion
# region math

def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
    q, k = apply_rope(q, k, pe)
    x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    return rearrange(x, "B H L D -> B L (H D)")

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    return rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2).float()

def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)

# endregion
# region layers

def to_cuda(x): 
    if isinstance(x, torch.Tensor): return x.cuda()
    if isinstance(x, (list, tuple)): return [to_cuda(elem) for elem in x]
    if isinstance(x, dict): return {k: to_cuda(v) for k, v in x.items()}
    return x

def to_cpu(x): 
    if isinstance(x, torch.Tensor): return x.cpu()
    if isinstance(x, (list, tuple)): return [to_cpu(elem) for elem in x]
    if isinstance(x, dict): return {k: to_cpu(v) for k, v in x.items()}
    return x

class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__(); self.dim, self.theta, self.axes_dim = dim, theta, axes_dim
    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)

def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    original_shape = t.shape
    if t.ndim > 1: t = t.reshape(-1) 
    t_scaled = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
    args = t_scaled[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2: embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if len(original_shape) > 1 and embedding.shape[0] == original_shape[0] * original_shape[1]:
        embedding = embedding.view(*original_shape, dim)
    original_t_dtype = t.dtype if t.ndim == 1 and original_shape == t.shape else t_scaled.dtype
    if torch.is_floating_point(t_scaled): embedding = embedding.to(original_t_dtype)
    return embedding

class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.gradient_checkpointing = False
    def enable_gradient_checkpointing(self): self.gradient_checkpointing = True
    def disable_gradient_checkpointing(self): self.gradient_checkpointing = False
    def _forward(self, x: Tensor) -> Tensor: return self.out_layer(self.silu(self.in_layer(x)))
    def forward(self, *args, **kwargs):
        if self.training and self.gradient_checkpointing:
            return checkpoint(self._forward, *args, use_reentrant=False, **kwargs)
        return self._forward(*args, **kwargs)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.scale = nn.Parameter(torch.ones(dim))
    def forward(self, x: Tensor):
        x_dtype = x.dtype; x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return ((x * rrms) * self.scale.float()).to(dtype=x_dtype)

class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__(); self.query_norm, self.key_norm = RMSNorm(dim), RMSNorm(dim)
    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        return self.query_norm(q).to(v), self.key_norm(k).to(v)

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__(); self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(dim // num_heads)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x: Tensor, pe: Tensor) -> Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)
        x = attention(q, k, v, pe=pe); return self.proj(x)

@dataclass
class ModulationOut: shift: Tensor; scale: Tensor; gate: Tensor
def log_tensor_stats(tensor: Optional[torch.Tensor], name: str = "tensor", logger_obj: Optional[logging.Logger] = None):
    """Logs statistics of a tensor if it's not None."""
    if logger_obj is None:
        logger_obj = logging.getLogger(__name__) # Fallback to current module's logger

    if tensor is not None:
        # Ensure operations are safe even for 0-element tensors if they can occur
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

class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__(); self.is_double, self.multiplier = double, 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)
    def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return (ModulationOut(*out[:3]), ModulationOut(*out[3:]) if self.is_double else None)

class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6) # Non-affine as per original
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        # adaLN_modulation is present but will be unused by Chroma if vec is None and distill_vec is used for shift/scale only
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
    
    def forward(self, x: Tensor, vec: Optional[Tensor] = None, distill_vec: Optional[List[Tensor]] = None) -> Tensor:
        # For Chroma, vec will be effectively zero if its inputs (time_in, vector_in) are None.
        # distill_vec is not used by Chroma's final_layer path.
        # The `final_layer_custom_path` in Flux class for Chroma bypasses this LastLayer's adaLN part.
        # If Flux `final_layer_custom_path` is False (e.g. for Dev/Schnell), then vec is used.
        
        log_tensor_stats(x, "LastLayer_input_x", logger_obj=logger) # Use your log_tensor_stats helper
        x_normed = self.norm_final(x)
        log_tensor_stats(x_normed, "LastLayer_x_normed", logger_obj=logger)

        if vec is not None and not (vec.ndim == 3 and vec.shape[1] == 1 and vec.shape[2] == self.adaLN_modulation[1].in_features and torch.all(vec == 0)):
            mod_out = self.adaLN_modulation(vec)
            shift, scale = mod_out.chunk(2, dim=1)
            shift, scale = shift.unsqueeze(1), scale.unsqueeze(1)
            x_mod = (1 + scale) * x_normed + shift
            log_tensor_stats(x_mod, "LastLayer_x_mod_with_adaLN", logger_obj=logger)
        else:
            x_mod = x_normed
            log_tensor_stats(x_mod, "LastLayer_x_mod_no_adaLN", logger_obj=logger)

        # You can also log the weight dtype of the linear layer if needed
        # logger.debug(f"LastLayer: self.linear.weight.dtype: {self.linear.weight.dtype}")
        x_out = self.linear(x_mod)
        log_tensor_stats(x_out, "LastLayer_x_out_FINAL", logger_obj=logger)
        return x_out

class ApproximatorLayer(nn.Module): # To match Ostris/Chroma checkpoint keys
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.act = nn.SiLU() # Or GELU if preferred for approximator MLP

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.act(self.in_layer(x)))

class Approximator(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, n_layers=4):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim, bias=True)
        # To match `layers.X.in_layer` and `layers.X.out_layer` from analyze_model_keys
        self.layers = nn.ModuleList([ApproximatorLayer(hidden_dim) for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(hidden_dim) for _ in range(n_layers)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor: 
        x = self.in_proj(x)
        for i, layer_group in enumerate(self.layers): # layer_group is ApproximatorLayer
            x_normed = self.norms[i](x)
            x = x + layer_group(x_normed) # Residual connection around norm + ApproximatorLayer
        return self.out_proj(x)


def distribute_modulations_from_approximator(
    mod_vectors: Tensor, num_double_blocks: int, num_single_blocks: int, flux_params: FluxParams
) -> Dict[str, Any]:
    mod_dict = {}; B, L_mod, H_mod = mod_vectors.shape
    dev, dtype = mod_vectors.device, mod_vectors.dtype
    
    # For Chroma (use_modulation=False), the number of vectors needed by blocks (excluding final layer)
    # is (num_double_blocks * 2 for img_mod + num_double_blocks * 2 for txt_mod) + (num_single_blocks * 1 for single_mod)
    # Each "mod" (img_mod, txt_mod, single_mod) itself needs 3 vectors from the approximator (shift, scale, gate).
    expected_vecs_for_blocks = (num_double_blocks * 2 * 3) + (num_double_blocks * 2 * 3) + (num_single_blocks * 1 * 3)
    # Final layer for Chroma if not custom_path uses LastLayer, which takes adaLN from `vec`.
    # If `final_layer_custom_path` is True, this part is not used.
    # If `final_layer_custom_path` is False, LastLayer's adaLN_modulation (if vec is not zero) needs 2 vectors (shift, scale).
    # However, Chroma's `flux_chroma_params` makes `final_layer_custom_path=True`, so we don't need vectors for final_layer from approximator.

    expected_total_vecs = expected_vecs_for_blocks
    
    if L_mod != flux_params.approximator_config.mod_index_length or L_mod < expected_total_vecs:
         logger.warning(f"Approximator produced {L_mod} mod vectors, mod_index_length is {flux_params.approximator_config.mod_index_length}, expected at least {expected_total_vecs} for blocks.")

    current_idx = 0
    def _get_mod_out():
        nonlocal current_idx
        if current_idx + 2 >= L_mod: # Need 3 vectors
            # logger.warning(f"Not enough mod_vectors from approximator for a block component. Requested 3 from {current_idx}, have {L_mod}. Returning zeros.")
            s = torch.zeros(B, 1, H_mod, device=dev, dtype=dtype)
            sc = torch.zeros(B, 1, H_mod, device=dev, dtype=dtype) 
            g = torch.ones(B, 1, H_mod, device=dev, dtype=dtype)   
            # Don't advance current_idx if we are returning zeros due to insufficient vectors
            return ModulationOut(s, sc, g)
        
        shift = mod_vectors[:, current_idx, :].unsqueeze(1)
        scale = mod_vectors[:, current_idx + 1, :].unsqueeze(1)
        gate = mod_vectors[:, current_idx + 2, :].unsqueeze(1)
        current_idx += 3
        return ModulationOut(shift, scale, gate)

    for i in range(num_double_blocks):
        mod_dict[f"double_blocks.{i}.img_mod.lin"] = [_get_mod_out(), _get_mod_out()] # mod1 (attn), mod2 (mlp)
        mod_dict[f"double_blocks.{i}.txt_mod.lin"] = [_get_mod_out(), _get_mod_out()] # mod1 (attn), mod2 (mlp)

    for i in range(num_single_blocks):
        mod_dict[f"single_blocks.{i}.modulation.lin"] = _get_mod_out()
        
    # For Chroma, final_layer_custom_path is True, so we don't use adaLN_modulation from approximator for final layer.
    # The `final_norm` and `final_linear` are directly part of Flux class for Chroma.

    if current_idx > L_mod :
        logger.error(f"Overran modulation vectors. Consumed {current_idx}, had {L_mod}.")
    elif current_idx < expected_total_vecs and L_mod >= expected_total_vecs : # Consumed less than expected, but had enough
        logger.warning(f"Underran modulation vectors but had enough. Consumed {current_idx}, available {L_mod}, expected for blocks {expected_total_vecs}.")
    elif L_mod < expected_total_vecs: # Didn't have enough to begin with
        logger.warning(f"Did not have enough modulation vectors. Consumed {current_idx}, available {L_mod}, needed for blocks {expected_total_vecs}.")

    return mod_dict


class FeatureFusionDistillerLayer(nn.Module): # To match Ostris/Chroma checkpoint keys
    def __init__(self, hidden_dim: int):
        super().__init__()
        # Checkpoint has keys like 'distilled_guidance_layer.layers.0.in_layer.weight'
        self.in_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.act = nn.GELU(approximate="tanh") # As per changelog

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.act(self.in_layer(x)))

class FeatureFusionDistiller(nn.Module): 
    def __init__(self, input_dim: int, internal_dim: int, output_dim: int, num_internal_layers: int = 5):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, internal_dim, bias=True)
        self.layers = nn.ModuleList([FeatureFusionDistillerLayer(internal_dim) for _ in range(num_internal_layers)])
        self.norms = nn.ModuleList([RMSNorm(internal_dim) for _ in range(num_internal_layers)])
        self.out_proj = nn.Linear(internal_dim, output_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        for i, layer_group in enumerate(self.layers): # layer_group is FeatureFusionDistillerLayer
            residual = x
            x_normed = self.norms[i](x)
            x = residual + layer_group(x_normed) # Pass normed x to the layer_group
        return self.out_proj(x)

# endregion

class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False,
                 use_modulation: bool = True, has_main_norms: bool = True):
        super().__init__()
        self.hidden_size, self.num_heads, self.use_modulation, self.has_main_norms = hidden_size, num_heads, use_modulation, has_main_norms
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        
        if self.use_modulation: # Schnell/Dev path
            self.img_mod, self.txt_mod = Modulation(hidden_size, True), Modulation(hidden_size, True)
        
        # For Chroma (use_modulation=False), if has_main_norms is True, LayerNorms are affine.
        # If has_main_norms is False (as per flux_chroma_params), they are Identity.
        if self.has_main_norms:
            # Affine if NOT using internal modulation (i.e., for Chroma if norms are present)
            # OR if using internal modulation but the norm is supposed to be affine anyway (original FLUX behavior)
            # The critical point is whether the checkpoint HAS these affine weights.
            # For Chroma, `double_block_has_main_norms` is set to False in params, so this branch is skipped.
            is_affine_for_norm = not self.use_modulation 
            self.img_norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=is_affine_for_norm)
            self.img_norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=is_affine_for_norm)
            self.txt_norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=is_affine_for_norm)
            self.txt_norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=is_affine_for_norm)
        else: # This path will be taken by Chroma due to flux_chroma_params
            self.img_norm1, self.img_norm2 = nn.Identity(), nn.Identity()
            self.txt_norm1, self.txt_norm2 = nn.Identity(), nn.Identity()
            # logger.debug("DoubleStreamBlock initialized with Identity norms (Chroma path).")

        self.img_attn = SelfAttention(hidden_size, num_heads, qkv_bias)
        self.img_mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden_dim, True), nn.GELU(approximate="tanh"), nn.Linear(mlp_hidden_dim, hidden_size, True))
        self.txt_attn = SelfAttention(hidden_size, num_heads, qkv_bias)
        self.txt_mlp = nn.Sequential(nn.Linear(hidden_size, mlp_hidden_dim, True), nn.GELU(approximate="tanh"), nn.Linear(mlp_hidden_dim, hidden_size, True))
        self.gradient_checkpointing = False; self.cpu_offload_checkpointing = False

    def enable_gradient_checkpointing(self, cpu_offload: bool = False): self.gradient_checkpointing, self.cpu_offload_checkpointing = True, cpu_offload
    def disable_gradient_checkpointing(self): self.gradient_checkpointing, self.cpu_offload_checkpointing = False, False
    def _forward(self, img: Tensor, txt: Tensor, vec: Tensor, pe: Tensor, txt_attention_mask: Optional[Tensor] = None,
                 distill_img_mod_params: Optional[List[ModulationOut]] = None, distill_txt_mod_params: Optional[List[ModulationOut]] = None) -> tuple[Tensor, Tensor]:
        img_res, txt_res = img, txt
        # Norms are applied regardless of being Identity or LayerNorm
        img_n, txt_n = self.img_norm1(img), self.txt_norm1(txt) 

        if self.use_modulation: # Schnell/Dev path
            img_m1, img_m2 = self.img_mod(vec); txt_m1, txt_m2 = self.txt_mod(vec)
        elif distill_img_mod_params and distill_txt_mod_params: # Chroma path with provided external modulations
            img_m1, img_m2 = distill_img_mod_params; txt_m1, txt_m2 = distill_txt_mod_params
        else: # Fallback for Chroma if approximator isn't used/set up, or for general unconditioned blocks
            s = torch.zeros_like(img_n[:,0:1,:]) if img_n.numel() > 0 else torch.tensor(0., device=img.device, dtype=img.dtype).view(1,1,1).expand(img.shape[0],1,img.shape[-1])
            sc = torch.zeros_like(s) # scale = 0 => 1+scale=1
            g = torch.ones_like(s)   # gate = 1
            img_m1 = img_m2 = txt_m1 = txt_m2 = ModulationOut(s,sc,g)
            if not self.use_modulation: logger.debug("DoubleStreamBlock (Chroma path) using identity mod_params as none were distilled/provided.")
        
        img_nm, txt_nm = (1+img_m1.scale)*img_n+img_m1.shift, (1+txt_m1.scale)*txt_n+txt_m1.shift
        img_q,img_k,img_v=rearrange(self.img_attn.qkv(img_nm),"B L (K H D)->K B H L D",K=3,H=self.num_heads); img_q,img_k=self.img_attn.norm(img_q,img_k,img_v)
        txt_q,txt_k,txt_v=rearrange(self.txt_attn.qkv(txt_nm),"B L (K H D)->K B H L D",K=3,H=self.num_heads); txt_q,txt_k=self.txt_attn.norm(txt_q,txt_k,txt_v)
        q,k,v=torch.cat((txt_q,img_q),2),torch.cat((txt_k,img_k),2),torch.cat((txt_v,img_v),2)
        mask_exp = None
        if txt_attention_mask is not None: # txt_attention_mask is (B, L_txt)
            img_mask_shape_L = img.shape[1] if img.ndim == 3 else (img.shape[1] * img.shape[2] if img.ndim == 4 else img.shape[1]) # Handle packed/unpacked
            img_mask = torch.ones(txt_attention_mask.shape[0], img_mask_shape_L, device=txt_attention_mask.device, dtype=torch.bool)
            full_mask = torch.cat((txt_attention_mask, img_mask), dim=1)
            # Ensure L_total matches q/k/v sequence length
            L_total_qkv = q.shape[2] 
            if full_mask.shape[1] != L_total_qkv:
                logger.warning(f"Attention mask length ({full_mask.shape[1]}) mismatch with QKV sequence length ({L_total_qkv}). Adjusting mask.")
                # This might happen if L_img changes due to packing/unpacking not accounted for.
                # A robust solution would be to ensure L_img is correctly derived.
                # For now, a simple truncation or padding might be a temporary fix, but it's not ideal.
                # Let's assume the logic for img_mask_shape_L should be correct based on `img` passed to the block.
                if full_mask.shape[1] > L_total_qkv : full_mask = full_mask[:, :L_total_qkv]
                elif full_mask.shape[1] < L_total_qkv: # Pad with ones (attend)
                    padding = torch.ones(full_mask.shape[0], L_total_qkv - full_mask.shape[1], device=full_mask.device, dtype=torch.bool)
                    full_mask = torch.cat((full_mask, padding), dim=1)
            mask_exp = ~(full_mask[:, None, None, :].expand(-1, q.shape[1], L_total_qkv, L_total_qkv))


        joint_out = attention(q,k,v,pe,mask_exp); txt_att,img_att = joint_out[:,:txt.shape[1]],joint_out[:,txt.shape[1]:]
        img_after = img_res + img_m1.gate * self.img_attn.proj(img_att)
        img = img_after + img_m2.gate * self.img_mlp((1+img_m2.scale)*self.img_norm2(img_after)+img_m2.shift)
        txt_after = txt_res + txt_m1.gate * self.txt_attn.proj(txt_att)
        txt = txt_after + txt_m2.gate * self.txt_mlp((1+txt_m2.scale)*self.txt_norm2(txt_after)+txt_m2.shift)
        return img, txt
    def forward(self, *args, **kwargs):
        if self.training and self.gradient_checkpointing:
            if not self.cpu_offload_checkpointing: return checkpoint(self._forward, *args, use_reentrant=False, **kwargs)
            def create_cf(func): return lambda *i: to_cpu(func(*to_cuda(i)))
            return torch.utils.checkpoint.checkpoint(create_cf(self._forward), *args, use_reentrant=False, **kwargs)
        return self._forward(*args, **kwargs)

class SingleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, use_modulation: bool = True):
        super().__init__()
        self.hidden_size,self.num_heads,self.use_modulation = hidden_size,num_heads,use_modulation
        h_dim,self.mlp_hidden_dim = hidden_size//num_heads,int(hidden_size*mlp_ratio)
        # For Chroma (use_modulation=False), pre_norm should be elementwise_affine=False as per missing keys.
        self.pre_norm = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False if not self.use_modulation else True)
        self.linear1 = nn.Linear(hidden_size, hidden_size*3+self.mlp_hidden_dim, bias=True)
        self.norm = QKNorm(h_dim); self.mlp_act = nn.GELU("tanh")
        self.linear2 = nn.Linear(hidden_size+self.mlp_hidden_dim, hidden_size, bias=True)
        if self.use_modulation: self.modulation = Modulation(hidden_size, False)
        self.gradient_checkpointing=False; self.cpu_offload_checkpointing=False
    def enable_gradient_checkpointing(self, cpu_offload: bool = False): self.gradient_checkpointing,self.cpu_offload_checkpointing = True,cpu_offload
    def disable_gradient_checkpointing(self): self.gradient_checkpointing,self.cpu_offload_checkpointing = False,False
    def _forward(self, x: Tensor, vec: Tensor, pe: Tensor, txt_attention_mask: Optional[Tensor] = None, distill_mod_params: Optional[ModulationOut] = None) -> Tensor:
        res,x_n=x,self.pre_norm(x); final_gate=None
        if self.use_modulation: mod,_=self.modulation(vec); x_mod,final_gate=(1+mod.scale)*x_n+mod.shift,mod.gate
        elif distill_mod_params: x_mod,final_gate=(1+distill_mod_params.scale)*x_n+distill_mod_params.shift,distill_mod_params.gate
        else: 
            x_mod=x_n
            if not self.use_modulation: logger.debug("SingleStreamBlock (Chroma path) using identity mod_params as none were distilled/provided.")
        
        proj_feat=self.linear1(x_mod); qkv_c,mlp_in_f=torch.split(proj_feat,[3*self.hidden_size,self.mlp_hidden_dim],dim=-1)
        q,k,v=rearrange(qkv_c,"B L (K H D)->K B H L D",K=3,H=self.num_heads); q_n,k_n=self.norm(q,k,v)
        mask_exp=None; 
        if txt_attention_mask is not None: # txt_attention_mask is (B, L_total)
             mask_exp = ~(txt_attention_mask.to(torch.bool)[:,None,None,:].expand(-1,self.num_heads,x.shape[1],-1))
        attn_o=attention(q_n,k_n,v,pe,mask_exp); mlp_o=self.mlp_act(mlp_in_f)
        out_upd=self.linear2(torch.cat((attn_o,mlp_o),dim=-1))
        return res+final_gate*out_upd if final_gate is not None else res+out_upd
    def forward(self, *args, **kwargs):
        if self.training and self.gradient_checkpointing:
            if not self.cpu_offload_checkpointing: return checkpoint(self._forward, *args, use_reentrant=False, **kwargs)
            def create_cf(func): return lambda *i: to_cpu(func(*to_cuda(i)))
            return torch.utils.checkpoint.checkpoint(create_cf(self._forward), *args, use_reentrant=False, **kwargs)
        return self._forward(*args, **kwargs)

class Flux(nn.Module):
    def __init__(self, params: FluxParams):
        super().__init__()
        self.params_dto = params; self.in_channels,self.out_channels=params.in_channels,params.in_channels
        if params.hidden_size%params.num_heads!=0: raise ValueError(f"Hidden {params.hidden_size} not div by {params.num_heads}")
        pe_dim=params.hidden_size//params.num_heads
        if sum(params.axes_dim)!=pe_dim: raise ValueError(f"axes_dim sum {sum(params.axes_dim)} != pe_dim {pe_dim}")
        self.hidden_size,self.num_heads=params.hidden_size,params.num_heads
        self.pe_embedder=EmbedND(pe_dim,params.theta,params.axes_dim); self.img_in=nn.Linear(params.in_channels,params.hidden_size,True)
        self.time_in=MLPEmbedder(256,params.hidden_size) if params.use_time_embed else None
        self.vector_in=MLPEmbedder(params.vec_in_dim,params.hidden_size) if params.use_vector_embed else None
        self.guidance_in=MLPEmbedder(256,params.hidden_size) if params.guidance_embed else nn.Identity()
        self.txt_in=nn.Linear(params.context_in_dim,params.hidden_size,True)
        self.double_blocks=nn.ModuleList([DoubleStreamBlock(params.hidden_size,params.num_heads,params.mlp_ratio,params.qkv_bias,params.use_modulation,params.double_block_has_main_norms) for _ in range(params.depth)])
        self.single_blocks=nn.ModuleList([SingleStreamBlock(params.hidden_size,params.num_heads,params.mlp_ratio,params.use_modulation) for _ in range(params.depth_single_blocks)])
        
        self.distilled_guidance_layer = FeatureFusionDistiller( 
            params.in_channels, params.distilled_guidance_dim, params.hidden_size
        ) if params.use_distilled_guidance_layer else None

        self.modulation_approximator=None
        if not params.use_modulation and params.approximator_config:
            app_cfg=params.approximator_config
            if app_cfg.out_dim_per_mod_vector==0: app_cfg.out_dim_per_mod_vector=params.hidden_size
            self.modulation_approximator=Approximator(app_cfg.in_dim,app_cfg.out_dim_per_mod_vector,app_cfg.hidden_dim,app_cfg.n_layers)
            self.register_buffer('mod_index',torch.arange(app_cfg.mod_index_length),persistent=False); self.mod_index_length=app_cfg.mod_index_length
        elif not params.use_modulation and not params.approximator_config: logger.warning("Flux configured use_modulation=False but no approximator_config.")
        
        # For Chroma, final_layer_custom_path is True if use_modulation is False
        # This means Chroma uses self.final_norm and self.final_linear defined here.
        # For Dev/Schnell, final_layer_custom_path is False, and they use self.final_layer (instance of LastLayer)
        self.final_layer_custom_path = not params.use_modulation 
        if self.final_layer_custom_path: 
            # These keys ('final_norm.weight', 'final_linear.weight', etc.) are reported missing for Chroma.
            # This suggests the Chroma checkpoint does *not* have these top-level final_norm/linear layers.
            # Instead, it might be using the .linear part of the original `final_layer` structure.
            # Let's revert to always using `LastLayer` and handle Chroma's specifics in its forward or by what's loaded.
            # The `final_layer.linear.weight/bias` are in "BOTH" for Schnell vs Chroma v34.
            # `final_layer.adaLN_modulation.*` are "pruned".
            # `LastLayer.norm_final` is non-affine so has no weights.
            # This means `final_layer_custom_path` should probably be False for all, and Chroma loads only the linear part.
            self.final_layer_custom_path = False # Always use LastLayer structure
            self.final_layer = LastLayer(params.hidden_size,1,self.out_channels)
            # logger.info("Using LastLayer for all variants, including Chroma (custom path logic removed).")
        else: # Dev/Schnell
            self.final_layer = LastLayer(params.hidden_size,1,self.out_channels)
        
        self.gradient_checkpointing=False; self.cpu_offload_checkpointing=False; self.blocks_to_swap=None
        self.offloader_double=None; self.offloader_single=None
        self.num_double_blocks,self.num_single_blocks=len(self.double_blocks),len(self.single_blocks)
    @property
    def device(self): return next(self.parameters()).device
    @property
    def dtype(self): return next(self.parameters()).dtype
    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing,self.cpu_offload_checkpointing=True,cpu_offload
        for emb in [self.time_in,self.vector_in,self.guidance_in]:
            if hasattr(emb,'enable_gradient_checkpointing'): emb.enable_gradient_checkpointing()
        for blk in self.double_blocks+self.single_blocks: blk.enable_gradient_checkpointing(cpu_offload)
        if self.modulation_approximator:
            for layer in self.modulation_approximator.layers: # these are ApproximatorLayer, not MLPEmbedder
                if hasattr(layer,'enable_gradient_checkpointing'): layer.enable_gradient_checkpointing() # MLPEmbedder had this, ApproximatorLayer doesn't
        logger.info(f"FLUX: GC enabled. CPU offload: {cpu_offload}")
    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing,self.cpu_offload_checkpointing=False,False
        for emb in [self.time_in,self.vector_in,self.guidance_in]:
            if hasattr(emb,'disable_gradient_checkpointing'): emb.disable_gradient_checkpointing()
        for blk in self.double_blocks+self.single_blocks: blk.disable_gradient_checkpointing()
        if self.modulation_approximator:
            for layer in self.modulation_approximator.layers:
                if hasattr(layer,'disable_gradient_checkpointing'): layer.disable_gradient_checkpointing()
        logger.info("FLUX: GC disabled.")
    def enable_block_swap(self, num_blocks: int, device: torch.device):
        self.blocks_to_swap=num_blocks; db_s=num_blocks//2; sb_s=(num_blocks-db_s)*2
        assert db_s<=self.num_double_blocks-2 and sb_s<=self.num_single_blocks-2
        self.offloader_double=custom_offloading_utils.ModelOffloader(self.double_blocks,self.num_double_blocks,db_s,device)
        self.offloader_single=custom_offloading_utils.ModelOffloader(self.single_blocks,self.num_single_blocks,sb_s,device)
        logger.info(f"FLUX: Block swap enabled. Swapping {num_blocks} blocks (D:{db_s}, S:{sb_s}).")
    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap: s_db,s_sb=self.double_blocks,self.single_blocks; self.double_blocks,self.single_blocks=None,None
        self.to(device)
        if self.blocks_to_swap: self.double_blocks,self.single_blocks=s_db,s_sb
    def prepare_block_swap_before_forward(self):
        if not self.blocks_to_swap: return
        self.offloader_double.prepare_block_devices_before_forward(self.double_blocks)
        self.offloader_single.prepare_block_devices_before_forward(self.single_blocks)
    def forward(self, img: Tensor, img_ids: Tensor, txt: Tensor, txt_ids: Tensor, timesteps: Tensor,
                y: Tensor, guidance: Tensor|None=None, txt_attention_mask: Tensor|None=None, **kwargs) -> Tensor:
        # img: (B, L_img_packed, D_in_img), from packed VAE output
        # img_ids: (B, L_img_packed, 3), positional IDs for image tokens
        # txt: (B, L_txt_seq, D_t5), T5 feature outputs
        # txt_ids: (B, L_txt_seq), T5 vocabulary token IDs (THIS IS THE ONE MISUSED FOR pe_embedder)

        if img.ndim != 3 or txt.ndim != 3: # txt is (B, L, D), img is (B, L, D)
            raise ValueError(f"Inputs img/txt must be 3D. Got img: {img.ndim}D, txt: {txt.ndim}D")
            
        img_p = self.img_in(img)
        if self.distilled_guidance_layer: img_p = img_p + self.distilled_guidance_layer(img)
        
        txt_f = self.txt_in(txt) # txt_f is (B, L_txt_seq, D_hidden)
        
        int_cond_parts=[]
        if self.time_in: int_cond_parts.append(self.time_in(timestep_embedding(timesteps,256)))
        if self.params_dto.guidance_embed and guidance is not None and not isinstance(self.guidance_in,nn.Identity):
            int_cond_parts.append(self.guidance_in(timestep_embedding(guidance,256)))
        if self.vector_in: int_cond_parts.append(self.vector_in(y))
        
        internal_cond_vec = sum(int_cond_parts) if int_cond_parts else \
                            torch.zeros(img.shape[0], self.hidden_size, device=img.device, dtype=img.dtype)
        
        mod_vecs_chroma=None
        if self.modulation_approximator:
            app_cfg=self.params_dto.approximator_config; _bs=img.shape[0]
            ts_emb_approx = timestep_embedding(timesteps,16)
            guid_val_approx = guidance if guidance is not None else torch.zeros_like(timesteps, device=timesteps.device, dtype=timesteps.dtype)
            guid_emb_approx = timestep_embedding(guid_val_approx,16)
            mod_idx_batched = self.mod_index.unsqueeze(0).repeat(_bs, 1)
            mod_idx_emb = timestep_embedding(mod_idx_batched,32)
            ts_guid_emb_approx = torch.cat([ts_emb_approx, guid_emb_approx], dim=1).unsqueeze(1).repeat(1,self.mod_index_length,1)
            approx_in = torch.cat([ts_guid_emb_approx, mod_idx_emb], dim=-1)
            mod_vecs=self.modulation_approximator(approx_in)
            mod_vecs_chroma=distribute_modulations_from_approximator(mod_vecs,self.num_double_blocks,self.num_single_blocks,self.params_dto)

        # B: batch_size, L_txt: sequence length of text features
        B, L_txt, _ = txt_f.shape 

        txt_seq_positions = torch.arange(L_txt, device=txt_f.device, dtype=img_ids.dtype) 
        txt_seq_positions = txt_seq_positions.unsqueeze(0).expand(B, -1) # (B, L_txt)

        # 2. Expand text positional IDs to 3D to match img_ids's structure (B, L, 3_axes)
        # We represent text 1D position as (sequence_pos, 0, 0) for the 3 axes.
        # The specific meaning of the two zeroed axes for text is arbitrary but needed for uniform structure.
        txt_pos_ids_3d = torch.zeros(B, L_txt, 3, device=txt_f.device, dtype=img_ids.dtype)
        txt_pos_ids_3d[..., 0] = txt_seq_positions # Fill the first axis with sequence positions

        # 3. Concatenate the 3D positional IDs for text and image tokens
        # img_ids is already (B, L_img_packed, 3)
        all_pos_ids = torch.cat((txt_pos_ids_3d, img_ids), dim=1) # Concatenate along sequence dimension (dim=1)
        
        # 4. Generate positional embeddings using the combined positional IDs
        pe = self.pe_embedder(all_pos_ids)
        # --- End of the fix ---
        
        current_img_features = img_p
        current_txt_features = txt_f

        for i,blk in enumerate(self.double_blocks):
            dist_img_m,dist_txt_m = (mod_vecs_chroma.get(f"double_blocks.{i}.img_mod.lin"),mod_vecs_chroma.get(f"double_blocks.{i}.txt_mod.lin")) if mod_vecs_chroma else (None,None)
            if self.blocks_to_swap and self.offloader_double: self.offloader_double.wait_for_block(i)
            current_img_features,current_txt_features=blk(current_img_features,current_txt_features,internal_cond_vec,pe,txt_attention_mask,dist_img_m,dist_txt_m)
            if self.blocks_to_swap and self.offloader_double: self.offloader_double.submit_move_blocks(self.double_blocks,i)
            
        sb_in=torch.cat((current_txt_features,current_img_features),dim=1)
        sb_attn_mask=None
        if txt_attention_mask is not None:
            img_mask_sb=torch.ones(txt_attention_mask.shape[0],current_img_features.shape[1],device=txt_attention_mask.device,dtype=txt_attention_mask.dtype) # Use txt_attention_mask.dtype
            sb_attn_mask=torch.cat((txt_attention_mask,img_mask_sb),dim=1)
            
        for i,blk in enumerate(self.single_blocks):
            dist_s_mod=mod_vecs_chroma.get(f"single_blocks.{i}.modulation.lin") if mod_vecs_chroma else None
            if self.blocks_to_swap and self.offloader_single: self.offloader_single.wait_for_block(i)
            sb_in=blk(sb_in,internal_cond_vec,pe,sb_attn_mask,dist_s_mod)
            if self.blocks_to_swap and self.offloader_single: self.offloader_single.submit_move_blocks(self.single_blocks,i)
            
        img_f_final=sb_in[:,current_txt_features.shape[1]:]
        if self.training and self.cpu_offload_checkpointing and self.blocks_to_swap:
            img_f_final,internal_cond_vec=img_f_final.to(self.device),internal_cond_vec.to(self.device)
            
        img_out = self.final_layer(img_f_final, vec=internal_cond_vec, distill_vec=None)
        return img_out

def zero_module(module): [nn.init.zeros_(p) for p in module.parameters()]; return module

class ControlNetFlux(nn.Module):
    def __init__(self, params: FluxParams, controlnet_depth=2, controlnet_single_depth=0):
        super().__init__()
        self.params=params; self.in_channels,self.out_channels=params.in_channels,params.in_channels
        if params.hidden_size%params.num_heads!=0: raise ValueError(f"Hidden {params.hidden_size} not div by {params.num_heads}")
        pe_dim=params.hidden_size//params.num_heads
        if sum(params.axes_dim)!=pe_dim: raise ValueError(f"axes_dim sum {sum(params.axes_dim)} != pe_dim {pe_dim}")
        self.hidden_size,self.num_heads=params.hidden_size,params.num_heads
        self.pe_embedder=EmbedND(pe_dim,params.theta,params.axes_dim); self.img_in=nn.Linear(self.in_channels,self.hidden_size,True)
        self.time_in=MLPEmbedder(256,self.hidden_size) if params.use_time_embed else None
        self.vector_in=MLPEmbedder(params.vec_in_dim,self.hidden_size) if params.use_vector_embed else None
        self.guidance_in=MLPEmbedder(256,self.hidden_size) if params.guidance_embed else nn.Identity()
        self.txt_in=nn.Linear(params.context_in_dim,self.hidden_size,True)
        self.double_blocks=nn.ModuleList([DoubleStreamBlock(self.hidden_size,self.num_heads,params.mlp_ratio,params.qkv_bias,params.use_modulation,params.double_block_has_main_norms) for _ in range(controlnet_depth)])
        self.single_blocks=nn.ModuleList([SingleStreamBlock(self.hidden_size,self.num_heads,params.mlp_ratio,params.use_modulation) for _ in range(controlnet_single_depth)])
        self.gradient_checkpointing=False; self.cpu_offload_checkpointing=False; self.blocks_to_swap=None
        self.offloader_double=None; self.offloader_single=None
        self.num_double_blocks,self.num_single_blocks=len(self.double_blocks),len(self.single_blocks)
        self.controlnet_blocks=nn.ModuleList([zero_module(nn.Linear(self.hidden_size,self.hidden_size)) for _ in range(controlnet_depth)])
        self.controlnet_blocks_for_single=nn.ModuleList([zero_module(nn.Linear(self.hidden_size,self.hidden_size)) for _ in range(controlnet_single_depth)])
        self.pos_embed_input=nn.Linear(self.in_channels,self.hidden_size,True)
        self.input_hint_block=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,16,3,padding=1,stride=2),nn.SiLU(),nn.Conv2d(16,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,16,3,padding=1,stride=2),nn.SiLU(),nn.Conv2d(16,16,3,padding=1),nn.SiLU(),nn.Conv2d(16,16,3,padding=1,stride=2),nn.SiLU(),zero_module(nn.Conv2d(16,16,3,padding=1)))
        self.controlnet_modulation_approximator=None
        if not params.use_modulation and params.approximator_config:
            app_cfg=params.approximator_config; out_d=app_cfg.out_dim_per_mod_vector if app_cfg.out_dim_per_mod_vector!=0 else params.hidden_size
            self.controlnet_modulation_approximator=Approximator(app_cfg.in_dim,out_d,app_cfg.hidden_dim,app_cfg.n_layers)
            self.register_buffer('controlnet_mod_index',torch.arange(app_cfg.mod_index_length),persistent=False); self.controlnet_mod_index_length=app_cfg.mod_index_length
    @property
    def device(self): return next(self.parameters()).device
    @property
    def dtype(self): return next(self.parameters()).dtype
    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        self.gradient_checkpointing,self.cpu_offload_checkpointing=True,cpu_offload
        for emb in [self.time_in,self.vector_in,self.guidance_in]:
            if hasattr(emb,'enable_gradient_checkpointing'): emb.enable_gradient_checkpointing()
        for blk in self.double_blocks+self.single_blocks: blk.enable_gradient_checkpointing(cpu_offload)
        logger.info(f"ControlNetFLUX: GC enabled. CPU offload: {cpu_offload}")
    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing,self.cpu_offload_checkpointing=False,False
        for emb in [self.time_in,self.vector_in,self.guidance_in]:
            if hasattr(emb,'disable_gradient_checkpointing'): emb.disable_gradient_checkpointing()
        for blk in self.double_blocks+self.single_blocks: blk.disable_gradient_checkpointing()
        logger.info("ControlNetFLUX: GC disabled.")
    def enable_block_swap(self, num_blocks: int, device: torch.device):
        self.blocks_to_swap=num_blocks
        if self.num_double_blocks>0 or self.num_single_blocks>0 :
            db_s=num_blocks//2 if self.num_double_blocks>0 else 0
            sb_s=(num_blocks-db_s)*2 if self.num_double_blocks>0 and self.num_single_blocks>0 else (num_blocks*2 if self.num_single_blocks>0 else 0)
            if self.num_double_blocks>2 and db_s>0: assert db_s<=self.num_double_blocks-2; self.offloader_double=custom_offloading_utils.ModelOffloader(self.double_blocks,self.num_double_blocks,db_s,device)
            if self.num_single_blocks>2 and sb_s>0: assert sb_s<=self.num_single_blocks-2; self.offloader_single=custom_offloading_utils.ModelOffloader(self.single_blocks,self.num_single_blocks,sb_s,device)
            logger.info(f"ControlNetFLUX: Block swap. Total: {num_blocks}, D:{db_s}, S:{sb_s}.")
        else: logger.info("ControlNetFLUX: No blocks to swap.")
    def move_to_device_except_swap_blocks(self, device: torch.device):
        if self.blocks_to_swap and (self.offloader_double or self.offloader_single):
            s_db,s_sb=self.double_blocks,self.single_blocks
            if self.offloader_double: self.double_blocks=None
            if self.offloader_single: self.single_blocks=None
        self.to(device)
        if self.blocks_to_swap and (self.offloader_double or self.offloader_single):
            if self.offloader_double: self.double_blocks=s_db
            if self.offloader_single: self.single_blocks=s_sb
    def prepare_block_swap_before_forward(self):
        if not self.blocks_to_swap: return
        if self.offloader_double: self.offloader_double.prepare_block_devices_before_forward(self.double_blocks)
        if self.offloader_single: self.offloader_single.prepare_block_devices_before_forward(self.single_blocks)
    def forward(self, img: Tensor, img_ids: Tensor, controlnet_cond: Tensor, txt: Tensor, txt_ids: Tensor,
                timesteps: Tensor, y: Tensor, guidance: Tensor|None=None, txt_attention_mask: Tensor|None=None) -> tuple[tuple[Tensor],tuple[Tensor]]:
        if img.ndim!=3 or txt.ndim!=3: raise ValueError("Inputs img/txt must be 3D.")
        img_p=self.img_in(img)+self.pos_embed_input(rearrange(self.input_hint_block(controlnet_cond),"b c (h ph) (w pw) -> b (h w) (c ph pw)",ph=(2 if self.params.in_channels==64 else 1),pw=(2 if self.params.in_channels==64 else 1)))
        int_cond_parts=[]
        if self.time_in: int_cond_parts.append(self.time_in(timestep_embedding(timesteps,256)))
        if self.params.guidance_embed and guidance is not None and not isinstance(self.guidance_in,nn.Identity):
            int_cond_parts.append(self.guidance_in(timestep_embedding(guidance,256)))
        if self.vector_in: int_cond_parts.append(self.vector_in(y))
        int_cond_vec=sum(int_cond_parts) if int_cond_parts else torch.zeros(img.shape[0],self.params.hidden_size,device=img.device,dtype=img.dtype)
        mod_vecs_ctrl=None
        if self.controlnet_modulation_approximator and self.params.approximator_config:
            app_cfg=self.params.approximator_config; _bs=img.shape[0]
            ts_e,guid_e=timestep_embedding(timesteps,16),timestep_embedding(guidance if guidance is not None else torch.zeros_like(timesteps),16)
            mod_idx_e=timestep_embedding(self.controlnet_mod_index.unsqueeze(0).repeat(_bs,1),32)
            ts_guid_e=torch.cat([ts_e,guid_e],dim=1).unsqueeze(1).repeat(1,self.controlnet_mod_index_length,1)
            approx_in=torch.cat([ts_guid_e,mod_idx_e],dim=-1)
            mod_vecs=self.controlnet_modulation_approximator(approx_in)
            mod_vecs_ctrl=distribute_modulations_from_approximator(mod_vecs,self.num_double_blocks,self.num_single_blocks,self.params)
        txt_p,pe=self.txt_in(txt),self.pe_embedder(torch.cat((txt_ids,img_ids),dim=1))
        db_outs,sb_outs=[],[]
        cur_img,cur_txt=img_p,txt_p
        for i,blk in enumerate(self.double_blocks):
            if self.blocks_to_swap and self.offloader_double: self.offloader_double.wait_for_block(i)
            dist_img_m,dist_txt_m=(mod_vecs_ctrl.get(f"double_blocks.{i}.img_mod.lin"),mod_vecs_ctrl.get(f"double_blocks.{i}.txt_mod.lin")) if mod_vecs_ctrl else (None,None)
            cur_img,cur_txt=blk(cur_img,cur_txt,int_cond_vec,pe,txt_attention_mask,dist_img_m,dist_txt_m)
            db_outs.append(self.controlnet_blocks[i](cur_img))
            if self.blocks_to_swap and self.offloader_double: self.offloader_double.submit_move_blocks(self.double_blocks,i)
        sb_in=torch.cat((cur_txt,cur_img),dim=1)
        sb_attn_mask=None
        if txt_attention_mask is not None:
            img_mask_sb=torch.ones(txt_attention_mask.shape[0],cur_img.shape[1],device=txt_attention_mask.device,dtype=txt_attention_mask.dtype)
            sb_attn_mask=torch.cat((txt_attention_mask,img_mask_sb),dim=1)
        for i,blk in enumerate(self.single_blocks):
            if self.blocks_to_swap and self.offloader_single: self.offloader_single.wait_for_block(i)
            dist_s_mod=mod_vecs_ctrl.get(f"single_blocks.{i}.modulation.lin") if mod_vecs_ctrl else None
            sb_in=blk(sb_in,int_cond_vec,pe,sb_attn_mask,dist_s_mod)
            sb_outs.append(self.controlnet_blocks_for_single[i](sb_in))
            if self.blocks_to_swap and self.offloader_single: self.offloader_single.submit_move_blocks(self.single_blocks,i)
        return tuple(db_outs),tuple(sb_outs)
