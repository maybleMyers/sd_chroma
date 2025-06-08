# temporary minimum implementation of LoRA
# FLUX doesn't have Conv2d, so we ignore it
# TODO commonize with the original implementation

# LoRA network module
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import math
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Type, Union
from diffusers import AutoencoderKL
from transformers import CLIPTextModel
import numpy as np
import torch
from torch import Tensor
import re
from library.utils import setup_logging
from library.sdxl_original_unet import SdxlUNet2DConditionModel

setup_logging()
import logging

logger = logging.getLogger(__name__)


NUM_DOUBLE_BLOCKS = 19
NUM_SINGLE_BLOCKS = 38


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        split_dims: Optional[List[int]] = None,
        ggpo_beta: Optional[float] = None,
        ggpo_sigma: Optional[float] = None,
    ):
        """
        if alpha == 0 or None, alpha is rank (no scaling).

        split_dims is used to mimic the split qkv of FLUX as same as Diffusers
        """
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim
        self.split_dims = split_dims

        if split_dims is None:
            if org_module.__class__.__name__ == "Conv2d":
                kernel_size = org_module.kernel_size
                stride = org_module.stride
                padding = org_module.padding
                self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
                self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
            else:
                self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
                self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

            torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_up.weight)
        else:
            # conv2d not supported
            assert sum(split_dims) == out_dim, "sum of split_dims must be equal to out_dim"
            assert org_module.__class__.__name__ == "Linear", "split_dims is only supported for Linear"
            # print(f"split_dims: {split_dims}")
            self.lora_down = torch.nn.ModuleList(
                [torch.nn.Linear(in_dim, self.lora_dim, bias=False) for _ in range(len(split_dims))]
            )
            self.lora_up = torch.nn.ModuleList([torch.nn.Linear(self.lora_dim, split_dim, bias=False) for split_dim in split_dims])
            for lora_down in self.lora_down:
                torch.nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
            for lora_up in self.lora_up:
                torch.nn.init.zeros_(lora_up.weight)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = self.lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        self.ggpo_sigma = ggpo_sigma
        self.ggpo_beta = ggpo_beta

        if self.ggpo_beta is not None and self.ggpo_sigma is not None:
            self.combined_weight_norms = None
            self.grad_norms = None
            self.perturbation_norm_factor = 1.0 / math.sqrt(org_module.weight.shape[0])
            self.initialize_norm_cache(org_module.weight)
            self.org_module_shape: tuple[int] = org_module.weight.shape

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward

        del self.org_module

    def forward(self, x):
        org_forwarded = self.org_forward(x)
        
        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        if self.split_dims is None:
            lx = self.lora_down(x)

            # normal dropout
            if self.dropout is not None and self.training:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)

            # rank dropout
            if self.rank_dropout is not None and self.training:
                mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
                if len(lx.size()) == 3:
                    mask = mask.unsqueeze(1)  # for Text Encoder
                elif len(lx.size()) == 4:
                    mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
                lx = lx * mask

                # scaling for rank dropout: treat as if the rank is changed
                # maskから計算することも考えられるが、augmentation的な効果を期待してrank_dropoutを用いる
                scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
            else:
                scale = self.scale

            lx = self.lora_up(lx)

            # LoRA Gradient-Guided Perturbation Optimization
            if self.training and self.ggpo_sigma is not None and self.ggpo_beta is not None and self.combined_weight_norms is not None and self.grad_norms is not None:
                with torch.no_grad():
                    perturbation_scale = (self.ggpo_sigma * torch.sqrt(self.combined_weight_norms ** 2)) + (self.ggpo_beta * (self.grad_norms ** 2))
                    perturbation_scale_factor = (perturbation_scale * self.perturbation_norm_factor).to(self.device)
                    perturbation = torch.randn(self.org_module_shape,  dtype=self.dtype, device=self.device)
                    perturbation.mul_(perturbation_scale_factor)
                    perturbation_output = x @ perturbation.T  # Result: (batch × n)
                return org_forwarded + (self.multiplier * scale * lx) + perturbation_output
            else:
                return org_forwarded + lx * self.multiplier * scale
        else:
            lxs = [lora_down(x) for lora_down in self.lora_down]

            # normal dropout
            if self.dropout is not None and self.training:
                lxs = [torch.nn.functional.dropout(lx, p=self.dropout) for lx in lxs]

            # rank dropout
            if self.rank_dropout is not None and self.training:
                masks = [torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout for lx in lxs]
                for i in range(len(lxs)):
                    if len(lx.size()) == 3:
                        masks[i] = masks[i].unsqueeze(1)
                    elif len(lx.size()) == 4:
                        masks[i] = masks[i].unsqueeze(-1).unsqueeze(-1)
                    lxs[i] = lxs[i] * masks[i]

                # scaling for rank dropout: treat as if the rank is changed
                scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
            else:
                scale = self.scale

            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]

            return org_forwarded + torch.cat(lxs, dim=-1) * self.multiplier * scale

    @torch.no_grad()
    def initialize_norm_cache(self, org_module_weight: Tensor):
        # Choose a reasonable sample size
        n_rows = org_module_weight.shape[0]
        sample_size = min(1000, n_rows)  # Cap at 1000 samples or use all if smaller
        
        # Sample random indices across all rows
        indices = torch.randperm(n_rows)[:sample_size]
        
        # Convert to a supported data type first, then index
        # Use float32 for indexing operations
        weights_float32 = org_module_weight.to(dtype=torch.float32)
        sampled_weights = weights_float32[indices].to(device=self.device)
        
        # Calculate sampled norms
        sampled_norms = torch.norm(sampled_weights, dim=1, keepdim=True)
        
        # Store the mean norm as our estimate
        self.org_weight_norm_estimate = sampled_norms.mean()
        
        # Optional: store standard deviation for confidence intervals
        self.org_weight_norm_std = sampled_norms.std()
        
        # Free memory
        del sampled_weights, weights_float32

    @torch.no_grad()
    def validate_norm_approximation(self, org_module_weight: Tensor, verbose=True):
        # Calculate the true norm (this will be slow but it's just for validation)
        true_norms = []
        chunk_size = 1024  # Process in chunks to avoid OOM
        
        for i in range(0, org_module_weight.shape[0], chunk_size):
            end_idx = min(i + chunk_size, org_module_weight.shape[0])
            chunk = org_module_weight[i:end_idx].to(device=self.device, dtype=self.dtype)
            chunk_norms = torch.norm(chunk, dim=1, keepdim=True)
            true_norms.append(chunk_norms.cpu())
            del chunk
            
        true_norms = torch.cat(true_norms, dim=0)
        true_mean_norm = true_norms.mean().item()
        
        # Compare with our estimate
        estimated_norm = self.org_weight_norm_estimate.item()
        
        # Calculate error metrics
        absolute_error = abs(true_mean_norm - estimated_norm)
        relative_error = absolute_error / true_mean_norm * 100  # as percentage
        
        if verbose:
            logger.info(f"True mean norm: {true_mean_norm:.6f}")
            logger.info(f"Estimated norm: {estimated_norm:.6f}")
            logger.info(f"Absolute error: {absolute_error:.6f}")
            logger.info(f"Relative error: {relative_error:.2f}%")
            
        return {
            'true_mean_norm': true_mean_norm,
            'estimated_norm': estimated_norm,
            'absolute_error': absolute_error,
            'relative_error': relative_error
        }


    @torch.no_grad()
    def update_norms(self):
        # Not running GGPO so not currently running update norms
        if self.ggpo_beta is None or self.ggpo_sigma is None:
            return

        # only update norms when we are training 
        if self.training is False:
            return

        module_weights = self.lora_up.weight @ self.lora_down.weight
        module_weights.mul(self.scale)

        self.weight_norms = torch.norm(module_weights, dim=1, keepdim=True)
        self.combined_weight_norms = torch.sqrt((self.org_weight_norm_estimate**2) + 
                                           torch.sum(module_weights**2, dim=1, keepdim=True))

    @torch.no_grad()
    def update_grad_norms(self):
        if self.training is False:
            print(f"skipping update_grad_norms for {self.lora_name}")
            return

        lora_down_grad = None
        lora_up_grad = None

        for name, param in self.named_parameters():
            if name == "lora_down.weight":
                lora_down_grad = param.grad
            elif name == "lora_up.weight":
                lora_up_grad = param.grad

        # Calculate gradient norms if we have both gradients
        if lora_down_grad is not None and lora_up_grad is not None:
            with torch.autocast(self.device.type):
                approx_grad = self.scale * ((self.lora_up.weight @ lora_down_grad) + (lora_up_grad @ self.lora_down.weight))
                self.grad_norms = torch.norm(approx_grad, dim=1, keepdim=True)


    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class LoRAInfModule(LoRAModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        # no dropout for inference
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha)

        self.org_module_ref = [org_module]  # 後から参照できるように
        self.enabled = True
        self.network: LoRANetwork = None

    def set_network(self, network):
        self.network = network

    # freezeしてマージする
    def merge_to(self, sd, dtype, device):
        # extract weight from org_module
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device
        weight = weight.to(torch.float)  # calc in float

        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        if self.split_dims is None:
            # get up/down weight
            down_weight = sd["lora_down.weight"].to(torch.float).to(device)
            up_weight = sd["lora_up.weight"].to(torch.float).to(device)

            # merge weight
            if len(weight.size()) == 2:
                # linear
                weight = weight + self.multiplier * (up_weight @ down_weight) * self.scale
            elif down_weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                weight = (
                    weight
                    + self.multiplier
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * self.scale
                )
            else:
                # conv2d 3x3
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                # logger.info(conved.size(), weight.size(), module.stride, module.padding)
                weight = weight + self.multiplier * conved * self.scale

            # set weight to org_module
            org_sd["weight"] = weight.to(dtype)
            self.org_module.load_state_dict(org_sd)
        else:
            # split_dims
            total_dims = sum(self.split_dims)
            for i in range(len(self.split_dims)):
                # get up/down weight
                down_weight = sd[f"lora_down.{i}.weight"].to(torch.float).to(device)  # (rank, in_dim)
                up_weight = sd[f"lora_up.{i}.weight"].to(torch.float).to(device)  # (split dim, rank)

                # pad up_weight -> (total_dims, rank)
                padded_up_weight = torch.zeros((total_dims, up_weight.size(0)), device=device, dtype=torch.float)
                padded_up_weight[sum(self.split_dims[:i]) : sum(self.split_dims[: i + 1])] = up_weight

                # merge weight
                weight = weight + self.multiplier * (up_weight @ down_weight) * self.scale

            # set weight to org_module
            org_sd["weight"] = weight.to(dtype)
            self.org_module.load_state_dict(org_sd)

    # 復元できるマージのため、このモジュールのweightを返す
    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        # get up/down weight from module
        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)

        # pre-calculated weight
        if len(down_weight.size()) == 2:
            # linear
            weight = self.multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                self.multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                * self.scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            weight = self.multiplier * conved * self.scale

        return weight

    def set_region(self, region):
        self.region = region
        self.region_mask = None

    def default_forward(self, x):
        # logger.info(f"default_forward {self.lora_name} {x.size()}")
        if self.split_dims is None:
            lx = self.lora_down(x)
            lx = self.lora_up(lx)
            return self.org_forward(x) + lx * self.multiplier * self.scale
        else:
            lxs = [lora_down(x) for lora_down in self.lora_down]
            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]
            return self.org_forward(x) + torch.cat(lxs, dim=-1) * self.multiplier * self.scale

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    ae: AutoencoderKL,
    text_encoders: List[CLIPTextModel],
    flux,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    # attn dim, mlp dim: only for DoubleStreamBlock. SingleStreamBlock is not supported because of combined qkv
    img_attn_dim = kwargs.get("img_attn_dim", None)
    txt_attn_dim = kwargs.get("txt_attn_dim", None)
    img_mlp_dim = kwargs.get("img_mlp_dim", None)
    txt_mlp_dim = kwargs.get("txt_mlp_dim", None)
    img_mod_dim = kwargs.get("img_mod_dim", None)
    txt_mod_dim = kwargs.get("txt_mod_dim", None)
    single_dim = kwargs.get("single_dim", None)  # SingleStreamBlock
    single_mod_dim = kwargs.get("single_mod_dim", None)  # SingleStreamBlock
    if img_attn_dim is not None:
        img_attn_dim = int(img_attn_dim)
    if txt_attn_dim is not None:
        txt_attn_dim = int(txt_attn_dim)
    if img_mlp_dim is not None:
        img_mlp_dim = int(img_mlp_dim)
    if txt_mlp_dim is not None:
        txt_mlp_dim = int(txt_mlp_dim)
    if img_mod_dim is not None:
        img_mod_dim = int(img_mod_dim)
    if txt_mod_dim is not None:
        txt_mod_dim = int(txt_mod_dim)
    if single_dim is not None:
        single_dim = int(single_dim)
    if single_mod_dim is not None:
        single_mod_dim = int(single_mod_dim)
    type_dims = [img_attn_dim, txt_attn_dim, img_mlp_dim, txt_mlp_dim, img_mod_dim, txt_mod_dim, single_dim, single_mod_dim]
    if all([d is None for d in type_dims]):
        type_dims = None

    # in_dims [img, time, vector, guidance, txt]
    in_dims = kwargs.get("in_dims", None)
    if in_dims is not None:
        in_dims = in_dims.strip()
        if in_dims.startswith("[") and in_dims.endswith("]"):
            in_dims = in_dims[1:-1]
        in_dims = [int(d) for d in in_dims.split(",")]  # is it better to use ast.literal_eval?
        assert len(in_dims) == 5, f"invalid in_dims: {in_dims}, must be 5 dimensions (img, time, vector, guidance, txt)"

    # double/single train blocks
    def parse_block_selection(selection: str, total_blocks: int) -> List[bool]:
        """
        Parse a block selection string and return a list of booleans.

        Args:
        selection (str): A string specifying which blocks to select.
        total_blocks (int): The total number of blocks available.

        Returns:
        List[bool]: A list of booleans indicating which blocks are selected.
        """
        if selection == "all":
            return [True] * total_blocks
        if selection == "none" or selection == "":
            return [False] * total_blocks

        selected = [False] * total_blocks
        ranges = selection.split(",")

        for r in ranges:
            if "-" in r:
                start, end = map(str.strip, r.split("-"))
                start = int(start)
                end = int(end)
                assert 0 <= start < total_blocks, f"invalid start index: {start}"
                assert 0 <= end < total_blocks, f"invalid end index: {end}"
                assert start <= end, f"invalid range: {start}-{end}"
                for i in range(start, end + 1):
                    selected[i] = True
            else:
                index = int(r)
                assert 0 <= index < total_blocks, f"invalid index: {index}"
                selected[index] = True

        return selected

    train_double_block_indices = kwargs.get("train_double_block_indices", None)
    train_single_block_indices = kwargs.get("train_single_block_indices", None)
    if train_double_block_indices is not None:
        train_double_block_indices = parse_block_selection(train_double_block_indices, NUM_DOUBLE_BLOCKS)
    if train_single_block_indices is not None:
        train_single_block_indices = parse_block_selection(train_single_block_indices, NUM_SINGLE_BLOCKS)

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    # single or double blocks
    train_blocks = kwargs.get("train_blocks", None)  # None (default), "all" (same as None), "single", "double"
    if train_blocks is not None:
        assert train_blocks in ["all", "single", "double"], f"invalid train_blocks: {train_blocks}"

    # split qkv
    split_qkv = kwargs.get("split_qkv", False)
    if split_qkv is not None:
        split_qkv = True if split_qkv == "True" else False

    ggpo_beta = kwargs.get("ggpo_beta", None)
    ggpo_sigma = kwargs.get("ggpo_sigma", None)

    if ggpo_beta is not None:
        ggpo_beta = float(ggpo_beta)

    if ggpo_sigma is not None:
        ggpo_sigma = float(ggpo_sigma)


    # train T5XXL
    train_t5xxl = kwargs.get("train_t5xxl", False)
    if train_t5xxl is not None:
        train_t5xxl = True if train_t5xxl == "True" else False

    # verbose
    verbose = kwargs.get("verbose", False)
    if verbose is not None:
        verbose = True if verbose == "True" else False

    # すごく引数が多いな ( ^ω^)･･･
    network = LoRANetwork(
        text_encoders,
        flux,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
        train_blocks=train_blocks,
        split_qkv=split_qkv,
        train_t5xxl=train_t5xxl,
        type_dims=type_dims,
        in_dims=in_dims,
        train_double_block_indices=train_double_block_indices,
        train_single_block_indices=train_single_block_indices,
        ggpo_beta=ggpo_beta,
        ggpo_sigma=ggpo_sigma,
        verbose=verbose,
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_unet_lr_ratio = kwargs.get("loraplus_unet_lr_ratio", None)
    loraplus_text_encoder_lr_ratio = kwargs.get("loraplus_text_encoder_lr_ratio", None)
    loraplus_lr_ratio = float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    loraplus_unet_lr_ratio = float(loraplus_unet_lr_ratio) if loraplus_unet_lr_ratio is not None else None
    loraplus_text_encoder_lr_ratio = float(loraplus_text_encoder_lr_ratio) if loraplus_text_encoder_lr_ratio is not None else None
    if loraplus_lr_ratio is not None or loraplus_unet_lr_ratio is not None or loraplus_text_encoder_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio)

    return network


# Create network from weights for inference, weights are not loaded here (because can be merged)
def create_network_from_weights(multiplier, file, ae, text_encoders, flux, weights_sd=None, for_inference=False, **kwargs):
    # if unet is an instance of SdxlUNet2DConditionModel or subclass, set is_sdxl to True
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    # get dim/alpha mapping, and train t5xxl
    modules_dim = {}
    modules_alpha = {}
    train_t5xxl = None
    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            dim = value.size()[0]
            modules_dim[lora_name] = dim
            # logger.info(lora_name, value.size(), dim)

        if train_t5xxl is None or train_t5xxl is False:
            train_t5xxl = "lora_te3" in lora_name

    if train_t5xxl is None:
        train_t5xxl = False

    # # split qkv
    # double_qkv_rank = None
    # single_qkv_rank = None
    # rank = None
    # for lora_name, dim in modules_dim.items():
    #     if "double" in lora_name and "qkv" in lora_name:
    #         double_qkv_rank = dim
    #     elif "single" in lora_name and "linear1" in lora_name:
    #         single_qkv_rank = dim
    #     elif rank is None:
    #         rank = dim
    #     if double_qkv_rank is not None and single_qkv_rank is not None and rank is not None:
    #         break
    # split_qkv = (double_qkv_rank is not None and double_qkv_rank != rank) or (
    #     single_qkv_rank is not None and single_qkv_rank != rank
    # )
    split_qkv = False  # split_qkv is not needed to care, because state_dict is qkv combined

    module_class = LoRAInfModule if for_inference else LoRAModule

    network = LoRANetwork(
        text_encoders,
        flux,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        split_qkv=split_qkv,
        train_t5xxl=train_t5xxl,
    )
    return network, weights_sd


class LoRANetwork(torch.nn.Module):
    FLUX_TARGET_REPLACE_MODULE_DOUBLE = ["DoubleStreamBlock"]
    FLUX_TARGET_REPLACE_MODULE_SINGLE = ["SingleStreamBlock"]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = {
        0: ["CLIPAttention", "CLIPMLP"],
        1: ["T5Attention", "T5DenseGatedActDense", "T5DenseActDense"]
    }
    LORA_PREFIX_FLUX = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER_CLIP = "lora_te1"
    LORA_PREFIX_TEXT_ENCODER_T5 = "lora_te3"

    def __init__(
        self,
        text_encoders: Union[List[Optional[torch.nn.Module]], Optional[torch.nn.Module]], # Adjusted for potential None
        unet: torch.nn.Module, # unet is SdxlUNet2DConditionModel (FLUX)
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        module_class: Type[Union[LoRAModule, LoRAInfModule]] = LoRAModule,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        train_blocks: Optional[str] = None,
        split_qkv: bool = False,
        train_t5xxl: bool = False,
        type_dims: Optional[List[int]] = None,
        in_dims: Optional[List[int]] = None,
        train_double_block_indices: Optional[List[bool]] = None,
        train_single_block_indices: Optional[List[bool]] = None,
        ggpo_beta: Optional[float] = None,
        ggpo_sigma: Optional[float] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.train_blocks = train_blocks if train_blocks is not None else "all"
        self.split_qkv = split_qkv
        self.train_t5xxl_lora = train_t5xxl # Renamed to avoid confusion

        self.type_dims = type_dims
        self.in_dims = in_dims
        self.train_double_block_indices = train_double_block_indices
        self.train_single_block_indices = train_single_block_indices
        self.ggpo_beta = ggpo_beta
        self.ggpo_sigma = ggpo_sigma
        self.verbose_creation = verbose # Renamed to avoid conflict

        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        if modules_dim is not None:
            logger.info(f"create LoRA network from weights")
            # if self.in_dims is None: # Only init if not already provided by weights (in_dims is part of network structure, not weights)
            #      self.in_dims = [0] * 5 # This should not be here, in_dims is a structural param
        else:
            logger.info(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
            logger.info(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
            )

        if ggpo_beta is not None and ggpo_sigma is not None:
            logger.info(f"LoRA-GGPO training sigma: {ggpo_sigma} beta: {ggpo_beta}")

        if self.split_qkv:
            logger.info(f"split qkv for LoRA")
        if self.train_blocks != "all": # Corrected condition from self.train_blocks is not None
            logger.info(f"train {self.train_blocks} blocks only")


        def create_modules(
            is_flux: bool,
            text_encoder_idx: Optional[int],
            root_module: torch.nn.Module,
            target_replace_modules: Optional[List[str]], # Can be None for in_proj
            filter_str: Optional[str] = None, 
            specific_dim: Optional[int] = None,
        ) -> Tuple[List[Union[LoRAModule, LoRAInfModule]], List[str]]:
            
            _loras_list = [] 
            _skipped_names_list = [] 

            _prefix = (
                self.LORA_PREFIX_FLUX
                if is_flux
                else (self.LORA_PREFIX_TEXT_ENCODER_CLIP if text_encoder_idx == 0 else self.LORA_PREFIX_TEXT_ENCODER_T5)
            )
            
            if root_module is None:
                return _loras_list, _skipped_names_list

            modules_to_iterate = []
            if target_replace_modules is None: # For in_proj or similar direct targeting
                 for child_name, child_module_obj in root_module.named_modules():
                     modules_to_iterate.append((child_name, child_module_obj, "")) # module_name_from_root, module_obj, parent_block_name=""
            else: # For targeting modules within specific block types
                for parent_block_name, parent_block_module in root_module.named_modules():
                    if parent_block_module.__class__.__name__ in target_replace_modules:
                        for child_module_name_in_block, child_module_obj in parent_block_module.named_modules():
                             modules_to_iterate.append((child_module_name_in_block, child_module_obj, parent_block_name))

            for name_of_module_in_parent, actual_module_to_lora, name_of_parent_block in modules_to_iterate:
                is_linear = actual_module_to_lora.__class__.__name__ == "Linear"
                is_conv2d = actual_module_to_lora.__class__.__name__ == "Conv2d"
                # FLUX doesn't have Conv2D, but keep for LoRAModule compatibility if it were to be used elsewhere
                is_conv2d_1x1 = is_conv2d and actual_module_to_lora.kernel_size == (1, 1)

                if is_linear or is_conv2d: # Only apply LoRA to these types
                    if name_of_parent_block: # Module is inside a specified block type
                        # name_of_module_in_parent is like "qkv" or "attn.qkv"
                        lora_name_suffix = name_of_parent_block + "." + name_of_module_in_parent
                    else: # Module is a direct child (target_replace_modules was None, e.g. in_proj)
                        # name_of_module_in_parent is already the full name relative to root_module (e.g., "time_in_proj")
                        lora_name_suffix = name_of_module_in_parent
                    
                    lora_name = _prefix + "." + lora_name_suffix
                    lora_name = lora_name.replace(".", "_")


                    if filter_str is not None and filter_str not in lora_name:
                        continue
                    
                    dim_to_use = None
                    alpha_to_use = None

                    if modules_dim is not None: # Loading from weights
                        if lora_name in modules_dim:
                            dim_to_use = modules_dim[lora_name]
                            # modules_alpha is also from __init__ scope
                            alpha_to_use = modules_alpha.get(lora_name, self.alpha) 
                    else: # Creating new network
                        if specific_dim is not None: # For in_proj or specific overrides
                            dim_to_use = specific_dim
                            alpha_to_use = self.alpha 
                        elif is_linear or is_conv2d_1x1: # General Linear or 1x1 Conv
                            dim_to_use = self.lora_dim
                            alpha_to_use = self.alpha
                            
                            if is_flux and self.type_dims is not None:
                                identifier = [
                                    ("img_attn",), ("txt_attn",), ("img_mlp",), ("txt_mlp",),
                                    ("img_mod",), ("txt_mod",), 
                                    ("single_blocks", "linear"), # For single_dim
                                    ("single_blocks", "modulation") # For single_mod_dim
                                ]
                                # Order of type_dims: img_attn, txt_attn, img_mlp, txt_mlp, img_mod, txt_mod, single_dim, single_mod_dim
                                # Check single_dim/single_mod_dim first if they are more specific
                                type_dim_candidates = [
                                    (self.type_dims[6], identifier[6]), # single_dim
                                    (self.type_dims[7], identifier[7]), # single_mod_dim
                                    (self.type_dims[0], identifier[0]), # img_attn
                                    (self.type_dims[1], identifier[1]), # txt_attn
                                    (self.type_dims[2], identifier[2]), # img_mlp
                                    (self.type_dims[3], identifier[3]), # txt_mlp
                                    (self.type_dims[4], identifier[4]), # img_mod
                                    (self.type_dims[5], identifier[5]), # txt_mod
                                ]
                                for d_val, id_tuple in type_dim_candidates:
                                    if d_val is not None and all(id_s in lora_name for id_s in id_tuple):
                                        dim_to_use = d_val
                                        # alpha_to_use could also be type-specific if needed
                                        break
                            
                            if (is_flux and dim_to_use is not None and dim_to_use > 0 and
                                (self.train_double_block_indices is not None or self.train_single_block_indices is not None) and
                                ("double_blocks" in lora_name or "single_blocks" in lora_name)):
                                try:
                                    # Extract block index, e.g. from lora_unet_double_blocks_0_...
                                    block_index_match = re.search(r"(?:double|single)_blocks_(\d+)", lora_name)
                                    if block_index_match:
                                        block_index = int(block_index_match.group(1))
                                        if ("double_blocks" in lora_name and self.train_double_block_indices is not None and
                                            not self.train_double_block_indices[block_index]):
                                            dim_to_use = 0
                                        elif ("single_blocks" in lora_name and self.train_single_block_indices is not None and
                                            not self.train_single_block_indices[block_index]):
                                            dim_to_use = 0
                                except (IndexError, ValueError) as e:
                                    logger.warning(f"Could not parse block index for {lora_name}: {e}")
                        elif is_conv2d and self.conv_lora_dim is not None: # For other Conv2d (e.g. 3x3), FLUX doesn't use these
                            dim_to_use = self.conv_lora_dim
                            alpha_to_use = self.conv_alpha if self.conv_alpha is not None else self.alpha
                    
                    if dim_to_use is None or dim_to_use == 0:
                        # Log skipped module only if it would have been targeted
                        if is_linear or is_conv2d_1x1 or (is_conv2d and (modules_dim is not None or self.conv_lora_dim is not None or specific_dim is not None)):
                             _skipped_names_list.append(lora_name)
                        continue
                    
                    current_alpha_val = alpha_to_use if alpha_to_use is not None else self.alpha

                    split_dims_for_lora = None
                    if is_flux and self.split_qkv and actual_module_to_lora.__class__.__name__ == "Linear":
                        # child_module is the Linear layer itself.
                        # name_of_module_in_parent is its name, e.g., "qkv" or "linear1"
                        if "double_blocks" in lora_name and "qkv" in name_of_module_in_parent: # name_of_module_in_parent is like 'attn.qkv'
                             # Assuming Q, K, V are concatenated and have equal dimensions
                            if actual_module_to_lora.out_features % 3 == 0:
                                split_dims_for_lora = [actual_module_to_lora.out_features // 3] * 3
                            else:
                                logger.warning(f"Cannot split QKV layer {lora_name} with out_features {actual_module_to_lora.out_features}")
                        elif "single_blocks" in lora_name and "linear1" in name_of_module_in_parent:
                            # For FLUX SingleStreamBlock, linear1 might be QKV + Proj_out
                            # Original code used hardcoded: [3072]*3 + [12288]
                            # This needs exact architectural knowledge of FLUX.
                            # If linear1 output is 3072*3 + 12288 = 9216 + 12288 = 21504
                            # This is highly specific. The prompt's example had 'pass'.
                            # Sticking to prompt's example of not splitting for single block linear1 by default.
                            pass 
                    
                    # module_class is from __init__ scope
                    lora_module = module_class( 
                        lora_name, actual_module_to_lora, self.multiplier,
                        dim_to_use, current_alpha_val,
                        dropout=self.dropout, rank_dropout=self.rank_dropout, module_dropout=self.module_dropout,
                        split_dims=split_dims_for_lora,
                        ggpo_beta=self.ggpo_beta, ggpo_sigma=self.ggpo_sigma,
                    )
                    _loras_list.append(lora_module)
            
            return _loras_list, _skipped_names_list


        self.text_encoder_loras: List[Optional[List[Union[LoRAModule, LoRAInfModule]]]] = []
        skipped_te_modules_total_count = 0 # Use a counter

        _text_encoders_list = text_encoders
        if not isinstance(text_encoders, list):
            _text_encoders_list = [text_encoders] if text_encoders is not None else []


        for i, text_encoder_model_part in enumerate(_text_encoders_list):
            text_encoder_idx_for_logging_and_prefix = i + 1 

            should_create_lora_for_this_te = True
            if text_encoder_model_part is None:
                if self.verbose_creation:
                    logger.info(f"Skipping LoRA for Text Encoder {text_encoder_idx_for_logging_and_prefix} as model part is None.")
                should_create_lora_for_this_te = False
            
            # Assuming index 0 is CLIP-L like, index 1 is T5XXL like for FLUX
            if i == 1: # T5XXL part
                if not self.train_t5xxl_lora: 
                    if self.verbose_creation:
                        logger.info(f"Skipping LoRA for Text Encoder {text_encoder_idx_for_logging_and_prefix} (assumed T5XXL) as train_t5xxl_lora is False.")
                    should_create_lora_for_this_te = False
            
            if not should_create_lora_for_this_te:
                self.text_encoder_loras.append(None)
                continue

            logger.info(f"Creating LoRA for Text Encoder {text_encoder_idx_for_logging_and_prefix}:")

            target_modules_for_current_te = LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE.get(i)

            if not target_modules_for_current_te:
                logger.warning(f"No target replace modules defined for Text Encoder index {i}. Skipping LoRA for TE {text_encoder_idx_for_logging_and_prefix}.")
                self.text_encoder_loras.append(None)
                continue
            
            loras_for_current_te_list, skipped_names_for_current_te = create_modules(
                is_flux=False, 
                text_encoder_idx=i, 
                root_module=text_encoder_model_part, 
                target_replace_modules=target_modules_for_current_te,
            )
            
            if self.verbose_creation and skipped_names_for_current_te:
                logger.info(f"Skipped {len(skipped_names_for_current_te)} LoRA modules for Text Encoder {text_encoder_idx_for_logging_and_prefix}: {', '.join(skipped_names_for_current_te)}")

            self.text_encoder_loras.append(loras_for_current_te_list if loras_for_current_te_list else None)
            skipped_te_modules_total_count += len(skipped_names_for_current_te)


        if self.train_blocks == "all":
            target_replace_modules_flux = LoRANetwork.FLUX_TARGET_REPLACE_MODULE_DOUBLE + LoRANetwork.FLUX_TARGET_REPLACE_MODULE_SINGLE
        elif self.train_blocks == "single":
            target_replace_modules_flux = LoRANetwork.FLUX_TARGET_REPLACE_MODULE_SINGLE
        elif self.train_blocks == "double":
            target_replace_modules_flux = LoRANetwork.FLUX_TARGET_REPLACE_MODULE_DOUBLE
        else: 
            target_replace_modules_flux = []


        self.unet_loras: List[Union[LoRAModule, LoRAInfModule]] = []
        skipped_un_modules_total_count = 0
        if target_replace_modules_flux: 
            unet_main_loras, unet_main_skipped_names = create_modules(
                is_flux=True, 
                text_encoder_idx=None, 
                root_module=unet, 
                target_replace_modules=target_replace_modules_flux,
            )
            self.unet_loras.extend(unet_main_loras)
            skipped_un_modules_total_count += len(unet_main_skipped_names)
            if self.verbose_creation and unet_main_skipped_names:
                 logger.info(f"Skipped {len(unet_main_skipped_names)} LoRA modules for UNet main blocks: {', '.join(unet_main_skipped_names)}")


        if self.in_dims:
            for filter_suffix, in_dim_val in zip(["_img_in", "_time_in", "_vector_in", "_guidance_in", "_txt_in"], self.in_dims):
                if in_dim_val > 0: 
                    in_proj_loras, skipped_in_proj_names = create_modules(
                        is_flux=True, text_encoder_idx=None, root_module=unet, 
                        target_replace_modules=None, # Target direct Linear/Conv2D layers
                        filter_str=filter_suffix, 
                        specific_dim=in_dim_val,
                    )
                    self.unet_loras.extend(in_proj_loras)
                    skipped_un_modules_total_count += len(skipped_in_proj_names)
                    if self.verbose_creation and skipped_in_proj_names:
                        logger.info(f"Skipped {len(skipped_in_proj_names)} LoRA modules for UNet in_proj (filter: {filter_suffix}): {', '.join(skipped_in_proj_names)}")


        logger.info(f"Create LoRA for FLUX {self.train_blocks} blocks: {len(self.unet_loras)} modules.")
        if self.verbose_creation:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:50} {lora.lora_dim}, {lora.alpha.item() if isinstance(lora.alpha, torch.Tensor) else lora.alpha}") # Safely get alpha value

        total_skipped_modules = skipped_te_modules_total_count + skipped_un_modules_total_count
        if total_skipped_modules > 0: # Always log if any modules were skipped due to dim 0 or filtering
            logger.warning(
                f"Because dim (rank) is 0 or modules filtered out, {total_skipped_modules} LoRA modules were skipped in total." 
            )
        
        all_created_loras = []
        for te_lora_list in self.text_encoder_loras:
            if te_lora_list: 
                all_created_loras.extend(te_lora_list)
        all_created_loras.extend(self.unet_loras)

        names = set()
        for lora_module_instance in all_created_loras:
            if lora_module_instance is None: continue 
            assert lora_module_instance.lora_name not in names, f"Duplicated LoRA name: {lora_module_instance.lora_name}"
            names.add(lora_module_instance.lora_name)


    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora_list in self.text_encoder_loras:
            if lora_list:
                for lora in lora_list:
                    lora.multiplier = self.multiplier
        for lora in self.unet_loras:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora_list in self.text_encoder_loras:
            if lora_list:
                for lora in lora_list:
                    lora.enabled = is_enabled
        for lora in self.unet_loras:
            lora.enabled = is_enabled

    def update_norms(self):
        for lora_list in self.text_encoder_loras:
            if lora_list:
                for lora in lora_list:
                    lora.update_norms()
        for lora in self.unet_loras:
            lora.update_norms()

    def update_grad_norms(self):
        for lora_list in self.text_encoder_loras:
            if lora_list:
                for lora in lora_list:
                    lora.update_grad_norms()
        for lora in self.unet_loras:
            lora.update_grad_norms()

    def grad_norms(self) -> Tensor | None:
        grad_norms = []
        for lora_list in self.text_encoder_loras:
            if lora_list:
                for lora in lora_list:
                    if hasattr(lora, "grad_norms") and lora.grad_norms is not None:
                        grad_norms.append(lora.grad_norms.mean(dim=0))
        for lora in self.unet_loras:
            if hasattr(lora, "grad_norms") and lora.grad_norms is not None:
                grad_norms.append(lora.grad_norms.mean(dim=0))
        return torch.stack(grad_norms) if len(grad_norms) > 0 else None

    def weight_norms(self) -> Tensor | None:
        weight_norms = []
        for lora_list in self.text_encoder_loras:
            if lora_list:
                for lora in lora_list:
                    if hasattr(lora, "weight_norms") and lora.weight_norms is not None:
                        weight_norms.append(lora.weight_norms.mean(dim=0))
        for lora in self.unet_loras:
            if hasattr(lora, "weight_norms") and lora.weight_norms is not None:
                weight_norms.append(lora.weight_norms.mean(dim=0))
        return torch.stack(weight_norms) if len(weight_norms) > 0 else None

    def combined_weight_norms(self) -> Tensor | None:
        combined_weight_norms = []
        for lora_list in self.text_encoder_loras:
            if lora_list:
                for lora in lora_list:
                    if hasattr(lora, "combined_weight_norms") and lora.combined_weight_norms is not None:
                        combined_weight_norms.append(lora.combined_weight_norms.mean(dim=0))
        for lora in self.unet_loras:
            if hasattr(lora, "combined_weight_norms") and lora.combined_weight_norms is not None:
                combined_weight_norms.append(lora.combined_weight_norms.mean(dim=0))
        return torch.stack(combined_weight_norms) if len(combined_weight_norms) > 0 else None


    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        info = self.load_state_dict(weights_sd, False)
        return info

    def load_state_dict(self, state_dict, strict=True):
        # override to convert original weight to split qkv
        if not self.split_qkv:
            return super().load_state_dict(state_dict, strict)

        # split qkv
        for key in list(state_dict.keys()):
            if "double" in key and "qkv" in key:
                split_dims = [3072] * 3
            elif "single" in key and "linear1" in key:
                split_dims = [3072] * 3 + [12288]
            else:
                continue

            weight = state_dict[key]
            lora_name = key.split(".")[0]
            if "lora_down" in key and "weight" in key:
                # dense weight (rank*3, in_dim)
                split_weight = torch.chunk(weight, len(split_dims), dim=0)
                for i, split_w in enumerate(split_weight):
                    state_dict[f"{lora_name}.lora_down.{i}.weight"] = split_w

                del state_dict[key]
                # print(f"split {key}: {weight.shape} to {[w.shape for w in split_weight]}")
            elif "lora_up" in key and "weight" in key:
                # sparse weight (out_dim=sum(split_dims), rank*3)
                rank = weight.size(1) // len(split_dims)
                i = 0
                for j in range(len(split_dims)):
                    state_dict[f"{lora_name}.lora_up.{j}.weight"] = weight[i : i + split_dims[j], j * rank : (j + 1) * rank]
                    i += split_dims[j]
                del state_dict[key]

                # # check is sparse
                # i = 0
                # is_zero = True
                # for j in range(len(split_dims)):
                #     for k in range(len(split_dims)):
                #         if j == k:
                #             continue
                #         is_zero = is_zero and torch.all(weight[i : i + split_dims[j], k * rank : (k + 1) * rank] == 0)
                #     i += split_dims[j]
                # if not is_zero:
                #     logger.warning(f"weight is not sparse: {key}")
                # else:
                #     logger.info(f"weight is sparse: {key}")

                # print(
                #     f"split {key}: {weight.shape} to {[state_dict[k].shape for k in [f'{lora_name}.lora_up.{j}.weight' for j in range(len(split_dims))]]}"
                # )

            # alpha is unchanged

        return super().load_state_dict(state_dict, strict)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if not self.split_qkv:
            return super().state_dict(destination, prefix, keep_vars)

        # merge qkv
        state_dict = super().state_dict(destination, prefix, keep_vars)
        new_state_dict = {}
        for key in list(state_dict.keys()):
            if "double" in key and "qkv" in key:
                split_dims = [3072] * 3
            elif "single" in key and "linear1" in key:
                split_dims = [3072] * 3 + [12288]
            else:
                new_state_dict[key] = state_dict[key]
                continue

            if key not in state_dict:
                continue  # already merged

            lora_name = key.split(".")[0]

            # (rank, in_dim) * 3
            down_weights = [state_dict.pop(f"{lora_name}.lora_down.{i}.weight") for i in range(len(split_dims))]
            # (split dim, rank) * 3
            up_weights = [state_dict.pop(f"{lora_name}.lora_up.{i}.weight") for i in range(len(split_dims))]

            alpha = state_dict.pop(f"{lora_name}.alpha")

            # merge down weight
            down_weight = torch.cat(down_weights, dim=0)  # (rank, split_dim) * 3 -> (rank*3, sum of split_dim)

            # merge up weight (sum of split_dim, rank*3)
            rank = up_weights[0].size(1)
            up_weight = torch.zeros((sum(split_dims), down_weight.size(0)), device=down_weight.device, dtype=down_weight.dtype)
            i = 0
            for j in range(len(split_dims)):
                up_weight[i : i + split_dims[j], j * rank : (j + 1) * rank] = up_weights[j]
                i += split_dims[j]

            new_state_dict[f"{lora_name}.lora_down.weight"] = down_weight
            new_state_dict[f"{lora_name}.lora_up.weight"] = up_weight
            new_state_dict[f"{lora_name}.alpha"] = alpha

            # print(
            #     f"merged {lora_name}: {lora_name}, {[w.shape for w in down_weights]}, {[w.shape for w in up_weights]} to {down_weight.shape}, {up_weight.shape}"
            # )
            print(f"new key: {lora_name}.lora_down.weight, {lora_name}.lora_up.weight, {lora_name}.alpha")

        return new_state_dict

    def apply_to(self, text_encoders, flux, apply_text_encoder=True, apply_unet=True):
        all_loras_to_apply = []
        if apply_text_encoder:
            logger.info(f"enable LoRA for text encoder") 
            for i, lora_list in enumerate(self.text_encoder_loras):
                if lora_list:
                    logger.info(f"Text Encoder {i+1}: {len(lora_list)} modules enabled.")
                    all_loras_to_apply.extend(lora_list)
        else:
            logger.info("Skipping LoRA for text encoders.")


        if apply_unet:
            logger.info(f"enable LoRA for U-Net: {len(self.unet_loras)} modules")
            all_loras_to_apply.extend(self.unet_loras)
        else:
            logger.info("Skipping LoRA for U-Net.")


        for lora in all_loras_to_apply:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def is_mergeable(self):
        return True

    def merge_to(self, text_encoders, flux, weights_sd, dtype=None, device=None):
        apply_text_encoder_loras = False
        apply_unet_loras = False
        for key in weights_sd.keys():
            if key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER_CLIP) or \
               key.startswith(LoRANetwork.LORA_PREFIX_TEXT_ENCODER_T5):
                apply_text_encoder_loras = True
            elif key.startswith(LoRANetwork.LORA_PREFIX_FLUX):
                apply_unet_loras = True
        
        all_loras_for_merge = []
        if apply_text_encoder_loras:
            logger.info("Merging LoRA for text encoders.")
            for lora_list in self.text_encoder_loras:
                if lora_list:
                    all_loras_for_merge.extend(lora_list)
        
        if apply_unet_loras:
            logger.info("Merging LoRA for U-Net.")
            all_loras_for_merge.extend(self.unet_loras)

        for lora in all_loras_for_merge:
            has_weights_for_this_lora = any(key.startswith(lora.lora_name) for key in weights_sd.keys())
            if not has_weights_for_this_lora:
                continue

            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[key]
            
            if sd_for_lora:
                 lora.merge_to(sd_for_lora, dtype, device)

        logger.info(f"weights are merged")

    def set_loraplus_lr_ratio(self, loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio):
        self.loraplus_lr_ratio = loraplus_lr_ratio
        self.loraplus_unet_lr_ratio = loraplus_unet_lr_ratio
        self.loraplus_text_encoder_lr_ratio = loraplus_text_encoder_lr_ratio

        logger.info(f"LoRA+ UNet LR Ratio: {self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio}")
        logger.info(f"LoRA+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}")

    def prepare_optimizer_params_with_multiple_te_lrs(self, text_encoder_lr, unet_lr, default_lr):
        if text_encoder_lr is None or (isinstance(text_encoder_lr, list) and len(text_encoder_lr) == 0):
            text_encoder_lr = [default_lr, default_lr]
        elif isinstance(text_encoder_lr, (float, int)):
            text_encoder_lr = [float(text_encoder_lr), float(text_encoder_lr)]
        elif len(text_encoder_lr) == 1:
            text_encoder_lr = [text_encoder_lr[0], text_encoder_lr[0]]

        self.requires_grad_(True)

        all_params = []
        lr_descriptions = []

        def assemble_params(loras_list, lr_val, loraplus_ratio_val):
            param_groups = {"lora": {}, "plus": {}}
            if not loras_list: 
                return [], []

            for lora_module in loras_list:
                for name, param in lora_module.named_parameters():
                    if loraplus_ratio_val is not None and "lora_up" in name:
                        param_groups["plus"][f"{lora_module.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora_module.lora_name}.{name}"] = param

            current_params_list = []
            current_descriptions_list = []
            for key in param_groups.keys():
                if not param_groups[key]:
                    continue

                param_data = {"params": list(param_groups[key].values())} 

                current_lr = None
                if lr_val is not None:
                    if key == "plus":
                        current_lr = lr_val * loraplus_ratio_val
                    else:
                        current_lr = lr_val
                
                if current_lr is None or current_lr == 0 : 
                    # If default_lr itself is 0, this path might be taken more often.
                    # Only log if lr_val was non-zero initially and became zero, or if explicitly skipping.
                    if (lr_val is not None and lr_val != 0) or current_lr == 0:
                         logger.info(f"Skipping param group with LR {current_lr} for {key} (original LR: {lr_val})")
                    continue
                
                param_data["lr"] = current_lr
                current_params_list.append(param_data)
                current_descriptions_list.append("plus" if key == "plus" else "")
            
            return current_params_list, current_descriptions_list
        
        loraplus_te_ratio = self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio

        if len(self.text_encoder_loras) > 0 and self.text_encoder_loras[0]:
            te1_loras = self.text_encoder_loras[0]
            te1_lr = text_encoder_lr[0] if text_encoder_lr and len(text_encoder_lr) > 0 else default_lr
            logger.info(f"Text Encoder 1 (CLIP-L like): {len(te1_loras)} modules, LR {te1_lr}")
            params, descriptions = assemble_params(te1_loras, te1_lr, loraplus_te_ratio)
            all_params.extend(params)
            lr_descriptions.extend(["textencoder 1 " + (d if d else "") for d in descriptions])
        
        if len(self.text_encoder_loras) > 1 and self.text_encoder_loras[1]:
            te3_loras = self.text_encoder_loras[1] 
            te3_lr = text_encoder_lr[1] if text_encoder_lr and len(text_encoder_lr) > 1 else default_lr
            logger.info(f"Text Encoder 2 (T5XXL like): {len(te3_loras)} modules, LR {te3_lr}")
            params, descriptions = assemble_params(te3_loras, te3_lr, loraplus_te_ratio)
            all_params.extend(params)
            lr_descriptions.extend(["textencoder 2 " + (d if d else "") for d in descriptions])


        if self.unet_loras:
            loraplus_unet_ratio = self.loraplus_unet_lr_ratio or self.loraplus_lr_ratio
            unet_actual_lr = unet_lr if unet_lr is not None else default_lr
            logger.info(f"U-Net (FLUX): {len(self.unet_loras)} modules, LR {unet_actual_lr}")
            params, descriptions = assemble_params(
                self.unet_loras,
                unet_actual_lr,
                loraplus_unet_ratio,
            )
            all_params.extend(params)
            lr_descriptions.extend(["unet" + (" " + d if d else "") for d in descriptions])

        return all_params, lr_descriptions

    def enable_gradient_checkpointing(self):
        pass

    def prepare_grad_etc(self, text_encoder, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, text_encoder, unet):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from library import train_util

            if metadata is None:
                metadata = {}
            model_hash, legacy_hash = train_util.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def backup_weights(self):
        all_loras_for_backup_restore = []
        for lora_list in self.text_encoder_loras:
            if lora_list: all_loras_for_backup_restore.extend(lora_list)
        all_loras_for_backup_restore.extend(self.unet_loras)
        
        for lora in all_loras_for_backup_restore:
            if not isinstance(lora, LoRAInfModule): continue
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                sd = org_module.state_dict()
                org_module._lora_org_weight = sd["weight"].detach().clone()
                org_module._lora_restored = True 

    def restore_weights(self):
        all_loras_for_backup_restore = []
        for lora_list in self.text_encoder_loras:
            if lora_list: all_loras_for_backup_restore.extend(lora_list)
        all_loras_for_backup_restore.extend(self.unet_loras)

        for lora in all_loras_for_backup_restore:
            if not isinstance(lora, LoRAInfModule): continue
            org_module = lora.org_module_ref[0]
            if hasattr(org_module, "_lora_org_weight") and (not hasattr(org_module, "_lora_restored") or not org_module._lora_restored) :
                sd = org_module.state_dict()
                sd["weight"] = org_module._lora_org_weight
                org_module.load_state_dict(sd)
                org_module._lora_restored = True

    def pre_calculation(self):
        all_loras_for_precalc = []
        for lora_list in self.text_encoder_loras:
            if lora_list: all_loras_for_precalc.extend(lora_list)
        all_loras_for_precalc.extend(self.unet_loras)

        for lora in all_loras_for_precalc:
            if not isinstance(lora, LoRAInfModule): continue
            org_module = lora.org_module_ref[0]
            sd = org_module.state_dict()

            org_weight = sd["weight"]
            lora_weight = lora.get_weight().to(org_weight.device, dtype=org_weight.dtype)
            sd["weight"] = org_weight + lora_weight
            assert sd["weight"].shape == org_weight.shape
            org_module.load_state_dict(sd)

            org_module._lora_restored = False 
            lora.enabled = False 

    def apply_max_norm_regularization(self, max_norm_value, device):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        # Use named_parameters to get references to actual parameters
        # This requires mapping lora_name to module.
        lora_modules_map = {name: mod for name, mod in self.named_modules() if isinstance(mod, (LoRAModule, LoRAInfModule))}

        for lora_name, lora_module in lora_modules_map.items():
            if lora_module.split_dims is not None: # Max norm for split_dims not directly handled here, skip or adapt
                # logger.warning(f"Max norm for LoRA with split_dims ({lora_name}) is not implemented, skipping.")
                continue

            down_param = lora_module.lora_down.weight
            up_param = lora_module.lora_up.weight
            alpha_val = lora_module.alpha.item() # Get Python float from tensor
            
            dim = down_param.shape[0] # Rank
            scale = alpha_val / dim

            # Calculate effective weight without in-place conv2d if possible
            # Use float32 for norm calculation precision
            down = down_param.to(device, dtype=torch.float32)
            up = up_param.to(device, dtype=torch.float32)

            if up.ndim == 4 and down.ndim == 4 : # Conv2D
                if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1): # Conv1x1
                    # (out_C, rank, 1, 1) @ (rank, in_C, 1, 1) -> (out_C, rank) @ (rank, in_C)
                    updown = (up.squeeze(3).squeeze(2) @ down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                else: # Conv3x3 or other
                    # This requires convolving weights, which is tricky for norm.
                    # For simplicity, approximate or skip. Here, we try to estimate.
                    # Permute down to (in_C, rank, K, K) then conv with up (out_C, rank, 1, 1) if up is 1x1,
                    # or if up is also (out_C, rank, K', K'), it's more complex.
                    # LoRA typically uses 1x1 for up. Assuming up is (out_C, rank, 1, 1)
                    # updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up.permute(1,0,2,3).transpose(0,1)).permute(1, 0, 2, 3) - this is not right.
                    # The norm of W = W_A W_B is tricky for convs.
                    # A common approximation is to norm W_A and W_B separately or their product if linear.
                    # For convs, this is harder. Let's skip complex conv cases for now or use a simpler norm.
                    # For now, let's assume we are mostly dealing with Linear or Conv1x1 in terms of LoRA structure.
                    # If actual 3x3 conv LoRAs are used, this part needs refinement.
                    # Fallback to Linear-like product for norm estimation for non-1x1 conv
                    # This is a rough approximation for non-1x1 conv.
                    updown = up.flatten(1) @ down.flatten(1) # (Out, Rank*K*K) @ (Rank*K*K, In) -> (Out, In) -- this is not quite right
                                                            # Let's use a Frobenius norm on each and combine, or norm of product for Linear.
                                                            # Reverting to simpler product for norm, hoping it's mostly Linear.
                    logger.warning(f"Max norm for non-1x1 Conv2D LoRA ({lora_name}) uses a simplified norm. Accurate norm calculation is complex.")
                    # For Linear layers that might be shaped as Conv.
                    if up.shape[2:] == (1,1) and down.shape[2:] == (1,1): # Reshaped Linear
                         updown = (up.squeeze(3).squeeze(2) @ down.squeeze(3).squeeze(2))
                    else: # True conv, this norm is an approximation
                         updown = up.reshape(up.shape[0], -1) @ down.reshape(down.shape[0], -1).T # (Out_C, R*k*k) @ (In_C, R*k*k).T is not right.
                                                                                                # (Out_C, R) @ (R, In_C) for Linear part
                         # A simple placeholder: norm the parameters individually.
                         # This is not ideal. For true convs, norm of effective delta is hard.
                         # Let's assume LoRA on convs means lora_down output is spatial, lora_up is 1x1.
                         # So lora_down (R, C_in, k, k), lora_up (C_out, R, 1, 1)
                         # Effective W_delta_c = sum_r lora_up_c,r * lora_down_r (convolution kernel)
                         # This is too complex for here. For now, only Linear and Conv1x1 are accurately handled.
                         # Skip complex conv if not 1x1 for LoRA up
                         if not (up.shape[2:] == (1,1)):
                             #logger.debug(f"Skipping max_norm for complex Conv2D LoRA: {lora_name}")
                             continue # Skip this lora module for max_norm
                         else: # up is 1x1, down is kxk
                             # Treat as grouped linear ops for norm approximation
                             temp_down = down.reshape(down.shape[0], -1) # (R, C_in*k*k)
                             temp_up = up.squeeze(3).squeeze(2) # (C_out, R)
                             updown = temp_up @ temp_down # (C_out, C_in*k*k)
            else: # Linear
                updown = up @ down

            updown = updown * scale
            norm = torch.norm(updown)
            desired = torch.clamp(norm, max=max_norm_value)

            if norm > max_norm_value and norm != 0:
                ratio = desired / norm
                sqrt_ratio = ratio**0.5
                
                with torch.no_grad():
                    down_param.mul_(sqrt_ratio)
                    up_param.mul_(sqrt_ratio)
                
                keys_scaled += 1
                scalednorm = norm * ratio 
            else:
                scalednorm = norm

            norms.append(scalednorm.item())
        
        if not norms:
            return 0, 0.0, 0.0

        return keys_scaled, sum(norms) / len(norms) if norms else 0.0, max(norms) if norms else 0.0