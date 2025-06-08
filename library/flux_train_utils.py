import argparse
import math
import os
import numpy as np
import toml
import json
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, PartialState
from transformers import CLIPTextModel
from tqdm import tqdm
from PIL import Image
from safetensors.torch import save_file
from einops import rearrange # Required by vae_flatten/unflatten
from scipy.optimize import linear_sum_assignment # For cosine_optimal_transport fallback


from library import flux_models, flux_utils, strategy_base, train_util, strategy_flux # Added strategy_flux for type hints
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from .utils import setup_logging, mem_eff_save_file

setup_logging()
import logging

logger = logging.getLogger(__name__)

# region Chroma SOT Loss Utilities (from user prompt)

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    shift_constant = image_seq_len * m + b
    return shift_constant

def time_shift(shift_constant: float, timesteps: torch.Tensor, sigma: float = 1):
    # Ensure timesteps are not exactly 0 to avoid division by zero if 1/timesteps is used.
    # Small epsilon to prevent issues, though the formula (1/timesteps - 1) should handle t=1 correctly.
    # For t=0, (1/timesteps) would be Inf. The custom distribution should avoid exact 0.
    timesteps = torch.clamp(timesteps, min=1e-9)
    return math.exp(shift_constant) / (
        math.exp(shift_constant) + (1 / timesteps - 1) ** sigma
    )

def vae_flatten(latents):
    # nchw to nhwc then pixel shuffle 2 then flatten
    # n c h w -> n h w c
    # n (h dh) (w dw) c -> n h w (c dh dw)
    # n h w c -> n (h w) c
    # n, c, h, w = latents.shape
    return (
        rearrange(latents, "n c (h dh) (w dw) -> n (h w) (c dh dw)", dh=2, dw=2),
        latents.shape,
    )

def vae_unflatten(latents, shape):
    # reverse of that operator above
    n, c_orig, h_orig, w_orig = shape # c_orig is original channels before flattening (e.g. 4 for VAE)
    # latents are (n, (h_orig/2 * w_orig/2), c_orig * dh * dw)
    # We need to ensure c in rearrange matches c_orig.
    # The number of channels in the flattened latents is c_orig * dh * dw.
    # So, the 'c' passed to rearrange should be c_orig.
    return rearrange(
        latents,
        "n (h w) (c dh dw) -> n c (h dh) (w dw)",
        dh=2,
        dw=2,
        c=c_orig, # Use original number of channels for 'c'
        h=h_orig // 2,
        w=w_orig // 2,
    )

def prepare_latent_image_ids(batch_size, height, width):
    # pos embedding for rope, 2d pos embedding, corner embedding and not center based
    # height and width are original VAE latent height/width
    latent_image_ids = torch.zeros(height // 2, width // 2, 3) # Creates for H/2, W/2 grid
    latent_image_ids[..., 1] = (
        latent_image_ids[..., 1] + torch.arange(height // 2)[:, None]
    )
    latent_image_ids[..., 2] = (
        latent_image_ids[..., 2] + torch.arange(width // 2)[None, :]
    )

    (
        latent_image_id_height, # H/2
        latent_image_id_width,  # W/2
        latent_image_id_channels, # 3
    ) = latent_image_ids.shape

    latent_image_ids = latent_image_ids[None, :].repeat(batch_size, 1, 1, 1) # (B, H/2, W/2, 3)
    latent_image_ids = latent_image_ids.reshape(
        batch_size,
        latent_image_id_height * latent_image_id_width, # L_flat = (H/2 * W/2)
        latent_image_id_channels, # 3
    )
    return latent_image_ids

def create_distribution(num_points, device=None):
    x = torch.linspace(0, 1, num_points, device=device)
    probabilities = -7.7 * ((x - 0.5) ** 2) + 2
    probabilities /= probabilities.sum()
    return x, probabilities

def sample_from_distribution(x, probabilities, num_samples, device=None):
    cdf = torch.cumsum(probabilities, dim=0)
    uniform_samples = torch.rand(num_samples, device=device)
    indices = torch.searchsorted(cdf, uniform_samples, right=True)
    sampled_values = x[indices]
    return sampled_values

def _cuda_assignment(C):
    try:
        from torch_linear_assignment import batch_linear_assignment, assignment_to_indices
        assignment = batch_linear_assignment(C.unsqueeze(dim=0))
        row_indices, col_indices = assignment_to_indices(assignment)
        matching_pairs = (row_indices, col_indices)
        return C, matching_pairs
    except ImportError as e:
        logger.warning("torch_linear_assignment not found. Install it for CUDA OT. Falling back to SciPy if CUDA fails.")
        raise e # Re-raise to trigger fallback in cosine_optimal_transport

def _scipy_assignment(C):
    C_np = C.to(torch.float32).detach().cpu().numpy() # Ensure float32 for SciPy
    row_ind, col_ind = linear_sum_assignment(C_np)
    matching_pairs = (
        torch.tensor([row_ind], device=C.device, dtype=torch.long), # Ensure long for indexing
        torch.tensor([col_ind], device=C.device, dtype=torch.long),
    )
    return C, matching_pairs

def cosine_optimal_transport(X, Y, backend="auto"):
    X_norm = X / torch.norm(X, dim=1, keepdim=True).clamp(min=1e-6)
    Y_norm = Y / torch.norm(Y, dim=1, keepdim=True).clamp(min=1e-6)
    if torch.isnan(X_norm).any() or torch.isinf(X_norm).any():
        logger.error("SOT: NaN/Inf in X_norm!")
        # Consider dumping X here if this happens
    if torch.isnan(Y_norm).any() or torch.isinf(Y_norm).any():
        logger.error("SOT: NaN/Inf in Y_norm!")
    C = -torch.mm(X_norm, Y_norm.t())
    if torch.isnan(C).any() or torch.isinf(C).any():
        logger.error(f"SOT: NaN/Inf in Cost Matrix C! C min: {C.min()}, C max: {C.max()}")

    if backend == "scipy": return _scipy_assignment(C)
    elif backend == "cuda": return _cuda_assignment(C)
    else: # auto
        try: return _cuda_assignment(C)
        except (ImportError, RuntimeError): return _scipy_assignment(C)

def prepare_sot_pairings(latents_bhwc: torch.Tensor, device: torch.device):
    logger.debug(f"SOT PREP: Input latents_bhwc shape: {latents_bhwc.shape}, dtype: {latents_bhwc.dtype}, NaN: {torch.isnan(latents_bhwc).any()}")
    latents_bhwc = latents_bhwc.to(device=device, dtype=torch.float32) # Ensure float32 for SOT math
    latents_flat, latent_shape_tuple = vae_flatten(latents_bhwc)
    logger.debug(f"SOT PREP: latents_flat shape: {latents_flat.shape}, NaN: {torch.isnan(latents_flat).any()}")

    n_batch, num_tokens, channels_flat = latents_flat.shape
    _, _, h_orig, w_orig = latent_shape_tuple
    image_pos_id_b_tokens_3 = prepare_latent_image_ids(n_batch, h_orig, w_orig).to(device)

    num_points_dist = 1000
    x_dist, probabilities_dist = create_distribution(num_points_dist, device=device)
    input_timestep_b = sample_from_distribution(x_dist, probabilities_dist, n_batch, device=device)
    logger.debug(f"SOT PREP: input_timestep_b min: {input_timestep_b.min()}, max: {input_timestep_b.max()}, NaN: {torch.isnan(input_timestep_b).any()}")
    timesteps_b_1_1 = input_timestep_b[:, None, None]

    noise_b_tokens_c = torch.randn_like(latents_flat)
    logger.debug(f"SOT PREP: initial noise_b_tokens_c NaN: {torch.isnan(noise_b_tokens_c).any()}")
    
    reshaped_latents_flat = latents_flat.reshape(n_batch, -1)
    reshaped_noise = noise_b_tokens_c.reshape(n_batch, -1)
    logger.debug(f"SOT PREP: reshaped_latents_flat for OT NaN: {torch.isnan(reshaped_latents_flat).any()}")
    logger.debug(f"SOT PREP: reshaped_noise for OT NaN: {torch.isnan(reshaped_noise).any()}")

    _, indices = cosine_optimal_transport(reshaped_latents_flat, reshaped_noise)
    
    noise_indices_for_pairing = indices[1].view(-1)
    noise_b_tokens_c_ot_paired = noise_b_tokens_c[noise_indices_for_pairing]
    logger.debug(f"SOT PREP: noise_b_tokens_c_ot_paired NaN: {torch.isnan(noise_b_tokens_c_ot_paired).any()}")

    noisy_latents_b_tokens_c = latents_flat * (1 - timesteps_b_1_1) + noise_b_tokens_c_ot_paired * timesteps_b_1_1
    logger.debug(f"SOT PREP: noisy_latents_b_tokens_c NaN: {torch.isnan(noisy_latents_b_tokens_c).any()}, Inf: {torch.isinf(noisy_latents_b_tokens_c).any()}")
    
    target_b_tokens_c = noise_b_tokens_c_ot_paired - latents_flat
    logger.debug(f"SOT PREP: target_b_tokens_c NaN: {torch.isnan(target_b_tokens_c).any()}, Inf: {torch.isinf(target_b_tokens_c).any()}")

    return (
        noisy_latents_b_tokens_c.to(torch.bfloat16),
        target_b_tokens_c.to(torch.bfloat16),
        input_timestep_b.to(torch.bfloat16),
        image_pos_id_b_tokens_3,
        latent_shape_tuple,
    )

def calculate_chroma_loss_step(
    model_prediction: torch.Tensor,
    target_for_minibatch: torch.Tensor,
    loss_weighting_for_minibatch: torch.Tensor,
    total_batch_size_for_loss_norm: int,
    current_minibatch_size: int
):
    logger.debug(f"SOT LOSS: model_prediction shape: {model_prediction.shape}, NaN: {torch.isnan(model_prediction).any()}, Inf: {torch.isinf(model_prediction).any()}")
    logger.debug(f"SOT LOSS: target_for_minibatch shape: {target_for_minibatch.shape}, NaN: {torch.isnan(target_for_minibatch).any()}, Inf: {torch.isinf(target_for_minibatch).any()}")
    logger.debug(f"SOT LOSS: loss_weighting_for_minibatch: {loss_weighting_for_minibatch}, NaN: {torch.isnan(loss_weighting_for_minibatch).any()}")

    # Ensure inputs are float32 for stable loss calculation, then cast back if needed.
    # model_prediction and target_for_minibatch are already cast to bfloat16 in prepare_sot_pairings return
    # So, they might need to be .float() here for the squared error.
    loss_per_sample = ((model_prediction.float() - target_for_minibatch.float()) ** 2)
    logger.debug(f"SOT LOSS: loss_per_sample (before mean) NaN: {torch.isnan(loss_per_sample).any()}, Inf: {torch.isinf(loss_per_sample).any()}")
    loss_per_sample = loss_per_sample.mean(dim=(1, 2))
    logger.debug(f"SOT LOSS: loss_per_sample (after mean): {loss_per_sample}, NaN: {torch.isnan(loss_per_sample).any()}")
    
    if total_batch_size_for_loss_norm == 0: num_minibatches_in_total_batch = 1.0
    else: num_minibatches_in_total_batch = total_batch_size_for_loss_norm / current_minibatch_size
    if num_minibatches_in_total_batch == 0 : num_minibatches_in_total_batch = 1.0

    loss_per_sample_normalized = loss_per_sample / num_minibatches_in_total_batch
    logger.debug(f"SOT LOSS: loss_per_sample_normalized: {loss_per_sample_normalized}, NaN: {torch.isnan(loss_per_sample_normalized).any()}")

    weights = loss_weighting_for_minibatch.to(device=loss_per_sample_normalized.device, dtype=loss_per_sample_normalized.dtype)
    weights_sum = weights.sum()
    logger.debug(f"SOT LOSS: weights_sum: {weights_sum}")

    if weights_sum.abs() > 1e-6: # check absolute value for sum being too small
        normalized_minibatch_weights = weights / weights_sum
    else:
        logger.warning("SOT LOSS: loss_weighting_for_minibatch sum is close to zero. Using uniform weights for this minibatch.")
        normalized_minibatch_weights = torch.ones_like(weights) / (weights.numel() if weights.numel() > 0 else 1.0)
    logger.debug(f"SOT LOSS: normalized_minibatch_weights: {normalized_minibatch_weights}, NaN: {torch.isnan(normalized_minibatch_weights).any()}")
    
    final_loss = (loss_per_sample_normalized * normalized_minibatch_weights).sum()
    logger.debug(f"SOT LOSS: final_loss: {final_loss.item()}, NaN: {torch.isnan(final_loss).any()}")
    return final_loss
# region sample images (existing code)
def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    flux_model, # Renamed to avoid conflict
    ae,
    text_encoders,
    sample_prompts_te_outputs,
    prompt_replacement=None,
    controlnet=None,
):
    if steps == 0:
        if not args.sample_at_first: return
    else:
        if args.sample_every_n_steps is None and args.sample_every_n_epochs is None: return
        if args.sample_every_n_epochs is not None:
            if epoch is None or epoch % args.sample_every_n_epochs != 0: return
        else:
            if steps % args.sample_every_n_steps != 0 or epoch is not None: return

    logger.info(f"generating sample images at step: {steps}")
    if not os.path.isfile(args.sample_prompts) and sample_prompts_te_outputs is None:
        logger.error(f"No prompt file: {args.sample_prompts}"); return

    distributed_state = PartialState()
    flux_unwrapped = accelerator.unwrap_model(flux_model)
    text_encoders_unwrapped = []
    if text_encoders is not None:
        for te in text_encoders: text_encoders_unwrapped.append(accelerator.unwrap_model(te) if te is not None else None)
    
    controlnet_unwrapped = accelerator.unwrap_model(controlnet) if controlnet is not None else None
    prompts = train_util.load_prompts(args.sample_prompts)
    save_dir = os.path.join(args.output_dir, "sample")
    os.makedirs(save_dir, exist_ok=True)

    rng_state_cpu = torch.get_rng_state()
    rng_state_gpu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None # Get all GPU states

    if distributed_state.num_processes <= 1:
        with torch.no_grad(), accelerator.autocast():
            for prompt_dict in prompts:
                sample_image_inference(accelerator, args, flux_unwrapped, text_encoders_unwrapped, ae, save_dir, prompt_dict, epoch, steps, sample_prompts_te_outputs, prompt_replacement, controlnet_unwrapped)
    else:
        per_process_prompts = [prompts[i::distributed_state.num_processes] for i in range(distributed_state.num_processes)]
        with torch.no_grad(), accelerator.autocast(): # Autocast here for multi-GPU
            with distributed_state.split_between_processes(per_process_prompts) as prompt_dict_lists_process:
                for prompt_dict in prompt_dict_lists_process[0]: # Each process gets its list
                     sample_image_inference(accelerator, args, flux_unwrapped, text_encoders_unwrapped, ae, save_dir, prompt_dict, epoch, steps, sample_prompts_te_outputs, prompt_replacement, controlnet_unwrapped)


    torch.set_rng_state(rng_state_cpu)
    if rng_state_gpu is not None: torch.cuda.set_rng_state_all(rng_state_gpu)
    clean_memory_on_device(accelerator.device)

def sample_image_inference(
    accelerator: Accelerator, args: argparse.Namespace, flux: flux_models.Flux,
    text_encoders: Optional[List[Optional[torch.nn.Module]]], ae: flux_models.AutoEncoder,
    save_dir, prompt_dict, epoch, steps, sample_prompts_te_outputs, prompt_replacement, controlnet
):
    negative_prompt = prompt_dict.get("negative_prompt")
    sample_steps_val = prompt_dict.get("sample_steps", 20) # Renamed to avoid conflict
    width = prompt_dict.get("width", args.resolution[0] if args.resolution else 1024)
    height = prompt_dict.get("height", args.resolution[1] if args.resolution else 1024)
    cfg_scale = prompt_dict.get("guidance_scale", 1.0)
    emb_guidance_scale = prompt_dict.get("scale", args.guidance_scale)
    seed = prompt_dict.get("seed")
    controlnet_image_path = prompt_dict.get("controlnet_image")
    prompt_str: str = prompt_dict.get("prompt", "") # Renamed

    if prompt_replacement:
        prompt_str = prompt_str.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt: negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None: torch.manual_seed(seed); 
    if torch.cuda.is_available() and seed is not None: torch.cuda.manual_seed_all(seed) # Seed all GPUs if present

    if negative_prompt is None: negative_prompt = ""
    height = max(64, height - height % 16); width = max(64, width - width % 16)
    logger.info(f"prompt: {prompt_str}" + (f", negative_prompt: {negative_prompt}" if cfg_scale != 1.0 else ""))

    tokenize_strategy: strategy_flux.FluxTokenizeStrategy = strategy_base.TokenizeStrategy.get_strategy() # type: ignore
    encoding_strategy: strategy_flux.FluxTextEncodingStrategy = strategy_base.TextEncodingStrategy.get_strategy() # type: ignore

    def encode_prompt_for_sampling_local(prpt_str_local): # Renamed
        if sample_prompts_te_outputs and prpt_str_local in sample_prompts_te_outputs:
            logger.info(f"Using cached TE outputs for: {prpt_str_local}")
            return sample_prompts_te_outputs[prpt_str_local]
        if text_encoders is None or all(te is None for te in text_encoders):
             raise ValueError("Text encoders not available for sample prompt encoding.")

        logger.info(f"Encoding TE for sampling: {prpt_str_local}")
        tokens_and_masks_dict = tokenize_strategy.tokenize(prpt_str_local)
        active_tes_on_device, original_te_devices = [], [] # Manage TE devices for sampling
        for te_model in text_encoders:
            if te_model is not None:
                original_te_devices.append(te_model.device)
                active_tes_on_device.append(te_model.to(accelerator.device))
            else: active_tes_on_device.append(None); original_te_devices.append(None)
        
        encoded_outputs = encoding_strategy.encode_tokens(tokenize_strategy, active_tes_on_device, tokens_and_masks_dict, args.apply_t5_attn_mask)
        for i, te_model_restore in enumerate(active_tes_on_device): # Restore original devices
            if te_model_restore is not None and original_te_devices[i] is not None: te_model_restore.to(original_te_devices[i])
        return encoded_outputs

    l_pooled, t5_out, txt_ids, t5_attn_mask_from_conds = encode_prompt_for_sampling_local(prompt_str)
    neg_cond = None
    if cfg_scale != 1.0:
        neg_l_pooled, neg_t5_out, _, neg_t5_attn_mask_from_conds_neg = encode_prompt_for_sampling_local(negative_prompt)
        actual_neg_t5_attn_mask = neg_t5_attn_mask_from_conds_neg if args.apply_t5_attn_mask and neg_t5_attn_mask_from_conds_neg is not None else None
        neg_cond = (cfg_scale, neg_l_pooled, neg_t5_out, actual_neg_t5_attn_mask)

    weight_dtype = flux.dtype # Use model's dtype
    packed_latent_height, packed_latent_width = height // 16, width // 16
    
    # Determine noise_shape based on actual model's in_channels
    if hasattr(flux, 'params_dto') and flux.params_dto is not None:
        noise_shape = (1, packed_latent_height * packed_latent_width, flux.params_dto.in_channels)
    else: # Fallback if params_dto is not available (e.g. LoRA only)
        logger.warning("flux.params_dto not available for sampling, assuming in_channels=64 for noise shape.")
        noise_shape = (1, packed_latent_height * packed_latent_width, 64)


    noise_tensor = torch.randn(noise_shape, device=accelerator.device, dtype=weight_dtype, # Renamed
                        generator=torch.Generator(device=accelerator.device).manual_seed(seed) if seed is not None else None)
    
    timesteps_schedule = get_schedule(sample_steps_val, noise_tensor.shape[1], shift=True)
    img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width).to(accelerator.device, dtype=torch.long) # IDs are usually long
    
    actual_t5_attn_mask = t5_attn_mask_from_conds if args.apply_t5_attn_mask and t5_attn_mask_from_conds is not None else None
    controlnet_img_tensor = None
    if controlnet_image_path and controlnet:
        controlnet_image = Image.open(controlnet_image_path).convert("RGB").resize((width, height), Image.LANCZOS)
        controlnet_img_tensor = (torch.from_numpy(np.array(controlnet_image) / 127.5) - 1.0).permute(2,0,1).unsqueeze(0).to(weight_dtype).to(accelerator.device)

    # Ensure main models are on accelerator.device for inference
    flux.to(accelerator.device)
    if controlnet: controlnet.to(accelerator.device)
    # AE will be moved during its use

    # Use accelerator.autocast() for the inference part only if mixed precision is active in accelerator
    with (accelerator.autocast() if accelerator.mixed_precision != "no" else contextlib.nullcontext()), torch.no_grad():
        x_denoised = denoise( # Renamed
            flux, noise_tensor, img_ids, t5_out, txt_ids, l_pooled,
            timesteps=timesteps_schedule, guidance=emb_guidance_scale,
            t5_attn_mask=actual_t5_attn_mask, controlnet=controlnet, controlnet_img=controlnet_img_tensor, neg_cond=neg_cond,
        )

    # Unpack latents (assuming x_denoised is in the "packed flat" format)
    # The vae_unflatten utility expects original shape (B, C_vae, H_vae, W_vae)
    # For FLUX, C_vae is typically 4 (before packing to 16 for AE or 64 for DiT)
    # The flux_utils.unpack_latents is for the DiT's internal representation.
    # After denoise, x_denoised is (B, L_flat, C_flat=params.in_channels). We need to unflatten it to (B, C_vae=4, H_vae, W_vae)
    # This requires knowing the original VAE latent channel count (usually 4 for FLUX's AE).
    # Let's assume params.in_channels was 64 (16*2*2) and original AE output channels were 16 (4*2*2) for vae_flatten.
    # The vae_unflatten needs the *original* VAE channel count, not the DiT's input channels.
    # The ae.encoder outputs (B, 2*z_channels, H/f, W/f), then reg => (B, z_channels, H/f, W/f)
    # For FLUX AE, z_channels=16. So vae_unflatten's `c` should be 16.
    # The output of denoise is (B, L_flat, DiT_in_channels). We need to reshape it to (B, DiT_in_channels/ (dh*dw), H_flat*dh, W_flat*dw)
    # then map DiT_in_channels to VAE_z_channels if they are different.
    # This part is tricky. The `flux_utils.unpack_latents` is likely the correct one if `denoise` outputs what `pack_latents` takes.
    
    # If denoise output matches the 'packed' format that flux_utils.unpack_latents expects:
    x_unpacked = flux_utils.unpack_latents(x_denoised, packed_latent_height, packed_latent_width) # (B, 16, H/16, W/16)

    clean_memory_on_device(accelerator.device)
    org_ae_device = ae.device; ae.to(accelerator.device)
    with (accelerator.autocast() if accelerator.mixed_precision != "no" else contextlib.nullcontext()), torch.no_grad():
        x_decoded = ae.decode(x_unpacked) # Renamed
    ae.to(org_ae_device); clean_memory_on_device(accelerator.device)

    x_decoded = x_decoded.clamp(-1, 1).permute(0, 2, 3, 1)
    image = Image.fromarray((127.5 * (x_decoded + 1.0)).float().cpu().numpy().astype(np.uint8)[0])

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    img_filename = f"{args.output_name or ''}{'_' if args.output_name else ''}{num_suffix}_{prompt_dict['enum']:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(save_dir, img_filename))

    if "wandb" in [tracker.name for tracker in accelerator.trackers]:
        import wandb; accelerator.get_tracker("wandb").log({f"sample_{prompt_dict['enum']}": wandb.Image(image, caption=prompt_str)}, commit=False)

# ... (rest of the sampling utilities: get_lin_function, get_schedule, denoise) ...
# Ensure denoise uses the correct model input format as expected by Flux.forward
def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1); b = y1 - m * x1
    return lambda x: m * x + b

def get_schedule(num_steps: int, image_seq_len: int, base_shift: float = 0.5, max_shift: float = 1.15, shift: bool = True) -> list[float]:
    timesteps = torch.linspace(1, 1e-9, num_steps + 1) # Avoid 0 for 1/t
    if shift:
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, timesteps, 1.0)
    return timesteps.tolist()

def denoise(
    model: flux_models.Flux, img: torch.Tensor, img_ids: torch.Tensor, txt: Optional[torch.Tensor],
    txt_ids: Optional[torch.Tensor], vec: Optional[torch.Tensor], timesteps: list[float],
    guidance: float = 4.0, t5_attn_mask: Optional[torch.Tensor] = None,
    controlnet: Optional[flux_models.ControlNetFlux] = None, controlnet_img: Optional[torch.Tensor] = None,
    neg_cond: Optional[Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]] = None,
):
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    do_cfg = neg_cond is not None
    img_cond = img # Store conditional input for CFG
    
    for t_curr, t_prev in zip(tqdm(timesteps[:-1], disable=True), timesteps[1:]): # Disable tqdm for non-interactive
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        if hasattr(model, 'prepare_block_swap_before_forward'): model.prepare_block_swap_before_forward()
        if controlnet and hasattr(controlnet, 'prepare_block_swap_before_forward'): controlnet.prepare_block_swap_before_forward()

        block_samples, block_single_samples = None, None
        if controlnet and controlnet_img is not None:
            block_samples, block_single_samples = controlnet(
                img=img, img_ids=img_ids, controlnet_cond=controlnet_img, txt=txt, txt_ids=txt_ids, y=vec,
                timesteps=t_vec / 1000.0, guidance=guidance_vec, txt_attention_mask=t5_attn_mask,
            )

        if not do_cfg:
            pred = model(img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids, y=vec,
                         block_controlnet_hidden_states=block_samples, block_controlnet_single_hidden_states=block_single_samples,
                         timesteps=t_vec / 1000.0, guidance=guidance_vec, txt_attention_mask=t5_attn_mask)
            img = img + (t_prev - t_curr) * pred
        else:
            cfg_scale_val, neg_l_pooled, neg_t5_out, actual_neg_t5_attn_mask = neg_cond
            
            # Unconditional inputs
            img_uncond = img # Noise is same for uncond
            txt_uncond = neg_t5_out
            vec_uncond = neg_l_pooled
            t5_attn_mask_uncond = actual_neg_t5_attn_mask
            # txt_ids for uncond should be derived from neg_t5_out's original tokens if available,
            # or use the same txt_ids if assuming same sequence length (might not be robust)
            # For simplicity, if txt_ids are T5 original tokens, they'd differ for neg prompt.
            # Assuming txt_ids here are positional, so they can be reused if lengths match.
            # If Flux uses txt_ids as T5 tokens for PE, then this needs care.
            # The FLUX model's forward uses txt_ids for PE construction (txt_pos_ids_3d = ... txt_ids[...,0] = seq_pos).
            # So, if neg_t5_out has different length, txt_ids needs to be for neg_t5_out.
            # Let's assume for sampling, the prompt TE outputs provide their own txt_ids.

            # Conditional inputs (already set)
            # img_cond, txt_cond (txt), vec_cond (vec), t5_attn_mask_cond (t5_attn_mask)

            # Batched inputs for model call
            batch_img = torch.cat([img_uncond, img_cond], dim=0)
            batch_img_ids = torch.cat([img_ids, img_ids], dim=0) # img_ids are positional, same for u and c
            
            # Handle text features and IDs based on presence
            batch_txt_list, batch_txt_ids_list, batch_t5_attn_mask_list = [], [], []
            for _txt, _txt_ids, _mask in [(txt_uncond, txt_ids, t5_attn_mask_uncond), (txt, txt_ids, t5_attn_mask)]: # Use original txt_ids for cond
                batch_txt_list.append(_txt if _txt is not None else torch.zeros_like(txt if txt is not None else t5_out)) # Placeholder if None
                batch_txt_ids_list.append(_txt_ids if _txt_ids is not None else torch.zeros_like(txt_ids)) # Placeholder
                batch_t5_attn_mask_list.append(_mask) # Can be None

            batch_txt = torch.cat(batch_txt_list, dim=0)
            batch_txt_ids_for_model = torch.cat(batch_txt_ids_list, dim=0) # These are T5 original tokens for Flux.forward
            batch_t5_attn_mask = torch.cat([m for m in batch_t5_attn_mask_list if m is not None], dim=0) if all(m is not None for m in batch_t5_attn_mask_list) else (batch_t5_attn_mask_list[1] if batch_t5_attn_mask_list[1] is not None else None) # Simplistic merge for mask

            batch_vec = torch.cat([
                vec_uncond if vec_uncond is not None else torch.zeros_like(vec), 
                vec if vec is not None else torch.zeros_like(vec_uncond)
            ], dim=0) if vec is not None or vec_uncond is not None else None
            
            batch_guidance_vec = torch.cat([guidance_vec, guidance_vec], dim=0)
            batch_t_vec = torch.cat([t_vec, t_vec], dim=0) / 1000.0

            # ControlNet features should also be batched if used
            batch_block_samples = torch.cat([block_samples, block_samples], dim=0) if block_samples is not None else None
            batch_block_single_samples = torch.cat([block_single_samples, block_single_samples], dim=0) if block_single_samples is not None else None
            
            # Model prediction for both uncond and cond
            full_pred = model(img=batch_img, img_ids=batch_img_ids, txt=batch_txt, txt_ids=batch_txt_ids_for_model, y=batch_vec,
                              block_controlnet_hidden_states=batch_block_samples, block_controlnet_single_hidden_states=batch_block_single_samples,
                              timesteps=batch_t_vec, guidance=batch_guidance_vec, txt_attention_mask=batch_t5_attn_mask)
            
            pred_uncond, pred_cond = torch.chunk(full_pred, 2, dim=0)
            final_pred = pred_uncond + cfg_scale_val * (pred_cond - pred_uncond)
            img = img + (t_prev - t_curr) * final_pred # Update with the guided prediction

    if hasattr(model, 'prepare_block_swap_before_forward'): model.prepare_block_swap_before_forward()
    return img

# endregion

# region train (existing FLUX/SD3 style, used if SOT is false)
def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    if not hasattr(noise_scheduler, 'sigmas'): prepare_scheduler_for_custom_training(noise_scheduler, device) # Ensure sigmas exist
    sigmas_all = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    # Ensure timesteps are tensor for indexing
    timesteps_tensor = timesteps if isinstance(timesteps, torch.Tensor) else torch.tensor(timesteps, device=device)
    # Handle cases where timesteps might be float from certain sampling, cast to long for indexing
    step_indices = [(schedule_timesteps == t.long()).nonzero(as_tuple=True)[0][0].item() for t in timesteps_tensor]
    sigma = sigmas_all[step_indices].flatten()
    while len(sigma.shape) < n_dim: sigma = sigma.unsqueeze(-1)
    return sigma


def compute_density_for_timestep_sampling(weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None):
    if weighting_scheme == "logit_normal": u = torch.sigmoid(torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu"))
    elif weighting_scheme == "mode": u = torch.rand(size=(batch_size,), device="cpu"); u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else: u = torch.rand(size=(batch_size,), device="cpu")
    return u

def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    if sigmas is None: return torch.tensor(1.0) # Should not happen if sigmas are required
    if weighting_scheme == "sigma_sqrt": weighting = (sigmas.float()**-2.0).float() # Ensure float for reciprocal
    elif weighting_scheme == "cosmap": bot = 1 - 2 * sigmas.float() + 2 * sigmas.float()**2; weighting = 2 / (math.pi * bot)
    else: weighting = torch.ones_like(sigmas.float())
    return weighting

def get_noisy_model_input_and_timesteps(
    args, noise_scheduler, latents: torch.Tensor, noise: torch.Tensor, device, dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz = latents.shape[0]
    num_train_timesteps_sched = noise_scheduler.config.num_train_timesteps

    if args.timestep_sampling == "uniform" or args.timestep_sampling == "sigmoid":
        sigmas_val = torch.sigmoid(args.sigmoid_scale * torch.randn((bsz,), device=device)) if args.timestep_sampling == "sigmoid" else torch.rand((bsz,), device=device)
        timesteps_val = sigmas_val * num_train_timesteps_sched # Map 0-1 sigmas to 0-1000 timesteps
    elif args.timestep_sampling == "shift":
        shift_val = args.discrete_flow_shift; sigmas_val = torch.randn(bsz, device=device) * args.sigmoid_scale
        sigmas_val = sigmas_val.sigmoid(); sigmas_val = (sigmas_val * shift_val) / (1 + (shift_val - 1) * sigmas_val)
        timesteps_val = sigmas_val * num_train_timesteps_sched
    elif args.timestep_sampling == "flux_shift":
        sigmas_val = torch.randn(bsz, device=device) * args.sigmoid_scale; sigmas_val = sigmas_val.sigmoid()
        # image_seq_len for flux_shift (packed latents H/2 * W/2)
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
        mu_val = get_lin_function(y1=0.5, y2=1.15)(image_seq_len)
        sigmas_val = time_shift(mu_val, sigmas_val, 1.0)
        timesteps_val = sigmas_val * num_train_timesteps_sched # These are float timesteps representing sigma levels
    elif args.timestep_sampling in ["logit_normal", "mode", "cosmap"]: # SD3-style discrete timesteps
        u_val = compute_density_for_timestep_sampling(args.timestep_sampling, bsz, args.logit_mean, args.logit_std, args.mode_scale)
        indices = (u_val * (num_train_timesteps_sched -1) ).long() # Ensure indices are within bounds
        timesteps_val = noise_scheduler.timesteps[indices].to(device=device) # Discrete timesteps
        sigmas_val = get_sigmas(noise_scheduler, timesteps_val, device, n_dim=latents.ndim, dtype=dtype)
    else: # Default to uniform sigma sampling if unknown
        logger.warning(f"Unknown timestep_sampling: {args.timestep_sampling}, defaulting to uniform sigma.")
        sigmas_val = torch.rand((bsz,), device=device)
        timesteps_val = sigmas_val * num_train_timesteps_sched


    sigmas_view = sigmas_val.view(-1, *([1]*(latents.ndim-1))) # Ensure sigmas_view matches latents ndim
    
    # Noise application based on (1-sigma)x0 + sigma*noise (continuous time formulation)
    # where sigma is fractional progress, not std deviation from scheduler.
    # This matches common flow matching / consistency model papers.
    if args.ip_noise_gamma and args.ip_noise_gamma > 0: # Ensure > 0
        xi = torch.randn_like(latents, device=latents.device, dtype=dtype)
        ip_noise_gamma_val = torch.rand(1, device=latents.device, dtype=dtype) * args.ip_noise_gamma if args.ip_noise_gamma_random_strength else args.ip_noise_gamma
        noisy_model_input = (1.0 - sigmas_view) * latents + sigmas_view * (noise + ip_noise_gamma_val * xi)
    else: noisy_model_input = (1.0 - sigmas_view) * latents + sigmas_view * noise
    
    # timesteps_val are already computed based on sampling scheme.
    # For SD3-style, they are discrete. For others, they are float representations of sigma levels.
    # The model's `timesteps / 1000.0` expects 0-1 range input.
    # If timesteps_val are already 0-1 (like sigmas_val essentially), then they are fine.
    # If they are 0-1000 (like for SD3 style), they'll be scaled by model.
    # For consistency, ensure timesteps_val passed out are what model expects for scaling.
    # The Flux model does `timesteps / 1000.0`. So, if `timesteps_val` are already 0-1 (like sigmas),
    # then the model will get very small values.
    # `get_noisy_model_input_and_timesteps` should return timesteps in the 0-1000 range for the model's internal division.
    # So, if timesteps_val were derived from sigmas (0-1), scale them back to 0-1000.
    if args.timestep_sampling not in ["logit_normal", "mode", "cosmap"]: # i.e. sigma-based sampling
        timesteps_for_model = timesteps_val # these are already effectively sigmas * 1000
    else: # SD3-style, timesteps_val are already discrete 0-999
        timesteps_for_model = timesteps_val

    return noisy_model_input.to(dtype), timesteps_for_model.to(dtype), sigmas_val.to(dtype)


def apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas):
    weighting = None
    sigmas_view = sigmas.view(-1, *([1]*(model_pred.ndim-1)))
    if args.model_prediction_type == "raw": pass # Model predicts (noise - latents) directly
    elif args.model_prediction_type == "additive": # Model predicts (noise - latents), then add x_t to get noise
        model_pred = model_pred + noisy_model_input
    elif args.model_prediction_type == "sigma_scaled": # Model predicts (noise - latents), scale by sigma, add x_t
        # This is essentially v-prediction like if target is noise, but model output is x0-like.
        # model_pred = model_pred * (-sigmas_view) + noisy_model_input # Predicts noise from x0-like
        # If model_pred is (target = noise - latents), then for sigma_scaled loss weighting:
        # The loss is weighted by 1/sigma^2. This seems to be the SD3 interpretation.
        # The "model_pred" here is what the network outputs.
        # The target is `noise - latents`.
        # If `model_prediction_type` is "sigma_scaled", this function does not modify `model_pred`.
        # It only computes a weighting.
        weighting = compute_loss_weighting_for_sd3(args.weighting_scheme if args.timestep_sampling in ["logit_normal", "mode", "cosmap"] else "sigma_sqrt", sigmas)
        # For sigma_sqrt, weighting is 1/sigmas^2. This is applied to MSE((pred - target)).
        # The original code's `model_pred = model_pred * (-sigmas) + noisy_model_input` was to transform
        # an x0-prediction to a noise-prediction. Here, model_pred is already noise-like (or (noise-target)-like).
        # So, no transformation of model_pred is done here for "sigma_scaled".
    return model_pred, weighting
# endregion (existing FLUX/SD3 style)


def add_flux_train_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--clip_l", type=str, default=None, help="path to clip_l (*.sft or *.safetensors)")
    parser.add_argument("--clip_l_tokenizer_path", type=str, default=None, help="path to clip_l tokenizer (directory or file)")
    parser.add_argument("--t5xxl", type=str, help="path to t5xxl (*.sft or *.safetensors)")
    parser.add_argument("--ae", type=str, help="path to ae (*.sft or *.safetensors)")
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None, help="path to controlnet")
    parser.add_argument("--t5xxl_max_token_length", type=int, default=None, help="max token length for T5-XXL (default: 256 for schnell/chroma, 512 for dev)")
    parser.add_argument("--apply_t5_attn_mask", action="store_true", help="apply attention mask to T5-XXL and FLUX double blocks")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="embedded guidance scale for FLUX.1 dev/Chroma")
    
    # Timestep/Loss args for non-SOT FLUX / SD3 style
    parser.add_argument("--timestep_sampling", choices=["uniform", "sigmoid", "shift", "flux_shift", "logit_normal", "mode", "cosmap"], default="uniform", help="Method to sample timesteps (sigmas for some, discrete for others)")
    parser.add_argument("--sigmoid_scale", type=float, default=1.0, help="Scale for sigmoid/shift timestep sampling")
    parser.add_argument("--model_prediction_type", choices=["raw", "additive", "sigma_scaled"], default="raw", help="Model prediction interpretation (raw means model predicts (noise - x0))")
    parser.add_argument("--discrete_flow_shift", type=float, default=3.0, help="Discrete flow shift for Euler Discrete Scheduler (shift sampling)")
    # Chroma SOT Loss Flag
    parser.add_argument("--use_chroma_sot_loss", action="store_true", help="Use Chroma-specific Stochastic Optimal Transport loss. Overrides some timestep/loss args.")