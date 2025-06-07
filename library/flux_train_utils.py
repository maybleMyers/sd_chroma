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

from library import flux_models, flux_utils, strategy_base, train_util
from library.device_utils import init_ipex, clean_memory_on_device

init_ipex()

from .utils import setup_logging, mem_eff_save_file

setup_logging()
import logging

logger = logging.getLogger(__name__)


# region sample images


def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    flux,
    ae,
    text_encoders,
    sample_prompts_te_outputs,
    prompt_replacement=None,
    controlnet=None,
):
    if steps == 0:
        if not args.sample_at_first:
            return
    else:
        if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
            return
        if args.sample_every_n_epochs is not None:
            # sample_every_n_steps は無視する
            if epoch is None or epoch % args.sample_every_n_epochs != 0:
                return
        else:
            if steps % args.sample_every_n_steps != 0 or epoch is not None:  # steps is not divisible or end of epoch
                return

    logger.info("")
    logger.info(f"generating sample images at step / サンプル画像生成 ステップ: {steps}")
    if not os.path.isfile(args.sample_prompts) and sample_prompts_te_outputs is None:
        logger.error(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
        return

    distributed_state = PartialState()  # for multi gpu distributed inference. this is a singleton, so it's safe to use it here

    # unwrap unet and text_encoder(s)
    flux_unwrapped = accelerator.unwrap_model(flux) # Renamed to avoid conflict
    text_encoders_unwrapped = None
    if text_encoders is not None:
        # Handle list of TEs, some of which might be None (like CLIP-L for Chroma)
        text_encoders_unwrapped = []
        for te in text_encoders:
            if te is not None:
                text_encoders_unwrapped.append(accelerator.unwrap_model(te))
            else:
                text_encoders_unwrapped.append(None)

    controlnet_unwrapped = None
    if controlnet is not None:
        controlnet_unwrapped = accelerator.unwrap_model(controlnet)


    prompts = train_util.load_prompts(args.sample_prompts)

    save_dir = args.output_dir + "/sample"
    os.makedirs(save_dir, exist_ok=True)

    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    if distributed_state.num_processes <= 1:
        with torch.no_grad(), accelerator.autocast():
            for prompt_dict in prompts:
                sample_image_inference(
                    accelerator,
                    args,
                    flux_unwrapped, # Use unwrapped
                    text_encoders_unwrapped, # Use unwrapped
                    ae, # AE is usually on CPU or handled differently, not wrapped by accelerator directly in training_models list
                    save_dir,
                    prompt_dict,
                    epoch,
                    steps,
                    sample_prompts_te_outputs,
                    prompt_replacement,
                    controlnet_unwrapped, # Use unwrapped
                )
    else:
        per_process_prompts = [] 
        for i in range(distributed_state.num_processes):
            per_process_prompts.append(prompts[i :: distributed_state.num_processes])

        with torch.no_grad():
            with distributed_state.split_between_processes(per_process_prompts) as prompt_dict_lists:
                for prompt_dict in prompt_dict_lists[0]:
                    sample_image_inference(
                        accelerator,
                        args,
                        flux_unwrapped,
                        text_encoders_unwrapped,
                        ae,
                        save_dir,
                        prompt_dict,
                        epoch,
                        steps,
                        sample_prompts_te_outputs,
                        prompt_replacement,
                        controlnet_unwrapped,
                    )

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    clean_memory_on_device(accelerator.device)


def sample_image_inference(
    accelerator: Accelerator,
    args: argparse.Namespace,
    flux: flux_models.Flux,
    text_encoders: Optional[List[Optional[torch.nn.Module]]], # Can contain None for CLIP-L
    ae: flux_models.AutoEncoder,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    sample_prompts_te_outputs,
    prompt_replacement,
    controlnet,
):
    assert isinstance(prompt_dict, dict)
    negative_prompt = prompt_dict.get("negative_prompt")
    sample_steps = prompt_dict.get("sample_steps", 20)
    width = prompt_dict.get("width", args.resolution[0] if args.resolution else 1024) # Use args.resolution as fallback
    height = prompt_dict.get("height", args.resolution[1] if args.resolution else 1024)
    cfg_scale = prompt_dict.get("guidance_scale", 1.0) 
    emb_guidance_scale = prompt_dict.get("scale", args.guidance_scale) # Use args.guidance_scale from command line as fallback
    seed = prompt_dict.get("seed")
    controlnet_image_path = prompt_dict.get("controlnet_image") # Renamed to avoid conflict with loaded image
    prompt: str = prompt_dict.get("prompt", "")

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt is not None:
            negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed(seed)
    else:
        torch.seed(); 
        if torch.cuda.is_available(): torch.cuda.seed()


    if negative_prompt is None: negative_prompt = ""
    height = max(64, height - height % 16) 
    width = max(64, width - width % 16)  
    logger.info(f"prompt: {prompt}")
    if cfg_scale != 1.0: logger.info(f"negative_prompt: {negative_prompt}")
    # ... (logging remains the same)

    tokenize_strategy: strategy_flux.FluxTokenizeStrategy = strategy_base.TokenizeStrategy.get_strategy()
    encoding_strategy: strategy_flux.FluxTextEncodingStrategy = strategy_base.TextEncodingStrategy.get_strategy()

    def encode_prompt_for_sampling(prpt_str):
        if sample_prompts_te_outputs and prpt_str in sample_prompts_te_outputs:
            logger.info(f"Using cached text encoder outputs for prompt: {prpt_str}")
            return sample_prompts_te_outputs[prpt_str]
        
        # text_encoders is [clip_l_or_None, t5xxl]
        # If text_encoders themselves are None (e.g. after caching and freeing), this won't work.
        # This sampling function assumes text_encoders are available if not using sample_prompts_te_outputs.
        if text_encoders is None or all(te is None for te in text_encoders):
             raise ValueError("Text encoders are not available for encoding sample prompts and pre-cached outputs are not found.")

        logger.info(f"Encoding prompt for sampling: {prpt_str}")
        tokens_and_masks_dict = tokenize_strategy.tokenize(prpt_str)
        # Ensure text_encoders are on the correct device for encoding
        active_tes_on_device = []
        if text_encoders[0] is not None : 
            text_encoders[0].to(accelerator.device)
            active_tes_on_device.append(text_encoders[0])
        else:
            active_tes_on_device.append(None)
        
        text_encoders[1].to(accelerator.device) # T5XXL always present
        active_tes_on_device.append(text_encoders[1])
        
        encoded_outputs = encoding_strategy.encode_tokens(
            tokenize_strategy, active_tes_on_device, tokens_and_masks_dict, args.apply_t5_attn_mask
        )
        # Move encoders back to CPU if they were temporarily moved
        if text_encoders[0] is not None and text_encoders[0].device != torch.device("cpu"): text_encoders[0].cpu()
        if text_encoders[1].device != torch.device("cpu"): text_encoders[1].cpu() # T5XXL to CPU
        return encoded_outputs

    l_pooled, t5_out, txt_ids, t5_attn_mask_from_conds = encode_prompt_for_sampling(prompt)
    
    neg_cond = None
    if cfg_scale != 1.0:
        neg_l_pooled, neg_t5_out, _, neg_t5_attn_mask_from_conds = encode_prompt_for_sampling(negative_prompt)
        actual_neg_t5_attn_mask = neg_t5_attn_mask_from_conds if args.apply_t5_attn_mask and neg_t5_attn_mask_from_conds is not None else None
        neg_cond = (cfg_scale, neg_l_pooled, neg_t5_out, actual_neg_t5_attn_mask)

    weight_dtype = ae.dtype 
    packed_latent_height, packed_latent_width = height // 16, width // 16
    noise_shape = (1, packed_latent_height * packed_latent_width, 16 * 2 * 2) # C_in for FLUX is 16 (packed from 4x4x4 for example)
    
    if flux.params_dto.in_channels == 64: # Packed (4*2*2 = 16 channels * 4 = 64)
         noise_shape = (1, packed_latent_height * packed_latent_width, flux.params_dto.in_channels)
    else: # Should not happen with current FLUX models
        raise ValueError(f"Unexpected in_channels for FLUX model: {flux.params_dto.in_channels}")


    noise = torch.randn(noise_shape, device=accelerator.device, dtype=weight_dtype,
                        generator=torch.Generator(device=accelerator.device).manual_seed(seed) if seed is not None else None)
    
    timesteps_schedule = get_schedule(sample_steps, noise.shape[1], shift=True) 
    img_ids = flux_utils.prepare_img_ids(1, packed_latent_height, packed_latent_width).to(accelerator.device, weight_dtype)
    
    actual_t5_attn_mask = t5_attn_mask_from_conds if args.apply_t5_attn_mask and t5_attn_mask_from_conds is not None else None

    controlnet_img_tensor = None
    if controlnet_image_path is not None and controlnet is not None:
        controlnet_image = Image.open(controlnet_image_path).convert("RGB")
        controlnet_image = controlnet_image.resize((width, height), Image.LANCZOS)
        controlnet_img_tensor = torch.from_numpy((np.array(controlnet_image) / 127.5) - 1)
        controlnet_img_tensor = controlnet_img_tensor.permute(2, 0, 1).unsqueeze(0).to(weight_dtype).to(accelerator.device)

    with accelerator.autocast(), torch.no_grad():
        x = denoise(
            flux, noise, img_ids, t5_out, txt_ids, l_pooled, # l_pooled can be None for Chroma
            timesteps=timesteps_schedule, guidance=emb_guidance_scale,
            t5_attn_mask=actual_t5_attn_mask, controlnet=controlnet, controlnet_img=controlnet_img_tensor, neg_cond=neg_cond,
        )

    x = flux_utils.unpack_latents(x, packed_latent_height, packed_latent_width)

    clean_memory_on_device(accelerator.device)
    org_vae_device = ae.device 
    ae.to(accelerator.device) 
    with accelerator.autocast(), torch.no_grad():
        x = ae.decode(x)
    ae.to(org_vae_device)
    clean_memory_on_device(accelerator.device)

    x = x.clamp(-1, 1).permute(0, 2, 3, 1)
    image = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i: int = prompt_dict["enum"]
    img_filename = f"{args.output_name or ''}{'_' if args.output_name else ''}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(save_dir, img_filename))

    if "wandb" in [tracker.name for tracker in accelerator.trackers]:
        import wandb
        accelerator.get_tracker("wandb").log({f"sample_{i}": wandb.Image(image, caption=prompt)}, commit=False)


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int, # This is packed_latent_height * packed_latent_width
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    timesteps = torch.linspace(1, 0, num_steps + 1) 
    if shift:
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)
    return timesteps.tolist()


def denoise(
    model: flux_models.Flux,
    img: torch.Tensor, # packed noisy latents
    img_ids: torch.Tensor,
    txt: Optional[torch.Tensor],  # t5_out
    txt_ids: Optional[torch.Tensor], # t5 input_ids
    vec: Optional[torch.Tensor],  # l_pooled (from CLIP-L, can be None for Chroma)
    timesteps: list[float],
    guidance: float = 4.0, # This is the embedded guidance scale
    t5_attn_mask: Optional[torch.Tensor] = None,
    controlnet: Optional[flux_models.ControlNetFlux] = None,
    controlnet_img: Optional[torch.Tensor] = None,
    neg_cond: Optional[Tuple[float, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]] = None, # cfg_scale, neg_l_pooled, neg_t5_out, neg_t5_attn_mask
):
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    do_cfg = neg_cond is not None

    for t_curr, t_prev in zip(tqdm(timesteps[:-1]), timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device) # Model expects 0-1 range
        
        # Ensure model is on the correct device if block swapping
        if hasattr(model, 'prepare_block_swap_before_forward'): model.prepare_block_swap_before_forward()
        if controlnet and hasattr(controlnet, 'prepare_block_swap_before_forward'): controlnet.prepare_block_swap_before_forward()

        block_samples, block_single_samples = None, None
        if controlnet is not None and controlnet_img is not None:
            # Note: ControlNetFlux forward signature needs to match what's passed
            block_samples, block_single_samples = controlnet(
                img=img, img_ids=img_ids, controlnet_cond=controlnet_img,
                txt=txt, txt_ids=txt_ids, y=vec, timesteps=t_vec / 1000.0, # ControlNet also expects 0-1 range
                guidance=guidance_vec, txt_attention_mask=t5_attn_mask,
            )

        if not do_cfg:
            pred = model(
                img=img, img_ids=img_ids, txt=txt, txt_ids=txt_ids, y=vec,
                block_controlnet_hidden_states=block_samples, block_controlnet_single_hidden_states=block_single_samples,
                timesteps=t_vec / 1000.0, guidance=guidance_vec, txt_attention_mask=t5_attn_mask,
            )
            img = img + (t_prev - t_curr) * pred 
        else:
            cfg_scale_val, neg_l_pooled, neg_t5_out, actual_neg_t5_attn_mask = neg_cond
            
            # Prepare inputs for CFG
            batch_img = torch.cat([img, img], dim=0)
            batch_img_ids = torch.cat([img_ids, img_ids], dim=0)
            batch_txt = torch.cat([neg_t5_out, txt], dim=0) if neg_t5_out is not None and txt is not None else (txt if txt is not None else neg_t5_out) # Handle if one is None
            batch_txt_ids = torch.cat([txt_ids, txt_ids], dim=0) # txt_ids (T5 input_ids) should always be present for T5

            # Handle conditional vec (l_pooled)
            if vec is not None and neg_l_pooled is not None: batch_vec = torch.cat([neg_l_pooled, vec], dim=0)
            elif vec is not None: batch_vec = torch.cat([torch.zeros_like(vec), vec], dim=0) # Assuming zero for neg if not provided but cond is
            elif neg_l_pooled is not None: batch_vec = torch.cat([neg_l_pooled, torch.zeros_like(neg_l_pooled)], dim=0)
            else: batch_vec = None # Both are None
            
            batch_t5_attn_mask = None
            if t5_attn_mask is not None and actual_neg_t5_attn_mask is not None:
                batch_t5_attn_mask = torch.cat([actual_neg_t5_attn_mask, t5_attn_mask], dim=0)
            elif t5_attn_mask is not None: # Only positive mask
                # Need a dummy negative mask of same shape for concatenation
                dummy_neg_mask = torch.ones_like(t5_attn_mask) if t5_attn_mask.dtype == torch.bool else torch.zeros_like(t5_attn_mask)
                batch_t5_attn_mask = torch.cat([dummy_neg_mask, t5_attn_mask], dim=0)

            batch_guidance_vec = torch.cat([guidance_vec, guidance_vec], dim=0) # Same guidance value for both

            batch_block_samples = torch.cat([block_samples, block_samples], dim=0) if block_samples is not None else None
            batch_block_single_samples = torch.cat([block_single_samples, block_single_samples], dim=0) if block_single_samples is not None else None
            
            nc_c_pred = model(
                img=batch_img, img_ids=batch_img_ids, txt=batch_txt, txt_ids=batch_txt_ids, y=batch_vec,
                block_controlnet_hidden_states=batch_block_samples, block_controlnet_single_hidden_states=batch_block_single_samples,
                timesteps=torch.cat([t_vec,t_vec],dim=0) / 1000.0, guidance=batch_guidance_vec, txt_attention_mask=batch_t5_attn_mask,
            )
            neg_pred, cond_pred = torch.chunk(nc_c_pred, 2, dim=0)
            final_pred = neg_pred + (cond_pred - neg_pred) * cfg_scale_val
            img = img + (t_prev - t_curr) * final_pred
            
    if hasattr(model, 'prepare_block_swap_before_forward'): model.prepare_block_swap_before_forward()
    return img

# endregion
# region train
def get_sigmas(noise_scheduler, timesteps, device, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    return sigma

def compute_density_for_timestep_sampling(
    weighting_scheme: str, batch_size: int, logit_mean: float = None, logit_std: float = None, mode_scale: float = None
):
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device="cpu"); u = torch.nn.functional.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(size=(batch_size,), device="cpu"); u = 1 - u - mode_scale * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else: u = torch.rand(size=(batch_size,), device="cpu")
    return u

def compute_loss_weighting_for_sd3(weighting_scheme: str, sigmas=None):
    if weighting_scheme == "sigma_sqrt": weighting = (sigmas**-2.0).float()
    elif weighting_scheme == "cosmap": bot = 1 - 2 * sigmas + 2 * sigmas**2; weighting = 2 / (math.pi * bot)
    else: weighting = torch.ones_like(sigmas)
    return weighting

def get_noisy_model_input_and_timesteps(
    args, noise_scheduler, latents: torch.Tensor, noise: torch.Tensor, device, dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bsz, _, h, w = latents.shape; assert bsz > 0
    num_timesteps = noise_scheduler.config.num_train_timesteps
    if args.timestep_sampling == "uniform" or args.timestep_sampling == "sigmoid":
        sigmas = torch.sigmoid(args.sigmoid_scale * torch.randn((bsz,), device=device)) if args.timestep_sampling == "sigmoid" else torch.rand((bsz,), device=device)
    elif args.timestep_sampling == "shift":
        shift = args.discrete_flow_shift; sigmas = torch.randn(bsz, device=device) * args.sigmoid_scale
        sigmas = sigmas.sigmoid(); sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
    elif args.timestep_sampling == "flux_shift":
        sigmas = torch.randn(bsz, device=device) * args.sigmoid_scale; sigmas = sigmas.sigmoid()
        mu = get_lin_function(y1=0.5, y2=1.15)((h // 2) * (w // 2)) 
        sigmas = time_shift(mu, 1.0, sigmas)
    else: # SD3 weighting schemes
        u = compute_density_for_timestep_sampling(args.weighting_scheme, bsz, args.logit_mean, args.logit_std, args.mode_scale)
        indices = (u * num_timesteps).long(); timesteps_long = noise_scheduler.timesteps[indices].to(device=device)
        sigmas = get_sigmas(noise_scheduler, timesteps_long, device, n_dim=latents.ndim, dtype=dtype)
        # Return early if SD3 specific timestep handling, as timesteps are derived differently
        sigmas_view = sigmas.view(-1, 1, 1, 1)
        noisy_model_input = (1.0 - sigmas_view) * latents + sigmas_view * noise # Simplified, no ip_noise_gamma for now
        return noisy_model_input.to(dtype), timesteps_long.to(dtype), sigmas # Return the long timesteps

    timesteps = sigmas * num_timesteps # For non-SD3 schemes
    sigmas_view = sigmas.view(-1, 1, 1, 1)
    if args.ip_noise_gamma:
        xi = torch.randn_like(latents, device=latents.device, dtype=dtype)
        ip_noise_gamma_val = torch.rand(1, device=latents.device, dtype=dtype) * args.ip_noise_gamma if args.ip_noise_gamma_random_strength else args.ip_noise_gamma
        noisy_model_input = (1.0 - sigmas_view) * latents + sigmas_view * (noise + ip_noise_gamma_val * xi)
    else: noisy_model_input = (1.0 - sigmas_view) * latents + sigmas_view * noise
    return noisy_model_input.to(dtype), timesteps.to(dtype), sigmas

def apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas):
    weighting = None
    if args.model_prediction_type == "raw": pass
    elif args.model_prediction_type == "additive": model_pred = model_pred + noisy_model_input
    elif args.model_prediction_type == "sigma_scaled":
        model_pred = model_pred * (-sigmas) + noisy_model_input
        weighting = compute_loss_weighting_for_sd3(args.weighting_scheme, sigmas)
    return model_pred, weighting

def save_models(ckpt_path: str, flux: flux_models.Flux, sai_metadata: Optional[dict], save_dtype: Optional[torch.dtype] = None, use_mem_eff_save: bool = False):
    state_dict = {}
    def update_sd(prefix, sd):
        for k, v in sd.items():
            key = prefix + k
            if save_dtype is not None and v.dtype != save_dtype: v = v.detach().clone().to("cpu").to(save_dtype)
            state_dict[key] = v
    update_sd("", flux.state_dict())
    save_func = mem_eff_save_file if use_mem_eff_save else save_file
    save_func(state_dict, ckpt_path, metadata=sai_metadata)

def save_flux_model_on_train_end(args: argparse.Namespace, save_dtype: torch.dtype, epoch: int, global_step: int, flux: flux_models.Flux):
    def sd_saver(ckpt_file, epoch_no, step): sai_metadata = train_util.get_sai_model_spec(None, args, False, False, False, is_stable_diffusion_ckpt=True, flux=flux_utils.MODEL_TYPE_FLUX_DEV); save_models(ckpt_file, flux, sai_metadata, save_dtype, args.mem_eff_save) # Use flux_utils.MODEL_TYPE_FLUX_DEV for now
    train_util.save_sd_model_on_train_end_common(args, True, True, epoch, global_step, sd_saver, None)

def save_flux_model_on_epoch_end_or_stepwise(args: argparse.Namespace, on_epoch_end: bool, accelerator, save_dtype: torch.dtype, epoch: int, num_train_epochs: int, global_step: int, flux: flux_models.Flux):
    def sd_saver(ckpt_file, epoch_no, step): sai_metadata = train_util.get_sai_model_spec(None, args, False, False, False, is_stable_diffusion_ckpt=True, flux=flux_utils.MODEL_TYPE_FLUX_DEV); save_models(ckpt_file, flux, sai_metadata, save_dtype, args.mem_eff_save)
    train_util.save_sd_model_on_epoch_end_or_stepwise_common(args, on_epoch_end, accelerator, True, True, epoch, num_train_epochs, global_step, sd_saver, None)
# endregion

def add_flux_train_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--clip_l", type=str, default=None, help="path to clip_l (*.sft or *.safetensors)")
    parser.add_argument("--clip_l_tokenizer_path", type=str, default=None, help="path to clip_l tokenizer (directory or file)")
    parser.add_argument("--t5xxl", type=str, help="path to t5xxl (*.sft or *.safetensors)")
    parser.add_argument("--ae", type=str, help="path to ae (*.sft or *.safetensors)")
    parser.add_argument("--controlnet_model_name_or_path", type=str, default=None, help="path to controlnet")
    parser.add_argument("--t5xxl_max_token_length", type=int, default=None, help="max token length for T5-XXL (default: 256 for schnell/chroma, 512 for dev)")
    parser.add_argument("--apply_t5_attn_mask", action="store_true", help="apply attention mask to T5-XXL and FLUX double blocks")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="embedded guidance scale for FLUX.1 dev/Chroma")
    parser.add_argument("--timestep_sampling", choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift", "logit_normal", "mode", "cosmap"], default="sigma", help="Method to sample timesteps") # Added SD3 options
    parser.add_argument("--sigmoid_scale", type=float, default=1.0, help="Scale for sigmoid/shift timestep sampling")
    parser.add_argument("--model_prediction_type", choices=["raw", "additive", "sigma_scaled"], default="sigma_scaled", help="Model prediction interpretation")
    parser.add_argument("--discrete_flow_shift", type=float, default=3.0, help="Discrete flow shift for Euler Discrete Scheduler")