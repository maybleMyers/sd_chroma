import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import copy
import math
import random
from typing import Any, Optional, Union

import torch
#torch.autograd.set_detect_anomaly(True)
from accelerate import Accelerator
from diffusers import DDPMScheduler # <--- ADD THIS IMPORT

from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

import train_network
from library import (
    flux_models,
    flux_train_utils,
    flux_utils,
    sd3_train_utils, # This import seems unused now, sd3_train_utils might have been for previous SD3 attempts
    strategy_base,
    strategy_flux,
    train_util,
    custom_train_functions, # Import for fix_noise_scheduler_betas_for_zero_terminal_snr
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class FluxNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.model_type_str: Optional[str] = None # Store analyzed model type
        self.is_swapping_blocks: bool = False
        self.train_clip_l: bool = False  # Initialize here
        self.train_t5xxl: bool = False # Initialize here

    def all_reduce_network(self, accelerator: Accelerator, network: torch.nn.Module):
        """
        Manually reduces gradients for the network parameters across DDP processes.
        This is called when accelerator.sync_gradients is True.
        Often, accelerator.backward() handles this implicitly, but an explicit
        call might be needed for specific network types or DDP configurations.
        """
        if accelerator.state.num_processes > 1:
            # Check if the network has parameters that require gradients.
            # The `network` object here is the LoRA network.
            # We are interested in the gradients of its trainable parameters.
            trainable_params = [p for p in network.parameters() if p.requires_grad and p.grad is not None]
            
            if not trainable_params:
                logger.debug("all_reduce_network: No trainable parameters with gradients found in the network to reduce.")
                return

            logger.debug(f"all_reduce_network: Attempting to reduce gradients for {len(trainable_params)} parameter(s)/tensor(s) in the network.")
            
            # It's generally safer to reduce gradients one by one if they are not part of a single contiguous block
            # that `accelerator.backward` would handle perfectly for a simple model.
            # For LoRA, parameters are typically distinct.
            for param in trainable_params:
                try:
                    # The reduction operation should be 'mean' for gradients.
                    param.grad = accelerator.reduce(param.grad, reduction="mean")
                except Exception as e:
                    logger.error(f"Error during gradient reduction for a parameter: {e}")
                    # Optionally, re-raise or handle as appropriate
            logger.info(f"Manually reduced network gradients across DDP processes (num_processes: {accelerator.state.num_processes}).")
        else:
            logger.debug("all_reduce_network: Single process, no DDP reduction needed for the network.")

    def on_step_start(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=True):
        """
        Called at the start of each training step. Implements basic logging and ensures compatibility
        with the base NetworkTrainer's train method.
        """
        logger.info(f"Starting training step. Device: {unet.device}, dtype: {weight_dtype}")
        # Add any Flux-specific step initialization here if needed (e.g., resetting block swaps, updating schedules)
        if self.is_swapping_blocks:
            unet.prepare_block_swap_before_forward()
        # No other specific actions needed for now; can extend later if required
        pass

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        if args.timestep_sampling in ["logit_normal", "mode", "cosmap"]: 
            logger.info(f"Using DDPMScheduler for SD3-style timestep sampling: {args.timestep_sampling}")
            noise_scheduler = DDPMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", 
                num_train_timesteps=1000, clip_sample=False,
            )
            # train_util.prepare_scheduler_for_custom_training(noise_scheduler, device) # Not strictly needed here
            if args.zero_terminal_snr: 
                custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)
            return noise_scheduler
        else: 
            logger.info(f"FLUX timestep sampling ({args.timestep_sampling}) does not strictly require a Diffusers scheduler object here. Returning a basic DDPMScheduler for compatibility.")
            return DDPMScheduler(num_train_timesteps=1000) 

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizers, text_encoders, unet):
        # tokenizers will be [clip_l_tokenizer_or_None, t5_tokenizer]
        # text_encoders will be [clip_l_model_or_None, t5_model]
        # unet is the FLUX model
        # vae is the FLUX AE
        logger.info("FluxNetworkTrainer: Calling flux_train_utils.sample_images")
        flux_train_utils.sample_images(
            accelerator,
            args,
            epoch,
            global_step,
            unet, # FLUX model
            vae,  # FLUX AE
            text_encoders, # List of [CLIP-L or None, T5XXL]
            self.sample_prompts_te_outputs, # Cached outputs for sample prompts
            # prompt_replacement and controlnet are not used in this specific LoRA trainer context for now
        )

    def encode_images_to_latents(self, args, vae: flux_models.AutoEncoder, images: torch.FloatTensor) -> torch.FloatTensor:
        # FLUX AutoEncoder's encode method directly returns the latents (already sampled if it's a VAE part)
        # and applies internal scaling/shifting.
        logger.debug("FluxNetworkTrainer: using FLUX AE direct encode.")
        return vae.encode(images)

    def shift_scale_latents(self, args, latents: torch.FloatTensor) -> torch.FloatTensor:
        # The output of flux_models.AutoEncoder.encode() is already scaled and shifted
        # according to its internal scale_factor and shift_factor.
        # So, no further scaling by the old SD vae_scale_factor (0.18215) is needed.
        logger.debug("FluxNetworkTrainer: shift_scale_latents is a no-op as FLUX AE handles scaling internally.")
        return latents # Return latents as is

    def post_process_loss(self, loss: torch.FloatTensor, args: argparse.Namespace, timesteps: torch.Tensor, noise_scheduler) -> torch.FloatTensor:
        # FLUX models might not use the same SNR weighting or v-prediction-like loss adjustments as SD1.5/2.x.
        # The primary loss is typically MSE between (model_pred + latents) and noise, or similar,
        # after model_pred is adjusted by apply_model_prediction_type.
        # If args.min_snr_gamma or other SD-specific loss args are used, they might be misapplied here.
        # For now, let's assume these are not used or handled correctly by their conditions.
        # If specific FLUX/SD3 loss weighting is needed, it would go here.
        
        # The `weighting` from `apply_model_prediction_type` (for sigma_scaled) is already applied
        # to the loss *before* this function in the base NetworkTrainer.process_batch if that weighting is returned.
        # This function in the base trainer then applies min_snr_gamma etc.
        
        # For FLUX, if args.model_prediction_type is "sigma_scaled", a weighting is already computed.
        # Other SD-specific weightings like min_snr_gamma are likely not applicable or need re-derivation for flow matching.

        if args.min_snr_gamma:
            logger.warning("min_snr_gamma is specified but may not be directly applicable to FLUX models without careful scheduler setup.")
            # If a compatible scheduler was set up in get_noise_scheduler, this might work, but needs verification.
            # For now, let parent handle it, but be aware.
            return super().post_process_loss(loss, args, timesteps, noise_scheduler)
        
        # Other SD-specific losses:
        if args.scale_v_pred_loss_like_noise_pred or args.v_pred_like_loss or args.debiased_estimation_loss:
            logger.warning("SD-specific loss processing (scale_v_pred, v_pred_like, debiased_estimation) may not apply to FLUX.")
            return super().post_process_loss(loss, args, timesteps, noise_scheduler) # Let parent handle if flags are on

        logger.debug("FluxNetworkTrainer: post_process_loss returning loss as is (or after parent's if flags were set).")
        return loss # Or return super().post_process_loss(loss, args, timesteps, noise_scheduler) if we want to keep parent's logic

    def get_sai_model_spec(self, args: argparse.Namespace) -> dict:
        # FLUX model has its own SAI model spec
        # Use flux_utils.MODEL_TYPE_FLUX_DEV, flux_utils.MODEL_TYPE_FLUX_SCHNELL, or flux_utils.MODEL_TYPE_CHROMA
        # self.model_type_str should be set
        if self.model_type_str is None: # Fallback if not set, though it should be
             logger.warning("model_type_str not set in get_sai_model_spec, attempting to analyze checkpoint again.")
             if args.pretrained_model_name_or_path:
                self.model_type_str, _, _, _ = flux_utils.analyze_checkpoint_state(args.pretrained_model_name_or_path)
             else: # Cannot determine type
                 logger.error("Cannot determine FLUX model type for SAI metadata as pretrained_model_name_or_path is missing.")
                 return train_util.get_sai_model_spec(None, args, is_sdxl=False, is_stable_diffusion_ckpt=True, flux="unknown_flux_type")


        # Map internal model_type_str to the SAI flux argument string if necessary,
        # or directly use a general one like flux_utils.MODEL_TYPE_FLUX_DEV for all trained LoRAs for now.
        # For simplicity, let's use MODEL_TYPE_FLUX_DEV as a general tag for LoRAs trained on any FLUX variant.
        # More specific tagging could be done if needed.
        logger.info(f"FluxNetworkTrainer: Using flux='{flux_utils.MODEL_TYPE_FLUX_DEV}' for SAI metadata (original base: {self.model_type_str}).")
        return train_util.get_sai_model_spec(
            None, args, is_sdxl=False, is_stable_diffusion_ckpt=True, flux=flux_utils.MODEL_TYPE_FLUX_DEV # Generic tag
        )

    def is_text_encoder_not_needed_for_training(self, args):
        # For FLUX, text encoders are always used by the DiT for conditioning,
        # and also for sampling. So, they are "needed" in a general sense.
        # This flag in the base trainer is more about whether they can be *deleted*
        # after caching IF they are also not trained.
        # If text encoders are not trained AND their outputs are cached, they might be deletable from VRAM.
        # However, they are still needed for the sample_images step.
        
        # If TEs are not trained AND caching is on, they are moved to CPU after caching anyway.
        # If TEs are not trained AND caching is OFF, they are kept on GPU for on-the-fly encoding.
        # If TEs ARE trained, they are on GPU.
        
        # The main loop's sample_images call needs them.
        # So, they are never truly "not needed" to the point of deletion from the trainer object.
        return False
    
    def update_metadata(self, metadata: dict, args: argparse.Namespace):
        """
        Update metadata with FLUX-specific training arguments.
        """
        # REMOVED: super().update_metadata(metadata, args) # Parent does not have this method

        metadata["ss_flux_model_type"] = self.model_type_str 
        
        if args.t5xxl_max_token_length is None:
            if self.model_type_str == flux_utils.MODEL_TYPE_FLUX_SCHNELL or \
               self.model_type_str == flux_utils.MODEL_TYPE_CHROMA:
                effective_t5_max_length = 256
            else: 
                effective_t5_max_length = 512
        else:
            effective_t5_max_length = args.t5xxl_max_token_length
        
        metadata["ss_t5xxl_max_token_length"] = effective_t5_max_length
        metadata["ss_apply_t5_attn_mask"] = args.apply_t5_attn_mask
        metadata["ss_guidance_scale_flux"] = args.guidance_scale 
        
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        if args.timestep_sampling in ["sigmoid", "shift", "flux_shift"]:
            metadata["ss_sigmoid_scale"] = args.sigmoid_scale
        if args.timestep_sampling == "shift":
            metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift
        if args.timestep_sampling in ["logit_normal", "mode", "cosmap"]: 
            if args.logit_mean is not None: metadata["ss_logit_mean"] = args.logit_mean
            if args.logit_std is not None: metadata["ss_logit_std"] = args.logit_std
            if args.mode_scale is not None: metadata["ss_mode_scale"] = args.mode_scale
        
        metadata["ss_model_prediction_type_flux"] = args.model_prediction_type

        if args.clip_l:
            metadata["ss_clip_l_path"] = os.path.basename(args.clip_l) if os.path.exists(args.clip_l) else args.clip_l
        if args.t5xxl:
            metadata["ss_t5xxl_path"] = os.path.basename(args.t5xxl) if os.path.exists(args.t5xxl) else args.t5xxl
        if args.ae:
            metadata["ss_ae_path_flux"] = os.path.basename(args.ae) if os.path.exists(args.ae) else args.ae
        
        if args.controlnet_model_name_or_path:
            metadata["ss_controlnet_model_flux"] = os.path.basename(args.controlnet_model_name_or_path) \
                if os.path.exists(args.controlnet_model_name_or_path) else args.controlnet_model_name_or_path
            
    def prepare_unet_with_accelerator(
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module # unet is the FLUX model here
    ) -> torch.nn.Module:
        """
        Prepares the U-Net (FLUX model in this case) with the accelerator.
        The default behavior from the parent class is usually sufficient.
        """
        logger.info(f"Preparing FLUX model (U-Net) with accelerator. Original device: {unet.device}")
        prepared_unet = accelerator.prepare(unet)
        logger.info(f"FLUX model (U-Net) prepared. New device: {prepared_unet.device}")
        return prepared_unet
    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        """
        Post-processing for the network after it's created and before it's applied.
        For Flux LoRA, typically no extra steps are needed here as LoRA application
        handles the necessary integration.
        """
        pass

    def is_train_text_encoder(self, args):
        """
        Determines if any text encoder (CLIP-L or T5XXL) is set to be trained.
        This relies on self.train_clip_l and self.train_t5xxl which are
        determined in self.assert_extra_args based on args.network_train_unet_only
        and the model type.
        """
        # These flags are set in assert_extra_args
        return self.train_clip_l or self.train_t5xxl
    
    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            if self.model_type_str is None and args.pretrained_model_name_or_path:
                try:
                    self.model_type_str, _, _, _ = flux_utils.analyze_checkpoint_state(args.pretrained_model_name_or_path)
                except Exception as e:
                    logger.error(f"Could not analyze checkpoint state for caching strategy: {e}. Assuming no CLIP-L.")
                    self.model_type_str = flux_utils.MODEL_TYPE_CHROMA  # Fallback to Chroma for safety

            has_clip_l = False
            if self.model_type_str in [flux_utils.MODEL_TYPE_FLUX_DEV, flux_utils.MODEL_TYPE_FLUX_SCHNELL]:
                has_clip_l = True
            elif args.clip_l is not None:
                has_clip_l = True
                logger.warning("CLIP-L path provided, but model type is Chroma or undetermined. CLIP-L will be ignored for Chroma.")

            return strategy_flux.FluxTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk,
                args.text_encoder_batch_size,
                args.skip_cache_check,
                has_clip_l=has_clip_l,
                is_train_clip_l=self.train_clip_l,
                is_train_t5=self.train_t5xxl,
                apply_t5_attn_mask=args.apply_t5_attn_mask,
            )
        return None

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[train_util.DatasetGroup, train_util.MinimalDataset],
        val_dataset_group: Optional[train_util.DatasetGroup],
    ):
        # REMOVED: super().assert_extra_args(args, train_dataset_group, val_dataset_group)
        # The parent's assert_extra_args calls verify_bucket_reso_steps(64), 
        # which is for SD/SDXL. Flux needs 32, which is handled below.

        if args.fp8_base_unet:
            args.fp8_base = True 

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            logger.warning(
                "cache_text_encoder_outputs_to_disk is enabled, so cache_text_encoder_outputs is also enabled"
            )
            args.cache_text_encoder_outputs = True

        if args.cache_text_encoder_outputs:
            assert (
                train_dataset_group.is_text_encoder_output_cacheable()
            ), "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or caption_tag_dropout_rate cannot be used"

        # Analyze model_type_str if not already done (e.g. by load_target_model, but that's later)
        # This is needed for train_clip_l and train_t5xxl determination.
        if self.model_type_str is None and args.pretrained_model_name_or_path:
            try:
                self.model_type_str, _, _, _ = flux_utils.analyze_checkpoint_state(args.pretrained_model_name_or_path)
            except Exception as e:
                logger.error(
                    f"Could not analyze checkpoint state in assert_extra_args: {e}. "
                    f"Flags for training CLIP-L/T5XXL might be based on defaults."
                )
        
        # Determine which text encoders are to be trained
        _train_clip_l = False
        _train_t5 = False

        if not args.network_train_unet_only:  # If TEs *can* be trained
            if self.model_type_str == flux_utils.MODEL_TYPE_CHROMA:
                _train_t5 = True 
                _train_clip_l = False 
                if args.clip_l is not None:
                     logger.warning("Chroma model type specified. --clip_l is provided but will be ignored for training.")
            elif self.model_type_str in [flux_utils.MODEL_TYPE_FLUX_DEV, flux_utils.MODEL_TYPE_FLUX_SCHNELL]:
                if args.clip_l is not None:
                    _train_clip_l = True
                else:
                    # This case should ideally raise an error if --clip_l is truly required for these models for TE training
                    logger.error(
                        f"Model type {self.model_type_str} is configured for Text Encoder training, "
                        f"but --clip_l argument (path to CLIP-L model) was not provided. CLIP-L cannot be trained."
                    )
                    # Depending on strictness, you might want to raise ValueError here
                    # raise ValueError(f"Model type {self.model_type_str} requires --clip_l for Text Encoder training.")
                    _train_clip_l = False 
                _train_t5 = True 
            else: 
                logger.warning(
                    f"Unknown or undetermined model type ('{self.model_type_str}'). "
                    f"Falling back to generic TE training flags based on --clip_l and --network_train_unet_only."
                )
                if args.clip_l is not None: # Only consider training CLIP-L if its path is given
                    _train_clip_l = True
                _train_t5 = True 
        
        self.train_clip_l = _train_clip_l
        self.train_t5xxl = _train_t5
        logger.info(f"DEBUG: In assert_extra_args: self.train_clip_l set to {self.train_clip_l}, self.train_t5xxl set to {self.train_t5xxl}")


        if args.max_token_length is not None: # This refers to SD-style max_token_length
            logger.warning("max_token_length (SD style) is not used in Flux training. Use --t5xxl_max_token_length for T5.")

        assert (
            args.blocks_to_swap is None or args.blocks_to_swap == 0
        ) or not args.cpu_offload_checkpointing, "blocks_to_swap is not supported with cpu_offload_checkpointing"

        if args.split_mode:
            if args.blocks_to_swap is not None:
                logger.warning("split_mode is deprecated and ignored because --blocks_to_swap is set.")
            else:
                logger.warning("split_mode is deprecated. Use --blocks_to_swap. Setting --blocks_to_swap 18.")
                args.blocks_to_swap = 18

        train_dataset_group.verify_bucket_reso_steps(32) 
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype, accelerator):
        loading_dtype = None if args.fp8_base else weight_dtype

        self.model_type_str, num_double, num_single, _ = flux_utils.analyze_checkpoint_state(args.pretrained_model_name_or_path)
        
        # Use self.model_type_str to make decisions
        # is_schnell is a bit limiting now, model_type_str is better
        # self.is_schnell = (self.model_type_str == flux_utils.MODEL_TYPE_FLUX_SCHNELL)

        _, model = flux_utils.load_flow_model( # model_type_str is already stored in self.model_type_str
            args.pretrained_model_name_or_path, loading_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors
        )
        
        if args.fp8_base:
            if model.dtype == torch.float8_e4m3fnuz or model.dtype == torch.float8_e5m2 or model.dtype == torch.float8_e5m2fnuz:
                raise ValueError(f"Unsupported fp8 model dtype: {model.dtype}")
            elif model.dtype == torch.float8_e4m3fn:
                logger.info("Loaded fp8 FLUX model")
            else:
                logger.info("Cast FLUX model to fp8. This may take a while.")
                model.to(torch.float8_e4m3fn)

        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
        if self.is_swapping_blocks:
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)

        clip_l = None
        if args.clip_l: # Only load if path is provided
            clip_l = flux_utils.load_clip_l(args.clip_l, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)
            clip_l.eval()
            logger.info("CLIP-L loaded.")
        elif self.model_type_str != flux_utils.MODEL_TYPE_CHROMA:
            # If not Chroma and no clip_l path provided, it's an issue for Dev/Schnell
            raise ValueError(f"Model type {self.model_type_str} requires CLIP-L, but --clip_l argument was not provided.")
        else:
            logger.info("CLIP-L not loaded as --clip_l not provided (expected for Chroma).")


        if args.fp8_base and not args.fp8_base_unet: # This fp8 logic seems specific to T5XXL
            loading_dtype_t5 = None 
        else:
            loading_dtype_t5 = weight_dtype

        t5xxl = flux_utils.load_t5xxl(args.t5xxl, loading_dtype_t5, "cpu", disable_mmap=args.disable_mmap_load_safetensors)
        t5xxl.eval()
        if args.fp8_base and not args.fp8_base_unet:
            if t5xxl.dtype == torch.float8_e4m3fnuz or t5xxl.dtype == torch.float8_e5m2 or t5xxl.dtype == torch.float8_e5m2fnuz:
                raise ValueError(f"Unsupported fp8 T5XXL model dtype: {t5xxl.dtype}")
            elif t5xxl.dtype == torch.float8_e4m3fn:
                logger.info("Loaded fp8 T5XXL model")
        
        logger.info("T5XXL loaded.")

        ae = flux_utils.load_ae(args.ae, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)
        logger.info("AE loaded.")
        
        text_encoders = []
        if clip_l is not None:
            text_encoders.append(clip_l)
        else: # Placeholder for consistent indexing if needed, or adjust downstream logic
            text_encoders.append(None) 
        text_encoders.append(t5xxl)

        return flux_utils.MODEL_VERSION_FLUX_V1, text_encoders, ae, model
    def get_text_encoding_strategy(self, args):
        logger.info("DEBUG_STRATEGY: FluxNetworkTrainer.get_text_encoding_strategy called")
        return strategy_flux.FluxTextEncodingStrategy(apply_t5_attn_mask=args.apply_t5_attn_mask)
    def get_tokenize_strategy(self, args):
        # self.model_type_str should be set by load_target_model if called before,
        # but this function might be called first by the base class.
        # Re-analyze or ensure load_target_model is called first.
        # For safety, re-analyze here if not set.
        if self.model_type_str is None:
             # This might be redundant if load_target_model already did it.
            self.model_type_str, _, _, _ = flux_utils.analyze_checkpoint_state(args.pretrained_model_name_or_path)

        if args.t5xxl_max_token_length is None:
            if self.model_type_str == flux_utils.MODEL_TYPE_FLUX_SCHNELL or \
               self.model_type_str == flux_utils.MODEL_TYPE_CHROMA:
                t5xxl_max_token_length = 256
            else: # Dev or Unknown (safer default for Dev)
                t5xxl_max_token_length = 512
        else:
            t5xxl_max_token_length = args.t5xxl_max_token_length

        logger.info(f"t5xxl_max_token_length: {t5xxl_max_token_length}")
        # Pass clip_l_tokenizer_path=None if args.clip_l is not provided
        clip_l_path_for_tokenizer = args.clip_l_tokenizer_path if args.clip_l is not None else None
        if args.clip_l is None and args.clip_l_tokenizer_path is not None:
            logger.warning("--clip_l_tokenizer_path provided but --clip_l is not. CLIP-L tokenizer will not be loaded.")
            clip_l_path_for_tokenizer = None

        return strategy_flux.FluxTokenizeStrategy(
            t5xxl_max_token_length, 
            args.tokenizer_cache_dir,
            clip_l_tokenizer_path=clip_l_path_for_tokenizer
        )

    def get_tokenizers(self, tokenize_strategy: strategy_flux.FluxTokenizeStrategy):
        tokenizers = []
        if tokenize_strategy.clip_l_tokenizer is not None:
            tokenizers.append(tokenize_strategy.clip_l_tokenizer)
        else:
            tokenizers.append(None) # Placeholder for CLIP-L if not used
        tokenizers.append(tokenize_strategy.t5xxl_tokenizer)
        return tokenizers
        
    def get_text_encoders_train_flags(self, args, text_encoders):
        # text_encoders[0] is clip_l, text_encoders[1] is t5xxl
        train_clip_l_flag = self.train_clip_l and (text_encoders[0] is not None)
        return [train_clip_l_flag, self.train_t5xxl]
    
    def get_latents_caching_strategy(self, args):
        # Flux models (like FLUX.1 Dev/Schnell/Chroma) use a VAE that outputs latents
        # with a different downscale factor than SD1.5/2.x/XL.
        # For Flux, the typical downscale factor for the AE is 16 (e.g., 1024 -> 64 latents).
        # However, the `FluxLatentsCachingStrategy` itself calculates the suffix based on image_size
        # and its `is_disk_cached_latents_expected` uses a downscale_factor of 8, which seems incorrect for Flux AE.
        # The `cache_batch_latents` in FluxLatentsCachingStrategy uses vae.encode directly.
        # The strategy_base._default_is_disk_cached_latents_expected takes `latents_stride`
        # which is the VAE downscale factor. For Flux AE this is 16.

        # For now, we will instantiate FluxLatentsCachingStrategy.
        # We might need to review FluxLatentsCachingStrategy if the downscale factor of 8 it uses internally is problematic.
        logger.info("Using FluxLatentsCachingStrategy for Flux models.")
        return strategy_flux.FluxLatentsCachingStrategy(
            args.cache_latents_to_disk,
            args.vae_batch_size, # or args.train_batch_size if None, but vae_batch_size is more specific
            args.skip_cache_check,
        )
    
    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            # Move diffusion model and VAE to CPU to free up VRAM for text encoders
            if not args.lowram:
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device = vae.device
                org_unet_device = unet.device
                vae.to("cpu")
                unet.to("cpu")
                clean_memory_on_device(accelerator.device)

            logger.info("move text encoders to gpu for caching")
            active_text_encoders_for_caching = []

            # Handle CLIP-L (expected at text_encoders[0])
            clip_l_for_caching = None
            if text_encoders[0] is not None:  # CLIP-L model object exists
                text_encoders[0].to(accelerator.device, dtype=weight_dtype)
                clip_l_for_caching = text_encoders[0]
            active_text_encoders_for_caching.append(clip_l_for_caching)  # Append CLIP-L or None

            # Handle T5XXL (expected at text_encoders[1])
            t5xxl_for_caching = None
            if len(text_encoders) > 1 and text_encoders[1] is not None:  # T5XXL model object exists
                text_encoders[1].to(accelerator.device)
                if hasattr(text_encoders[1], 'dtype') and text_encoders[1].dtype == torch.float8_e4m3fn:
                    if hasattr(self, 'prepare_text_encoder_fp8'):
                        self.prepare_text_encoder_fp8(1, text_encoders[1], text_encoders[1].dtype, weight_dtype)
                    else:
                        text_encoders[1].to(dtype=weight_dtype)
                else:
                    text_encoders[1].to(dtype=weight_dtype)
                t5xxl_for_caching = text_encoders[1]
            else:
                logger.error("T5XXL model is required for caching but is None or not provided.")
                raise ValueError("T5XXL model (text_encoders[1]) is missing during caching setup.")
            active_text_encoders_for_caching.append(t5xxl_for_caching)

            # Set the caching strategy explicitly before caching
            caching_strategy = self.get_text_encoder_outputs_caching_strategy(args)
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(caching_strategy)

            # Cache outputs for the main dataset
            logger.info("Caching text encoder outputs for dataset")
            with accelerator.autocast():
                try:
                    dataset.new_cache_text_encoder_outputs(active_text_encoders_for_caching, accelerator)
                except Exception as e:
                    logger.error(f"Failed to cache text encoder outputs: {e}")
                    raise

            # Cache outputs for sample prompts if specified
            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompt: {args.sample_prompts}")
                tokenize_strategy: strategy_flux.FluxTokenizeStrategy = strategy_base.TokenizeStrategy.get_strategy()
                current_text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
                assert isinstance(current_text_encoding_strategy, strategy_flux.FluxTextEncodingStrategy), \
                    f"Expected FluxTextEncodingStrategy, got {type(current_text_encoding_strategy)}"

                prompts = train_util.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p not in sample_prompts_te_outputs:
                                logger.info(f"cache Text Encoder outputs for prompt: {p}")
                                tokens_and_masks = tokenize_strategy.tokenize(p)
                                sample_prompts_te_outputs[p] = current_text_encoding_strategy.encode_tokens(
                                    tokenize_strategy, active_text_encoders_for_caching, tokens_and_masks, args.apply_t5_attn_mask
                                )
                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            accelerator.wait_for_everyone()

            # Move text encoders back to CPU if they are not being trained
            if active_text_encoders_for_caching[0] is not None and not self.train_clip_l:
                logger.info("move CLIP-L back to cpu")
                active_text_encoders_for_caching[0].to("cpu")
            if active_text_encoders_for_caching[1] is not None and not self.train_t5xxl:
                logger.info("move T5XXL back to cpu")
                active_text_encoders_for_caching[1].to("cpu")

            clean_memory_on_device(accelerator.device)

            # Move diffusion model and VAE back to original device
            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # If not caching, move available text encoders to device
            if text_encoders[0] is not None:
                text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            if len(text_encoders) > 1 and text_encoders[1] is not None:
                if hasattr(text_encoders[1], 'dtype') and text_encoders[1].dtype == torch.float8_e4m3fn:
                    text_encoders[1].to(accelerator.device)
                else:
                    text_encoders[1].to(accelerator.device, dtype=weight_dtype)

    def get_models_for_text_encoding(self, args, accelerator, text_encoders): # text_encoders are the original loaded models e.g. [clip_l_model_or_None, t5_model]
        return text_encoders
    
    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents, # Original (unpacked) latents from VAE, shape [B, C, H_orig_lat, W_orig_lat]
        batch,
        text_encoder_conds, 
        unet: flux_models.Flux, # This is the FLUX DiT model
        network, # LoRA network
        weight_dtype,
        train_unet, # Flag, not the unet model itself
        is_train=True,
    ):
        logger.info("DEBUG: Entered get_noise_pred_and_target.")
        noise = torch.randn_like(latents) # noise matches original latent shape
        bsz = latents.shape[0]

        # noisy_model_input is original latents + noise
        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )
        logger.info(f"DEBUG: Noisy latents (before packing for DiT input) shape: {noisy_model_input.shape}, device: {noisy_model_input.device}")

        # MODIFIED: Determine patch grid dimensions based on original latent shape
        patch_factor = 2 # Standard for FLUX-like models where patches are 2x2 from latent space
        patch_grid_h = noisy_model_input.shape[2] // patch_factor
        patch_grid_w = noisy_model_input.shape[3] // patch_factor
        logger.info(f"DEBUG: Calculated patch_grid_h={patch_grid_h}, patch_grid_w={patch_grid_w} from original latent shape {noisy_model_input.shape}")

        # Pack the noisy latents for DiT input
        packed_noisy_model_input = flux_utils.pack_latents(noisy_model_input)
        logger.info(f"DEBUG: Packed noisy latents (DiT input 'img') shape: {packed_noisy_model_input.shape}")


        # MODIFIED: Generate linear image patch IDs
        num_image_patches = patch_grid_h * patch_grid_w
        img_ids_linear = torch.arange(num_image_patches, device=accelerator.device, dtype=torch.long).unsqueeze(0).repeat(bsz, 1)
        logger.info(f"DEBUG: Generated img_ids_linear shape: {img_ids_linear.shape}")


        guidance_vec = torch.full((bsz,), float(args.guidance_scale), device=accelerator.device)

        l_pooled, t5_out, txt_ids, t5_attn_mask_from_conds = text_encoder_conds
        # ... (moving conds to device logic) ...
        if l_pooled is not None: l_pooled = l_pooled.to(accelerator.device, dtype=weight_dtype)
        if t5_out is not None: t5_out = t5_out.to(accelerator.device, dtype=weight_dtype)
        if txt_ids is not None: txt_ids = txt_ids.to(accelerator.device) # long
        if t5_attn_mask_from_conds is not None: t5_attn_mask_from_conds = t5_attn_mask_from_conds.to(accelerator.device)


        if args.gradient_checkpointing:
            packed_noisy_model_input.requires_grad_(True) 
            if l_pooled is not None and l_pooled.dtype.is_floating_point: l_pooled.requires_grad_(True)
            if t5_out is not None and t5_out.dtype.is_floating_point: t5_out.requires_grad_(True)
            # txt_ids and img_ids_linear are indices, should not require grad
            guidance_vec.requires_grad_(True)
        
        # ... (existing debug logs for conds) ...
        logger.info(f"FLUX_TRAINER_DEBUG: img_ids_linear.shape: {img_ids_linear.shape}, img_ids_linear.ndim: {img_ids_linear.ndim}, img_ids_linear.device: {img_ids_linear.device}")

        actual_t5_attn_mask_for_model = t5_attn_mask_from_conds if args.apply_t5_attn_mask else None

        # MODIFIED: Add patch_grid_h, patch_grid_w to call_dit_func
        def call_dit_func(img_in, img_ids_in, txt_in, txt_ids_in, y_in, timesteps_in, guidance_in, txt_attention_mask_in, pgh, pgw):
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                logger.info("DEBUG: Calling unet.forward()...")
                # unet is the FLUX DiT model
                model_pred_out = unet(
                    img=img_in, img_ids=img_ids_in, txt=txt_in, txt_ids=txt_ids_in, y=y_in,
                    timesteps=timesteps_in / 1000, guidance=guidance_in, txt_attention_mask=txt_attention_mask_in,
                    patch_grid_h=pgh, patch_grid_w=pgw, # Pass patch grid dimensions
                )
                logger.info("DEBUG: unet.forward() returned.")
            return model_pred_out

        # model_pred from DiT is in packed format: [B, NumPatches, FeaturesPerPatch]
        # e.g. [B, 1008, 64] for FLUX Chroma
        model_pred = call_dit_func(
            packed_noisy_model_input, img_ids_linear, t5_out, txt_ids, 
            l_pooled, 
            timesteps, guidance_vec, actual_t5_attn_mask_for_model,
            patch_grid_h, patch_grid_w # Pass patch grid dimensions
        )
        logger.info(f"DEBUG: model_pred (output from DiT, packed) shape: {model_pred.shape}")


        # MODIFIED: Unpack model_pred to match original latent space shape
        model_pred_unpacked = flux_utils.unpack_latents(model_pred.contiguous(), patch_grid_h, patch_grid_w)
        logger.info(f"DEBUG: model_pred_unpacked shape: {model_pred_unpacked.shape}. Expected to match noisy_model_input: {noisy_model_input.shape}")

        # noisy_model_input is in original latent shape [B, C_orig, H_orig_lat, W_orig_lat]
        model_pred_unpacked, weighting = flux_train_utils.apply_model_prediction_type(args, model_pred_unpacked, noisy_model_input, sigmas)
        
        # Target is also in original latent shape
        target = noise - latents 
        logger.info(f"DEBUG: target shape: {target.shape}")


        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                    diff_output_pr_indices.append(i)

            if len(diff_output_pr_indices) > 0:
                network.set_multiplier(0.0)
                if hasattr(unet, 'prepare_block_swap_before_forward'): 
                    unet.prepare_block_swap_before_forward() 
                with torch.no_grad():
                    l_pooled_prior = l_pooled[diff_output_pr_indices] if l_pooled is not None else None
                    t5_out_prior = t5_out[diff_output_pr_indices]
                    txt_ids_prior = txt_ids[diff_output_pr_indices]
                    t5_attn_mask_prior = actual_t5_attn_mask_for_model[diff_output_pr_indices] if actual_t5_attn_mask_for_model is not None else None
                    
                    # model_pred_prior from DiT is also packed
                    model_pred_prior = call_dit_func(
                        packed_noisy_model_input[diff_output_pr_indices],
                        img_ids_linear[diff_output_pr_indices],
                        t5_out_prior,
                        txt_ids_prior,
                        l_pooled_prior,
                        timesteps[diff_output_pr_indices],
                        guidance_vec[diff_output_pr_indices] if guidance_vec is not None else None,
                        t5_attn_mask_prior,
                        patch_grid_h, patch_grid_w # Pass patch grid dimensions
                    )
                network.set_multiplier(1.0) 
                
                # MODIFIED: Unpack model_pred_prior
                model_pred_prior_unpacked = flux_utils.unpack_latents(model_pred_prior.contiguous(), patch_grid_h, patch_grid_w)
                
                model_pred_prior_unpacked, _ = flux_train_utils.apply_model_prediction_type(
                    args,
                    model_pred_prior_unpacked, 
                    noisy_model_input[diff_output_pr_indices],
                    sigmas[diff_output_pr_indices] if sigmas is not None else None,
                )
                target[diff_output_pr_indices] = model_pred_prior_unpacked.to(target.dtype)
        
        return model_pred_unpacked, target, timesteps, weighting


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    flux_train_utils.add_flux_train_arguments(parser)
    # Remove --clip_l from required if Chroma, or handle it in arg parsing
    # For now, assume user won't provide --clip_l for Chroma.
    # We can add a check later if model_type is Chroma and --clip_l is given.

    parser.add_argument(
        "--split_mode",
        action="store_true",
        help="[Deprecated] This option is deprecated. Please use `--blocks_to_swap` instead.",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()

    # Add a check for Chroma: if --clip_l is not provided, it's fine.
    # If it *is* provided with a Chroma model, we might want to warn or ignore it.
    # This basic check can be done after args are parsed.
    if "chroma" in args.pretrained_model_name_or_path.lower() and args.clip_l is not None:
        logger.warning("Chroma model detected. --clip_l argument will be ignored as Chroma does not use CLIP-L.")
        # Optionally set args.clip_l to None here if downstream code strictly relies on it being None for Chroma
        # args.clip_l = None 
        # However, the load_target_model logic now handles args.clip_l being None or provided.

    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = FluxNetworkTrainer()
    trainer.train(args)