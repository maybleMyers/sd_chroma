# flux_train_network.py
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import copy
import math
import random
from typing import Any, Optional, Union

import torch
from accelerate import Accelerator

from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

import train_network
from library import (
    flux_models,
    flux_train_utils,
    flux_utils,
    sd3_train_utils,
    strategy_base,
    strategy_flux,
    train_util,
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

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        """
        Post-processing for the network after it's created and before it's applied.
        For Flux LoRA, typically no extra steps are needed here as LoRA application
        handles the necessary integration.
        """
        pass
    
    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            if self.model_type_str is None and args.pretrained_model_name_or_path: # Ensure model_type_str is available
                try:
                    self.model_type_str, _, _, _ = flux_utils.analyze_checkpoint_state(args.pretrained_model_name_or_path)
                except Exception as e:
                    logger.error(f"Could not analyze checkpoint state for caching strategy: {e}. Defaulting has_clip_l.")
            
            has_clip_l = False # Default for Chroma or if undetermined
            if self.model_type_str in [flux_utils.MODEL_TYPE_FLUX_DEV, flux_utils.MODEL_TYPE_FLUX_SCHNELL]:
                has_clip_l = True
            elif args.clip_l is not None and self.model_type_str != flux_utils.MODEL_TYPE_CHROMA: # Fallback if model_type_str is None but clip_l provided
                has_clip_l = True


            return strategy_flux.FluxTextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk,
                args.text_encoder_batch_size, 
                args.skip_cache_check,
                has_clip_l=has_clip_l, 
                is_train_clip_l=self.train_clip_l, # Relies on self.train_clip_l being set by assert_extra_args
                is_train_t5=self.train_t5xxl,     # Relies on self.train_t5xxl being set by assert_extra_args
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
            active_text_encoders_for_caching = [] # Initialize

            # Handle CLIP-L (expected at text_encoders[0])
            clip_l_for_caching = None
            if text_encoders[0] is not None: # CLIP-L model object exists
                text_encoders[0].to(accelerator.device, dtype=weight_dtype)
                clip_l_for_caching = text_encoders[0]
            active_text_encoders_for_caching.append(clip_l_for_caching) # Append CLIP-L or None

            # Handle T5XXL (expected at text_encoders[1])
            t5xxl_for_caching = None
            if len(text_encoders) > 1 and text_encoders[1] is not None: # T5XXL model object exists
                text_encoders[1].to(accelerator.device) # Move to device first
                if hasattr(text_encoders[1], 'dtype') and text_encoders[1].dtype == torch.float8_e4m3fn: # Check if it's already fp8
                    # If T5 is fp8, it might have specific handling (like prepare_text_encoder_fp8)
                    # Assuming prepare_text_encoder_fp8 handles dtype conversion if necessary or prepares it.
                    if hasattr(self, 'prepare_text_encoder_fp8'): # ensure method exists
                         self.prepare_text_encoder_fp8(1, text_encoders[1], text_encoders[1].dtype, weight_dtype)
                    # If not fp8, convert to weight_dtype
                else:
                    text_encoders[1].to(dtype=weight_dtype)
                t5xxl_for_caching = text_encoders[1]
            else:
                # This case should ideally not happen if flux_train_network.py setup is correct,
                # as T5XXL is mandatory for FLUX.
                logger.error("T5XXL model (text_encoders[1]) is None or not provided during caching setup.")
                # Fallback or raise error depending on strictness, appending None for now.
            active_text_encoders_for_caching.append(t5xxl_for_caching) # Append T5XXL or None

            # At this point, active_text_encoders_for_caching should be [clip_l_or_None, t5xxl_model_or_None]
            # Ensure T5XXL is actually present if needed by strategy
            if active_text_encoders_for_caching[1] is None:
                 logger.warning("T5XXL model is effectively None for caching. This might lead to errors in FluxTextEncodingStrategy if it strictly requires T5.")


            # Cache outputs for the main dataset
            with accelerator.autocast():
                dataset.new_cache_text_encoder_outputs(active_text_encoders_for_caching, accelerator)

            # Cache outputs for sample prompts if specified
            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompt: {args.sample_prompts}")
                tokenize_strategy: strategy_flux.FluxTokenizeStrategy = strategy_base.TokenizeStrategy.get_strategy()
                
                # Get the globally set text encoding strategy, which should be FluxTextEncodingStrategy
                current_text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()
                assert isinstance(current_text_encoding_strategy, strategy_flux.FluxTextEncodingStrategy), \
                    f"Expected FluxTextEncodingStrategy for sample prompts, got {type(current_text_encoding_strategy)}"

                prompts = train_util.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}
                with accelerator.autocast(), torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p not in sample_prompts_te_outputs:
                                logger.info(f"cache Text Encoder outputs for prompt: {p}")
                                tokens_and_masks = tokenize_strategy.tokenize(p) # Tokenize for both/all
                                # active_text_encoders_for_caching is [clip_l_or_None, t5xxl_or_None]
                                sample_prompts_te_outputs[p] = current_text_encoding_strategy.encode_tokens(
                                    tokenize_strategy, active_text_encoders_for_caching, tokens_and_masks, args.apply_t5_attn_mask
                                )
                self.sample_prompts_te_outputs = sample_prompts_te_outputs
            
            accelerator.wait_for_everyone()

            # Move text encoders back to CPU if they are not being trained
            # Check if the specific encoder in active_text_encoders_for_caching is not None before moving
            if active_text_encoders_for_caching[0] is not None and not self.is_train_text_encoder(args): # Checks all TEs based on base class
                # More specific check might be needed if self.is_train_text_encoder doesn't discern CLIP-L vs T5XXL
                # Assuming if any TE is trained, self.is_train_text_encoder is True.
                # Or use self.train_clip_l if it's accurately set.
                if hasattr(self, 'train_clip_l') and not self.train_clip_l :
                    logger.info("move CLIP-L back to cpu")
                    active_text_encoders_for_caching[0].to("cpu")
                elif not hasattr(self, 'train_clip_l') and not self.is_train_text_encoder(args): # Fallback if train_clip_l not present
                    logger.info("move CLIP-L back to cpu (general TE train flag)")
                    active_text_encoders_for_caching[0].to("cpu")


            if active_text_encoders_for_caching[1] is not None and not self.train_t5xxl: # self.train_t5xxl should exist
                logger.info("move t5XXL back to cpu")
                active_text_encoders_for_caching[1].to("cpu")
            
            clean_memory_on_device(accelerator.device)

            # Move diffusion model and VAE back to original device
            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device)
                unet.to(org_unet_device)
        else:
            # If not caching, move all available text encoders to device for training
            if text_encoders[0] is not None: # CLIP-L
                text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            
            if len(text_encoders) > 1 and text_encoders[1] is not None: # T5XXL
                # Check for fp8 before setting dtype to weight_dtype, as fp8 has its own dtype.
                if hasattr(text_encoders[1], 'dtype') and text_encoders[1].dtype == torch.float8_e4m3fn:
                    text_encoders[1].to(accelerator.device) # Already fp8, just move to device
                else:
                    text_encoders[1].to(accelerator.device, dtype=weight_dtype)
            elif len(text_encoders) == 1 and text_encoders[0] is not None and hasattr(text_encoders[0], 'config') and 't5' in text_encoders[0].config.model_type.lower():
                # Case where only T5XXL is passed as text_encoders[0]
                if hasattr(text_encoders[0], 'dtype') and text_encoders[0].dtype == torch.float8_e4m3fn:
                    text_encoders[0].to(accelerator.device)
                else:
                    text_encoders[0].to(accelerator.device, dtype=weight_dtype)

    def get_models_for_text_encoding(self, args, accelerator, text_encoders): # text_encoders are the original loaded models e.g. [clip_l_model_or_None, t5_model]
        return text_encoders
    
    # ... (rest of the methods need careful review for handling optional CLIP-L) ...
    # For example, in get_noise_pred_and_target, text_encoder_conds might have None for CLIP-L parts.

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds, # This will be a list [clip_l_outputs_or_None, t5_outputs]
        unet: flux_models.Flux,
        network, # LoRA network
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        logger.info("DEBUG: Entered get_noise_pred_and_target.") # <--- DEBUG
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]

        noisy_model_input, timesteps, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
        )
        logger.info(f"DEBUG: Noisy latents shape: {noisy_model_input.shape}, device: {noisy_model_input.device}") # <--- DEBUG

        packed_noisy_model_input = flux_utils.pack_latents(noisy_model_input)
        packed_latent_height, packed_latent_width = noisy_model_input.shape[2] // 2, noisy_model_input.shape[3] // 2
        img_ids = flux_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=accelerator.device)
        guidance_vec = torch.full((bsz,), float(args.guidance_scale), device=accelerator.device)

        if args.gradient_checkpointing:
            packed_noisy_model_input.requires_grad_(True) # Changed from noisy_model_input
            if text_encoder_conds[0] is not None and text_encoder_conds[0].dtype.is_floating_point: # clip_l_pooled_output
                 text_encoder_conds[0].requires_grad_(True)
            if text_encoder_conds[1] is not None and text_encoder_conds[1].dtype.is_floating_point: # t5_output
                 text_encoder_conds[1].requires_grad_(True)
            if text_encoder_conds[2] is not None and text_encoder_conds[2].dtype.is_floating_point: # clip_l_hidden_states (txt_ids for flux)
                 text_encoder_conds[2].requires_grad_(True) # txt_ids for flux are token ids, not floats usually
            # t5_attn_mask (text_encoder_conds[3]) is usually bool or int, not float

            img_ids.requires_grad_(True)
            guidance_vec.requires_grad_(True)

        l_pooled, t5_out, txt_ids, t5_attn_mask_from_conds = text_encoder_conds
        logger.info(f"DEBUG: TE conds device before move - l_pooled: {l_pooled.device if l_pooled is not None else 'None'}, t5_out: {t5_out.device if t5_out is not None else 'None'}") # <--- DEBUG
        if l_pooled is not None:
            l_pooled = l_pooled.to(accelerator.device, dtype=weight_dtype)
        if t5_out is not None:
            t5_out = t5_out.to(accelerator.device, dtype=weight_dtype)
        if txt_ids is not None:
            txt_ids = txt_ids.to(accelerator.device) # token ids are long/int, not float
        if t5_attn_mask_from_conds is not None:
            t5_attn_mask_from_conds = t5_attn_mask_from_conds.to(accelerator.device)

        if args.gradient_checkpointing:
            packed_noisy_model_input.requires_grad_(True) 
            # Re-check text_encoder_conds after they have been potentially moved to GPU and assigned
            if l_pooled is not None and l_pooled.dtype.is_floating_point:
                 l_pooled.requires_grad_(True)
            if t5_out is not None and t5_out.dtype.is_floating_point:
                 t5_out.requires_grad_(True)
            if txt_ids is not None and txt_ids.dtype.is_floating_point: # Unlikely but safe check
                 txt_ids.requires_grad_(True) 
            logger.info(f"DEBUG: TE conds device after move - l_pooled: {l_pooled.device if l_pooled is not None else 'None'}, t5_out: {t5_out.device if t5_out is not None else 'None'}")
            img_ids.requires_grad_(True)
            guidance_vec.requires_grad_(True)
        
        actual_t5_attn_mask_for_model = t5_attn_mask_from_conds if args.apply_t5_attn_mask else None

        def call_dit_func(img_in, img_ids_in, txt_in, txt_ids_in, y_in, timesteps_in, guidance_in, txt_attention_mask_in):
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                logger.info("DEBUG: Calling unet.forward()...")
                model_pred_out = unet(
                    img=img_in, img_ids=img_ids_in, txt=txt_in, txt_ids=txt_ids_in, y=y_in,
                    timesteps=timesteps_in / 1000, guidance=guidance_in, txt_attention_mask=txt_attention_mask_in,
                )
                logger.info("DEBUG: unet.forward() returned.") # <--- DEBUG
            return model_pred_out

        model_pred = call_dit_func(
            packed_noisy_model_input, img_ids, t5_out, txt_ids, 
            l_pooled, # This will be None for Chroma if CLIP-L is not used. Flux model needs to handle.
            timesteps, guidance_vec, actual_t5_attn_mask_for_model,
        )

        model_pred = flux_utils.unpack_latents(model_pred, packed_latent_height, packed_latent_width)
        model_pred, weighting = flux_train_utils.apply_model_prediction_type(args, model_pred, noisy_model_input, sigmas)
        target = noise - latents

        # ... (diff_output_pr_indices logic seems okay, assumes text_encoder_conds has the right structure) ...
        if "custom_attributes" in batch:
            diff_output_pr_indices = []
            for i, custom_attributes in enumerate(batch["custom_attributes"]):
                if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                    diff_output_pr_indices.append(i)

            if len(diff_output_pr_indices) > 0:
                network.set_multiplier(0.0)
                unet.prepare_block_swap_before_forward() # ensure this is called if blocks are swapped
                with torch.no_grad():
                    # Select the corresponding parts of text_encoder_conds
                    l_pooled_prior = l_pooled[diff_output_pr_indices] if l_pooled is not None else None
                    t5_out_prior = t5_out[diff_output_pr_indices]
                    txt_ids_prior = txt_ids[diff_output_pr_indices]
                    t5_attn_mask_prior = actual_t5_attn_mask_for_model[diff_output_pr_indices] if actual_t5_attn_mask_for_model is not None else None
                    
                    model_pred_prior = call_dit_func(
                        packed_noisy_model_input[diff_output_pr_indices],
                        img_ids[diff_output_pr_indices],
                        t5_out_prior,
                        txt_ids_prior,
                        l_pooled_prior,
                        timesteps[diff_output_pr_indices],
                        guidance_vec[diff_output_pr_indices] if guidance_vec is not None else None,
                        t5_attn_mask_prior,
                    )
                network.set_multiplier(1.0) 

                model_pred_prior = flux_utils.unpack_latents(model_pred_prior, packed_latent_height, packed_latent_width)
                model_pred_prior, _ = flux_train_utils.apply_model_prediction_type(
                    args,
                    model_pred_prior,
                    noisy_model_input[diff_output_pr_indices],
                    sigmas[diff_output_pr_indices] if sigmas is not None else None,
                )
                target[diff_output_pr_indices] = model_pred_prior.to(target.dtype)
        return model_pred, target, timesteps, weighting


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