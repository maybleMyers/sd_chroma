import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import copy
import math
import random
from typing import Any, Optional, Union

import torch
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

torch.autograd.set_detect_anomaly(True)

class FluxNetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.model_type_str: Optional[str] = None # Store analyzed model type
        self.is_swapping_blocks: bool = False
        self.train_clip_l: bool = False  # Initialize here
        self.train_t5xxl: bool = False # Initialize here

    def _is_chroma_sot_loss_active(self, args: argparse.Namespace) -> bool:
        # Helper to determine if Chroma SOT loss should be used
        # Check model_type_str if already determined, though args.use_chroma_sot_loss is the primary flag
        # is_chroma_model = self.model_type_str == flux_utils.MODEL_TYPE_CHROMA
        # return args.use_chroma_sot_loss and is_chroma_model
        return args.use_chroma_sot_loss


    def all_reduce_network(self, accelerator, network):
        for param in network.parameters():
            if param.grad is not None:
                param.grad = accelerator.reduce(param.grad, reduction="mean")

    def on_step_start(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=True):
        """
        Called at the start of each training step. Implements basic logging and ensures compatibility
        with the base NetworkTrainer's train method.
        """
        # logger.info(f"Starting training step. Device: {unet.device}, dtype: {weight_dtype}") # Too verbose
        if self.is_swapping_blocks:
            unet.prepare_block_swap_before_forward()
        pass

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        if self._is_chroma_sot_loss_active(args):
            logger.info("Chroma SOT loss is active. Diffusers scheduler not strictly used for SOT sampling. Returning basic DDPMScheduler for compatibility.")
            return DDPMScheduler(num_train_timesteps=1000) # Basic scheduler, SOT uses its own timestep logic
        elif args.timestep_sampling in ["logit_normal", "mode", "cosmap"]:
            logger.info(f"Using DDPMScheduler for SD3-style timestep sampling: {args.timestep_sampling}")
            noise_scheduler = DDPMScheduler(
                beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                num_train_timesteps=1000, clip_sample=False,
            )
            if args.zero_terminal_snr:
                custom_train_functions.fix_noise_scheduler_betas_for_zero_terminal_snr(noise_scheduler)
            return noise_scheduler
        else:
            logger.info(f"FLUX timestep sampling ({args.timestep_sampling}) does not strictly require a Diffusers scheduler object here. Returning a basic DDPMScheduler for compatibility.")
            return DDPMScheduler(num_train_timesteps=1000)

    def sample_images(self, accelerator, args, epoch, global_step, device, vae, tokenizers, text_encoders, unet):
        logger.info("FluxNetworkTrainer: Calling flux_train_utils.sample_images")
        flux_train_utils.sample_images(
            accelerator,
            args,
            epoch,
            global_step,
            unet,
            vae,
            text_encoders,
            self.sample_prompts_te_outputs,
        )

    def encode_images_to_latents(self, args, vae: flux_models.AutoEncoder, images: torch.FloatTensor) -> torch.FloatTensor:
        logger.debug("FluxNetworkTrainer: using FLUX AE direct encode.")
        return vae.encode(images)

    def shift_scale_latents(self, args, latents: torch.FloatTensor) -> torch.FloatTensor:
        logger.debug("FluxNetworkTrainer: shift_scale_latents is a no-op as FLUX AE handles scaling internally.")
        return latents

    def post_process_loss(self, loss: torch.FloatTensor, args: argparse.Namespace, timesteps: torch.Tensor, noise_scheduler) -> torch.FloatTensor:
        if self._is_chroma_sot_loss_active(args):
            logger.debug("Chroma SOT loss is active. post_process_loss is a no-op here.")
            return loss # Chroma SOT loss is finalized in NetworkTrainer.process_batch

        if args.min_snr_gamma:
            logger.warning("min_snr_gamma is specified but may not be directly applicable to FLUX models without careful scheduler setup.")
            return super().post_process_loss(loss, args, timesteps, noise_scheduler)

        if args.scale_v_pred_loss_like_noise_pred or args.v_pred_like_loss or args.debiased_estimation_loss:
            logger.warning("SD-specific loss processing (scale_v_pred, v_pred_like, debiased_estimation) may not apply to FLUX.")
            return super().post_process_loss(loss, args, timesteps, noise_scheduler)

        logger.debug("FluxNetworkTrainer: post_process_loss returning loss as is (or after parent's if flags were set).")
        return loss

    def get_sai_model_spec(self, args: argparse.Namespace) -> dict:
        if self.model_type_str is None:
             logger.warning("model_type_str not set in get_sai_model_spec, attempting to analyze checkpoint again.")
             if args.pretrained_model_name_or_path:
                self.model_type_str, _, _, _ = flux_utils.analyze_checkpoint_state(args.pretrained_model_name_or_path)
             else:
                 logger.error("Cannot determine FLUX model type for SAI metadata as pretrained_model_name_or_path is missing.")
                 return train_util.get_sai_model_spec(None, args, is_sdxl=False, is_stable_diffusion_ckpt=True, flux="unknown_flux_type")

        logger.info(f"FluxNetworkTrainer: Using flux='{flux_utils.MODEL_TYPE_FLUX_DEV}' for SAI metadata (original base: {self.model_type_str}).")
        return train_util.get_sai_model_spec(
            None, args, is_sdxl=False, is_stable_diffusion_ckpt=True, flux=flux_utils.MODEL_TYPE_FLUX_DEV
        )

    def is_text_encoder_not_needed_for_training(self, args):
        return False

    def update_metadata(self, metadata: dict, args: argparse.Namespace):
        metadata["ss_flux_model_type"] = self.model_type_str
        metadata["ss_use_chroma_sot_loss"] = args.use_chroma_sot_loss

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
        if not args.use_chroma_sot_loss: # These args are not used by SOT
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
        self, args: argparse.Namespace, accelerator: Accelerator, unet: torch.nn.Module
    ) -> torch.nn.Module:
        logger.info(f"Preparing FLUX model (U-Net) with accelerator. Original device: {unet.device}")
        prepared_unet = accelerator.prepare(unet)
        logger.info(f"FLUX model (U-Net) prepared. New device: {prepared_unet.device}")
        return prepared_unet

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        pass

    def is_train_text_encoder(self, args):
        return self.train_clip_l or self.train_t5xxl

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            if self.model_type_str is None and args.pretrained_model_name_or_path:
                try:
                    self.model_type_str, _, _, _ = flux_utils.analyze_checkpoint_state(args.pretrained_model_name_or_path)
                except Exception as e:
                    logger.error(f"Could not analyze checkpoint state for caching strategy: {e}. Assuming no CLIP-L.")
                    self.model_type_str = flux_utils.MODEL_TYPE_CHROMA

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

        if self.model_type_str is None and args.pretrained_model_name_or_path:
            try:
                self.model_type_str, _, _, _ = flux_utils.analyze_checkpoint_state(args.pretrained_model_name_or_path)
            except Exception as e:
                logger.error(
                    f"Could not analyze checkpoint state in assert_extra_args: {e}. "
                    f"Flags for training CLIP-L/T5XXL might be based on defaults."
                )

        _train_clip_l = False
        _train_t5 = False
        if not args.network_train_unet_only:
            if self.model_type_str == flux_utils.MODEL_TYPE_CHROMA:
                _train_t5 = True
                _train_clip_l = False
                if args.clip_l is not None:
                     logger.warning("Chroma model type specified. --clip_l is provided but will be ignored for training.")
            elif self.model_type_str in [flux_utils.MODEL_TYPE_FLUX_DEV, flux_utils.MODEL_TYPE_FLUX_SCHNELL]:
                if args.clip_l is not None: _train_clip_l = True
                else: logger.error(f"Model type {self.model_type_str} implies CLIP-L usage, but --clip_l not provided. CLIP-L cannot be trained.")
                _train_t5 = True
            else:
                logger.warning(f"Unknown model type ('{self.model_type_str}'). Fallback TE training flags.")
                if args.clip_l is not None: _train_clip_l = True
                _train_t5 = True
        self.train_clip_l = _train_clip_l
        self.train_t5xxl = _train_t5
        logger.info(f"DEBUG: In assert_extra_args: self.train_clip_l set to {self.train_clip_l}, self.train_t5xxl set to {self.train_t5xxl}")
        if args.use_chroma_sot_loss and self.model_type_str != flux_utils.MODEL_TYPE_CHROMA:
            logger.warning(f"use_chroma_sot_loss is True, but model type is '{self.model_type_str}', not Chroma. This might lead to unexpected behavior.")
        if args.use_chroma_sot_loss:
            logger.info("Chroma SOT loss is enabled. Some FLUX/SD3 specific loss/sampling args will be ignored.")


        if args.max_token_length is not None:
            logger.warning("max_token_length (SD style) is not used in Flux training. Use --t5xxl_max_token_length for T5.")
        assert (args.blocks_to_swap is None or args.blocks_to_swap == 0) or not args.cpu_offload_checkpointing, \
            "blocks_to_swap is not supported with cpu_offload_checkpointing"
        if args.split_mode:
            logger.warning("split_mode is deprecated. Use --blocks_to_swap.")
            if args.blocks_to_swap is None: args.blocks_to_swap = 18

        train_dataset_group.verify_bucket_reso_steps(32)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(32)

    def load_target_model(self, args, weight_dtype, accelerator):
        loading_dtype = None if args.fp8_base else weight_dtype
        self.model_type_str, num_double, num_single, _ = flux_utils.analyze_checkpoint_state(args.pretrained_model_name_or_path)

        _, model = flux_utils.load_flow_model(
            args.pretrained_model_name_or_path, loading_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors
        )

        if args.fp8_base:
            if model.dtype == torch.float8_e4m3fnuz or model.dtype == torch.float8_e5m2 or model.dtype == torch.float8_e5m2fnuz:
                raise ValueError(f"Unsupported fp8 model dtype: {model.dtype}")
            elif model.dtype == torch.float8_e4m3fn: logger.info("Loaded fp8 FLUX model")
            else: logger.info("Cast FLUX model to fp8."); model.to(torch.float8_e4m3fn)

        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
        if self.is_swapping_blocks:
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device)

        clip_l = None
        if args.clip_l:
            clip_l = flux_utils.load_clip_l(args.clip_l, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)
            clip_l.eval(); logger.info("CLIP-L loaded.")
        elif self.model_type_str != flux_utils.MODEL_TYPE_CHROMA:
            raise ValueError(f"Model type {self.model_type_str} requires CLIP-L, but --clip_l not provided.")
        else: logger.info("CLIP-L not loaded (expected for Chroma).")

        loading_dtype_t5 = None if args.fp8_base and not args.fp8_base_unet else weight_dtype
        t5xxl = flux_utils.load_t5xxl(args.t5xxl, loading_dtype_t5, "cpu", disable_mmap=args.disable_mmap_load_safetensors)
        t5xxl.eval()
        if args.fp8_base and not args.fp8_base_unet:
            if t5xxl.dtype in [torch.float8_e4m3fnuz, torch.float8_e5m2, torch.float8_e5m2fnuz]:
                raise ValueError(f"Unsupported fp8 T5XXL model dtype: {t5xxl.dtype}")
            elif t5xxl.dtype == torch.float8_e4m3fn: logger.info("Loaded fp8 T5XXL model")
        logger.info("T5XXL loaded.")

        ae = flux_utils.load_ae(args.ae, weight_dtype, "cpu", disable_mmap=args.disable_mmap_load_safetensors)
        logger.info("AE loaded.")

        text_encoders = [clip_l if clip_l is not None else None, t5xxl]
        return flux_utils.MODEL_VERSION_FLUX_V1, text_encoders, ae, model

    def get_text_encoding_strategy(self, args):
        return strategy_flux.FluxTextEncodingStrategy(apply_t5_attn_mask=args.apply_t5_attn_mask)

    def get_tokenize_strategy(self, args):
        if self.model_type_str is None:
            self.model_type_str, _, _, _ = flux_utils.analyze_checkpoint_state(args.pretrained_model_name_or_path)

        if args.t5xxl_max_token_length is None:
            t5xxl_max_token_length = 256 if self.model_type_str in [flux_utils.MODEL_TYPE_FLUX_SCHNELL, flux_utils.MODEL_TYPE_CHROMA] else 512
        else: t5xxl_max_token_length = args.t5xxl_max_token_length
        logger.info(f"t5xxl_max_token_length: {t5xxl_max_token_length}")

        clip_l_path_for_tokenizer = args.clip_l_tokenizer_path if args.clip_l is not None else None
        if args.clip_l is None and args.clip_l_tokenizer_path is not None:
            logger.warning("--clip_l_tokenizer_path provided but --clip_l is not. CLIP-L tokenizer will not be loaded.")
            clip_l_path_for_tokenizer = None

        return strategy_flux.FluxTokenizeStrategy(
            t5xxl_max_token_length, args.tokenizer_cache_dir, clip_l_tokenizer_path=clip_l_path_for_tokenizer
        )

    def get_tokenizers(self, tokenize_strategy: strategy_flux.FluxTokenizeStrategy):
        tokenizers = [tokenize_strategy.clip_l_tokenizer if tokenize_strategy.clip_l_tokenizer is not None else None,
                      tokenize_strategy.t5xxl_tokenizer]
        return tokenizers

    def get_text_encoders_train_flags(self, args, text_encoders):
        train_clip_l_flag = self.train_clip_l and (text_encoders[0] is not None)
        return [train_clip_l_flag, self.train_t5xxl]

    def get_latents_caching_strategy(self, args):
        logger.info("Using FluxLatentsCachingStrategy for Flux models.")
        return strategy_flux.FluxLatentsCachingStrategy(
            args.cache_latents_to_disk, args.vae_batch_size, args.skip_cache_check,
        )

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: train_util.DatasetGroup, weight_dtype
    ):
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                logger.info("move vae and unet to cpu to save memory")
                org_vae_device, org_unet_device = vae.device, unet.device
                vae.to("cpu"); unet.to("cpu"); clean_memory_on_device(accelerator.device)

            logger.info("move text encoders to gpu for caching")
            active_text_encoders_for_caching = []
            if text_encoders[0] is not None:
                text_encoders[0].to(accelerator.device, dtype=weight_dtype)
                active_text_encoders_for_caching.append(text_encoders[0])
            else: active_text_encoders_for_caching.append(None)

            if len(text_encoders) > 1 and text_encoders[1] is not None:
                text_encoders[1].to(accelerator.device)
                target_t5_dtype = weight_dtype
                if hasattr(text_encoders[1], 'dtype') and text_encoders[1].dtype == torch.float8_e4m3fn: # If T5 is already fp8
                     if hasattr(self, 'prepare_text_encoder_fp8'): self.prepare_text_encoder_fp8(1, text_encoders[1], text_encoders[1].dtype, weight_dtype) # Ensure embeddings are float
                     target_t5_dtype = text_encoders[1].dtype # Keep fp8
                text_encoders[1].to(dtype=target_t5_dtype)
                active_text_encoders_for_caching.append(text_encoders[1])
            else: raise ValueError("T5XXL model is required for caching but is None or not provided.")

            caching_strategy = self.get_text_encoder_outputs_caching_strategy(args)
            strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(caching_strategy)

            logger.info("Caching text encoder outputs for dataset")
            with accelerator.autocast():
                try: dataset.new_cache_text_encoder_outputs(active_text_encoders_for_caching, accelerator)
                except Exception as e: logger.error(f"Failed to cache text encoder outputs: {e}"); raise

            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompt: {args.sample_prompts}")
                tokenize_strategy: strategy_flux.FluxTokenizeStrategy = strategy_base.TokenizeStrategy.get_strategy()
                current_text_encoding_strategy: strategy_flux.FluxTextEncodingStrategy = strategy_base.TextEncodingStrategy.get_strategy()
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

            if active_text_encoders_for_caching[0] is not None and not self.train_clip_l:
                logger.info("move CLIP-L back to cpu"); active_text_encoders_for_caching[0].to("cpu")
            if active_text_encoders_for_caching[1] is not None and not self.train_t5xxl:
                logger.info("move T5XXL back to cpu"); active_text_encoders_for_caching[1].to("cpu")
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae and unet back to original device")
                vae.to(org_vae_device); unet.to(org_unet_device)
        else:
            if text_encoders[0] is not None: text_encoders[0].to(accelerator.device, dtype=weight_dtype)
            if len(text_encoders) > 1 and text_encoders[1] is not None:
                target_t5_dtype_no_cache = weight_dtype
                if hasattr(text_encoders[1], 'dtype') and text_encoders[1].dtype == torch.float8_e4m3fn: target_t5_dtype_no_cache = text_encoders[1].dtype
                text_encoders[1].to(accelerator.device, dtype=target_t5_dtype_no_cache)


    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        return text_encoders

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler, # May not be used if SOT loss is active
        latents, # Clean latents from VAE (B, C, H, W)
        batch,
        text_encoder_conds,
        unet: flux_models.Flux,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        logger.debug(f"Entered get_noise_pred_and_target. Chroma SOT: {self._is_chroma_sot_loss_active(args)}")
        bsz = latents.shape[0]
        weighting = None # Default, Chroma SOT loss does its own weighting

        if self._is_chroma_sot_loss_active(args):
            logger.debug(f"SOT Path: Input latents shape {latents.shape}, dtype {latents.dtype}, NaN: {torch.isnan(latents).any()}")
            noisy_model_input_sot, target_sot, timesteps_sot, img_ids_sot, original_latent_shape = \
                flux_train_utils.prepare_sot_pairings(latents, accelerator.device)
            logger.debug(f"SOT Path: noisy_model_input_sot shape {noisy_model_input_sot.shape}, dtype {noisy_model_input_sot.dtype}, NaN: {torch.isnan(noisy_model_input_sot).any()}")
            logger.debug(f"SOT Path: target_sot shape {target_sot.shape}, dtype {target_sot.dtype}, NaN: {torch.isnan(target_sot).any()}")
            # Chroma SOT loss logic
            # latents are (B, C, H, W)
            # prepare_sot_pairings returns:
            # noisy_latents_b_tokens_c (B, L_flat, C_flat) -> this is model input
            # target_b_tokens_c (B, L_flat, C_flat) -> this is loss target
            # input_timestep_b (B,) -> these are timesteps for model and loss
            # image_pos_id_b_tokens_3 (B, L_flat, 3) -> these are img_ids for model
            # latent_shape_tuple (B,C,H,W) -> original shape for unpacking model_pred if needed
            noisy_model_input_sot, target_sot, timesteps_sot, img_ids_sot, original_latent_shape = \
                flux_train_utils.prepare_sot_pairings(latents, accelerator.device)

            # noisy_model_input_sot is already "packed" in the sense of vae_flatten.
            # No need for flux_utils.pack_latents here.
            # Its dtype is bfloat16 as per SOT util. Ensure unet can handle.
            packed_noisy_model_input = noisy_model_input_sot.to(dtype=weight_dtype) # Ensure consistent dtype for UNet
            # img_ids are the SOT-generated positional IDs
            img_ids = img_ids_sot.to(device=accelerator.device, dtype=torch.long) # Ensure long for embeddings usually
            timesteps = timesteps_sot.to(device=accelerator.device, dtype=weight_dtype) # Ensure consistent dtype
            target = target_sot.to(dtype=weight_dtype) # Ensure consistent dtype for loss

            # We don't need these from flux_utils for SOT as SOT's output is already "packed"
            # packed_latent_height, packed_latent_width = latents.shape[2] // 2, latents.shape[3] // 2
            # img_ids_flux_style = flux_utils.prepare_img_ids(bsz, packed_latent_height, packed_latent_width).to(device=accelerator.device)
            # Use img_ids_sot instead

        else: # Original FLUX/SD3 style
            noise = torch.randn_like(latents)
            noisy_model_input_std, timesteps_std, sigmas_std = flux_train_utils.get_noisy_model_input_and_timesteps(
                args, noise_scheduler, latents, noise, accelerator.device, weight_dtype
            )
            logger.debug(f"Std Noisy latents shape: {noisy_model_input_std.shape}, device: {noisy_model_input_std.device}")
            packed_noisy_model_input = flux_utils.pack_latents(noisy_model_input_std)
            # packed_latent_height, packed_latent_width = noisy_model_input_std.shape[2] // 2, noisy_model_input_std.shape[3] // 2
            img_ids_std = flux_utils.prepare_img_ids(bsz, latents.shape[2] // 2, latents.shape[3] // 2).to(device=accelerator.device)

            timesteps = timesteps_std
            img_ids = img_ids_std


        guidance_vec = torch.full((bsz,), float(args.guidance_scale), device=accelerator.device)
        l_pooled, t5_out, txt_ids, t5_attn_mask_from_conds = text_encoder_conds

        if l_pooled is not None: l_pooled = l_pooled.to(accelerator.device, dtype=weight_dtype)
        if t5_out is not None: t5_out = t5_out.to(accelerator.device, dtype=weight_dtype)
        if txt_ids is not None: txt_ids = txt_ids.to(accelerator.device)
        if t5_attn_mask_from_conds is not None: t5_attn_mask_from_conds = t5_attn_mask_from_conds.to(accelerator.device)

        if args.gradient_checkpointing:
            packed_noisy_model_input.requires_grad_(True)
            if l_pooled is not None and l_pooled.dtype.is_floating_point: l_pooled.requires_grad_(True)
            if t5_out is not None and t5_out.dtype.is_floating_point: t5_out.requires_grad_(True)
            if txt_ids is not None and txt_ids.dtype.is_floating_point: txt_ids.requires_grad_(True)
            img_ids.requires_grad_(True) # Positional IDs might need grad if they are part of learnable embeddings or transformed
            guidance_vec.requires_grad_(True)

        # Ensure txt_ids is 2D for concatenation with img_ids in Flux model if it uses them together for PE
        # This part of Flux.forward was: self.pe_embedder(torch.cat((txt_ids,img_ids),dim=1))
        # So txt_ids for PE needs to be (B, L_txt, 3) similar to img_ids.
        # However, the Flux model's txt_ids input is also used as T5's original token IDs for lookup.
        # The SOT logic provides image_pos_id for img_ids. The txt_ids from text_encoder_conds are (B, L_t5_seq).
        # Flux.forward needs to handle this: pe = self.pe_embedder(torch.cat((txt_pos_ids_3d, img_ids), dim=1))
        # Where txt_pos_ids_3d is derived inside Flux.forward from txt_ids (T5 original tokens).
        # So, the txt_ids passed here should be the T5 original tokens.

        if txt_ids is not None and txt_ids.ndim == 3 and txt_ids.shape[1] == 1 : # From some caching strategies
             logger.debug(f"Squeezing txt_ids (T5 tokens) from {txt_ids.shape} to 2D before UNet call.")
             txt_ids_for_unet = txt_ids.squeeze(1)
        else:
             txt_ids_for_unet = txt_ids # Should be (B, L_t5_seq)


        actual_t5_attn_mask_for_model = t5_attn_mask_from_conds if args.apply_t5_attn_mask else None

        def call_dit_func(img_in, img_ids_in, txt_in, txt_ids_in_t5_tokens, y_in, timesteps_in, guidance_in, txt_attention_mask_in):
            with torch.set_grad_enabled(is_train), accelerator.autocast():
                logger.debug("Calling unet.forward()...")
                # img_ids_in here should be the positional IDs (e.g., from SOT or flux_utils.prepare_img_ids)
                # txt_ids_in_t5_tokens here should be the T5 original token IDs
                model_pred_out = unet(
                    img=img_in, img_ids=img_ids_in, txt=txt_in, txt_ids=txt_ids_in_t5_tokens, y=y_in,
                    timesteps=timesteps_in / 1000, guidance=guidance_in, txt_attention_mask=txt_attention_mask_in,
                )
                logger.debug("unet.forward() returned.")
            return model_pred_out
        logger.debug(f"UNET CALL: packed_noisy_model_input shape {packed_noisy_model_input.shape}, dtype {packed_noisy_model_input.dtype}, NaN: {torch.isnan(packed_noisy_model_input).any()}")
        logger.debug(f"UNET CALL: img_ids shape {img_ids.shape}, dtype {img_ids.dtype}") # img_ids shouldn't be NaN if long
        logger.debug(f"UNET CALL: t5_out shape {t5_out.shape if t5_out is not None else 'None'}, NaN: {torch.isnan(t5_out).any() if t5_out is not None else 'N/A'}")
        logger.debug(f"UNET CALL: txt_ids_for_unet shape {txt_ids_for_unet.shape if txt_ids_for_unet is not None else 'None'}")
        logger.debug(f"UNET CALL: l_pooled shape {l_pooled.shape if l_pooled is not None else 'None'}, NaN: {torch.isnan(l_pooled).any() if l_pooled is not None else 'N/A'}")
        logger.debug(f"UNET CALL: timesteps mean {timesteps.mean().item() if timesteps is not None and timesteps.numel() > 0 else 'None'}, NaN: {torch.isnan(timesteps).any() if timesteps is not None else 'N/A'}")
        model_pred = call_dit_func(
            packed_noisy_model_input, img_ids, t5_out, txt_ids_for_unet,
            l_pooled, timesteps, guidance_vec, actual_t5_attn_mask_for_model,
        )
        logger.debug(f"UNET OUTPUT: model_pred shape {model_pred.shape}, dtype {model_pred.dtype}, NaN: {torch.isnan(model_pred).any()}, Inf: {torch.isinf(model_pred).any()}")
        logger.debug(f"UNET CALL: model_pred output shape {model_pred.shape}, dtype {model_pred.dtype}, NaN: {torch.isnan(model_pred).any()}, Inf: {torch.isinf(model_pred).any()}")

        if not self._is_chroma_sot_loss_active(args): # Original FLUX/SD3 path
            # model_pred is (B, L_flat_packed, C_out_packed)
            # Unpack to (B, C_vae, H_vae, W_vae)
            # packed_latent_height, packed_latent_width already defined if not SOT
            model_pred = flux_utils.unpack_latents(model_pred, latents.shape[2] // 2, latents.shape[3] // 2)
            model_pred, weighting = flux_train_utils.apply_model_prediction_type(args, model_pred, noisy_model_input_std, sigmas_std)
            target = noise - latents # Target for original FLUX logic
        else: # Chroma SOT path
            # model_pred is already (B, L_flat, C_flat) as UNet output matches input shape for SOT
            # target is already target_sot (B, L_flat, C_flat)
            # No further unpacking or prediction type application needed here. Weighting is None.
            pass


        if "custom_attributes" in batch: # This diff_output_preservation logic
            diff_output_pr_indices = [
                i for i, ca in enumerate(batch["custom_attributes"])
                if "diff_output_preservation" in ca and ca["diff_output_preservation"]
            ]

            if len(diff_output_pr_indices) > 0:
                network.set_multiplier(0.0)
                if hasattr(unet, 'prepare_block_swap_before_forward'): unet.prepare_block_swap_before_forward()
                with torch.no_grad():
                    l_pooled_prior = l_pooled[diff_output_pr_indices] if l_pooled is not None else None
                    t5_out_prior = t5_out[diff_output_pr_indices] if t5_out is not None else None # Added None check
                    txt_ids_prior = txt_ids_for_unet[diff_output_pr_indices] if txt_ids_for_unet is not None else None
                    t5_attn_mask_prior = actual_t5_attn_mask_for_model[diff_output_pr_indices] if actual_t5_attn_mask_for_model is not None else None

                    # Inputs for prior prediction need to match the main prediction path
                    packed_noisy_model_input_prior = packed_noisy_model_input[diff_output_pr_indices]
                    img_ids_prior = img_ids[diff_output_pr_indices]
                    timesteps_prior = timesteps[diff_output_pr_indices]
                    guidance_vec_prior = guidance_vec[diff_output_pr_indices] if guidance_vec is not None else None


                    model_pred_prior = call_dit_func(
                        packed_noisy_model_input_prior, img_ids_prior, t5_out_prior, txt_ids_prior,
                        l_pooled_prior, timesteps_prior, guidance_vec_prior, t5_attn_mask_prior,
                    )
                network.set_multiplier(1.0)

                if not self._is_chroma_sot_loss_active(args): # Original FLUX/SD3 path
                    model_pred_prior = flux_utils.unpack_latents(model_pred_prior, latents.shape[2] // 2, latents.shape[3] // 2)
                    model_pred_prior, _ = flux_train_utils.apply_model_prediction_type(
                        args, model_pred_prior,
                        noisy_model_input_std[diff_output_pr_indices], # Use std versions for consistency
                        sigmas_std[diff_output_pr_indices] if sigmas_std is not None else None,
                    )
                # For SOT, model_pred_prior is already in the correct "packed flat" format.
                # No unpacking or apply_model_prediction_type needed.

                target[diff_output_pr_indices] = model_pred_prior.to(target.dtype)

        return model_pred, target, timesteps, weighting


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    train_util.add_dit_training_arguments(parser)
    flux_train_utils.add_flux_train_arguments(parser)
    parser.add_argument(
        "--split_mode", action="store_true",
        help="[Deprecated] This option is deprecated. Please use `--blocks_to_swap` instead.",
    )
    return parser


if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()

    if "chroma" in args.pretrained_model_name_or_path.lower() and args.clip_l is not None:
        logger.warning("Chroma model detected. --clip_l argument will be ignored as Chroma does not use CLIP-L.")

    train_util.verify_command_line_training_args(args)
    args = train_util.read_config_from_file(args, parser)

    trainer = FluxNetworkTrainer()
    trainer.train(args)