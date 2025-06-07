import os
import glob
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from transformers import CLIPTokenizer, T5TokenizerFast
import inspect
from library import flux_utils, train_util
from library import flux_models
from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


CLIP_L_TOKENIZER_PATH_OR_IDENTIFIER = "openai/clip-vit-large-patch14"
T5_XXL_TOKENIZER_PATH_OR_IDENTIFIER = "google/flan-t5-xxl"

class TextEncodingStrategy:
    _strategy = None  # strategy instance: actual strategy class

    @classmethod
    def set_strategy(cls, strategy):
        logger.info(f"DEBUG_STRATEGY: {cls.__name__}.set_strategy called with: {type(strategy)} (id: {id(strategy)})")
        if cls._strategy is not None and cls._strategy is not strategy :
             logger.warning(f"DEBUG_STRATEGY: Overwriting {cls.__name__}._strategy. Old: {type(cls._strategy)}, New: {type(strategy)}")
        cls._strategy = strategy

    @classmethod
    def get_strategy(cls) -> Optional["TextEncodingStrategy"]:
        logger.info(f"DEBUG_STRATEGY: {cls.__name__}.get_strategy called. Returning: {type(cls._strategy)} (id: {id(cls._strategy) if cls._strategy else 'None'})")
        if cls._strategy is None:
            # Fallback to a default if no strategy has been set.
            # This is where SdTextEncodingStrategy might be instantiated if not set by a specific trainer.
            # To avoid circular dependency, we cannot directly instantiate SdTextEncodingStrategy here
            # but the logger indicates if we are returning None, which would trigger default in FineTuneDataset.
            logger.warning(f"DEBUG_STRATEGY: {cls.__name__}._strategy is None. Defaulting might occur in caller.")
        return cls._strategy
class FluxLatentsCachingStrategy(LatentsCachingStrategy):
    FLUX_LATENTS_NPZ_SUFFIX = "_flux.npz"
    FLUX_VAE_DOWNSCALE_FACTOR = 16 # Correct downscale factor for FLUX AE

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return self.FLUX_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return (
            os.path.splitext(absolute_path)[0]
            + f"_{image_size[0]:04d}x{image_size[1]:04d}"
            + self.FLUX_LATENTS_NPZ_SUFFIX
        )

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        return self._default_is_disk_cached_latents_expected(
            latents_stride=self.FLUX_VAE_DOWNSCALE_FACTOR, # Use correct downscale factor
            bucket_reso=bucket_reso,
            npz_path=npz_path,
            flip_aug=flip_aug,
            alpha_mask=alpha_mask,
            multi_resolution=True # Flux uses multi-resolution style naming
        )

    def load_latents_from_disk(
        self, npz_path: str, bucket_reso: Tuple[int, int]
    ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
        # bucket_reso is (W, H) for the image
        return self._default_load_latents_from_disk(
            latents_stride=self.FLUX_VAE_DOWNSCALE_FACTOR, # Pass the correct stride
            npz_path=npz_path,
            bucket_reso=bucket_reso
        )

    def cache_batch_latents(self,
                            vae: flux_models.AutoEncoder, 
                            image_infos: List[train_util.ImageInfo],
                            flip_aug: bool,
                            alpha_mask: bool,
                            random_crop: bool):
        encode_by_vae_func = lambda img_tensor: vae.encode(img_tensor).to("cpu")
        vae_device = vae.device
        vae_dtype = vae.dtype
        super()._default_cache_batch_latents(
            encode_by_vae_func,
            vae_device,
            vae_dtype,
            image_infos,
            flip_aug,
            alpha_mask,
            random_crop,
            multi_resolution=True 
        )
        if not train_util.HIGH_VRAM:
            train_util.clean_memory_on_device(vae.device)


class FluxTokenizeStrategy(TokenizeStrategy):
    def __init__(self, t5xxl_max_length: int = 512, tokenizer_cache_dir: Optional[str] = None, clip_l_tokenizer_path: Optional[str] = None) -> None:
        super().__init__()
        self.t5xxl_max_length = t5xxl_max_length

        self.clip_l_tokenizer = None
        self.clip_l_max_length = 77
        if clip_l_tokenizer_path is not None:
            logger.info(f"Loading CLIP-L Tokenizer from: {clip_l_tokenizer_path}")
            try:
                self.clip_l_tokenizer = CLIPTokenizer.from_pretrained(clip_l_tokenizer_path, cache_dir=tokenizer_cache_dir)
            except Exception as e:
                logger.error(f"Could not load CLIP-L tokenizer from {clip_l_tokenizer_path}. It will be disabled. Error: {e}")
                self.clip_l_tokenizer = None
        else:
            logger.info("CLIP-L tokenizer path not provided. CLIP-L tokenization will be skipped.")

        t5_path = T5_XXL_TOKENIZER_PATH_OR_IDENTIFIER
        logger.info(f"Loading T5XXL Tokenizer from: {t5_path}")
        self.t5xxl_tokenizer = T5TokenizerFast.from_pretrained(t5_path, cache_dir=tokenizer_cache_dir, legacy=False)


    def tokenize(self, text: Union[str, List[str]]) -> Dict[str, Optional[Dict[str, torch.Tensor]]]:
        if isinstance(text, str):
            text_batch = [text]
        else:
            text_batch = text

        clip_l_token_info = None
        if self.clip_l_tokenizer is not None:
            clip_l_tokens = self.clip_l_tokenizer(
                text_batch, max_length=self.clip_l_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            clip_l_token_info = {"input_ids": clip_l_tokens.input_ids, "attention_mask": clip_l_tokens.attention_mask}

        t5_tokens = self.t5xxl_tokenizer(
            text_batch, max_length=self.t5xxl_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        t5_token_info = {"input_ids": t5_tokens.input_ids, "attention_mask": t5_tokens.attention_mask}

        if isinstance(text, str):
            if clip_l_token_info:
                clip_l_token_info = {k: v[0] for k,v in clip_l_token_info.items()}
            t5_token_info = {k: v[0] for k,v in t5_token_info.items()}

        return {
            "clip_l": clip_l_token_info,
            "t5xxl": t5_token_info,
        }


class FluxTextEncodingStrategy(TextEncodingStrategy):
    def __init__(self, apply_t5_attn_mask: Optional[bool] = False) -> None:
        super().__init__()
        self.apply_t5_attn_mask_default = apply_t5_attn_mask

    def encode_tokens(
        self,
        tokenize_strategy: FluxTokenizeStrategy,
        text_encoders: List[Optional[torch.nn.Module]],
        tokens_and_masks_dict: Dict[str, Optional[Dict[str, torch.Tensor]]],
        apply_t5_attn_mask: Optional[bool] = None,  # <<< MODIFIED LINE: Added this parameter
    ) -> List[Optional[torch.Tensor]]:

        clip_l_model = text_encoders[0] if len(text_encoders) > 0 else None
        t5xxl_model = text_encoders[1] if len(text_encoders) > 1 else \
                      (text_encoders[0] if len(text_encoders) == 1 and clip_l_model is None else None)

        if t5xxl_model is None:
            raise ValueError("T5XXL model must be provided for FluxTextEncodingStrategy.")

        use_t5_mask_for_encoding = self.apply_t5_attn_mask_default if apply_t5_attn_mask is None else apply_t5_attn_mask

        l_pooled_output = None
        is_batched_tokens = isinstance(tokens_and_masks_dict["t5xxl"]["input_ids"], torch.Tensor) and \
                            tokens_and_masks_dict["t5xxl"]["input_ids"].ndim > 1

        if clip_l_model is not None and tokens_and_masks_dict.get("clip_l") is not None:
            clip_l_input_ids = tokens_and_masks_dict["clip_l"]["input_ids"]
            if not is_batched_tokens: clip_l_input_ids = clip_l_input_ids.unsqueeze(0)
            clip_l_input_ids = clip_l_input_ids.to(clip_l_model.device)
            clip_l_outputs = clip_l_model(input_ids=clip_l_input_ids, output_hidden_states=True)
            l_pooled_output = clip_l_outputs.pooler_output

        t5_token_info = tokens_and_masks_dict.get("t5xxl")
        if t5_token_info is None: raise ValueError("T5XXL tokens not found")

        t5_input_ids = t5_token_info["input_ids"]
        if not is_batched_tokens: t5_input_ids = t5_input_ids.unsqueeze(0)
        
        # Defensive check and correction for t5_input_ids shape if it's batched
        if is_batched_tokens and t5_input_ids.ndim == 3 and t5_input_ids.shape[1] == 1:
            logger.warning(f"T5 input_ids has unexpected shape {t5_input_ids.shape}. Squeezing out singleton dimension.")
            t5_input_ids = t5_input_ids.squeeze(1) # (B, 1, S) -> (B, S)
        
        t5_input_ids = t5_input_ids.to(t5xxl_model.device)

        t5_attention_mask_for_model_input = None
        t5_attention_mask_for_flux_output = None

        if use_t5_mask_for_encoding:
            t5_attention_mask_for_model_input = t5_token_info["attention_mask"]
            if not is_batched_tokens: t5_attention_mask_for_model_input = t5_attention_mask_for_model_input.unsqueeze(0)
            t5_attention_mask_for_model_input = t5_attention_mask_for_model_input.to(t5xxl_model.device)
            t5_attention_mask_for_flux_output = t5_attention_mask_for_model_input

        t5_outputs = t5xxl_model(input_ids=t5_input_ids, attention_mask=t5_attention_mask_for_model_input, output_hidden_states=True)
        t5_encoder_hidden_states = t5_outputs.last_hidden_state

        txt_ids_for_flux = t5_input_ids

        if not is_batched_tokens:
            if l_pooled_output is not None: l_pooled_output = l_pooled_output[0]
            t5_encoder_hidden_states = t5_encoder_hidden_states[0]
            txt_ids_for_flux = txt_ids_for_flux[0]
            if t5_attention_mask_for_flux_output is not None:
                t5_attention_mask_for_flux_output = t5_attention_mask_for_flux_output[0]

        return [l_pooled_output, t5_encoder_hidden_states, txt_ids_for_flux, t5_attention_mask_for_flux_output]


class FluxTextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    FLUX_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_flux_te.npz"

    def __init__(self, cache_to_disk: bool, batch_size: Optional[int], skip_disk_cache_validity_check: bool,
                 has_clip_l: bool, is_train_clip_l: bool, is_train_t5: bool, apply_t5_attn_mask: bool): # Added new args
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)
        self.has_clip_l = has_clip_l
        self.is_train_clip_l = is_train_clip_l
        self.is_train_t5 = is_train_t5
        self.apply_t5_attn_mask = apply_t5_attn_mask

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + self.FLUX_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str):
        if not self.cache_to_disk or not os.path.exists(npz_path): return False
        if self.skip_disk_cache_validity_check: return True
        try:
            with np.load(npz_path) as npz:
                if "t5_out" not in npz or "txt_ids" not in npz: return False
                # If t5_attn_mask was saved, apply_t5_attn_mask should also have been saved.
                # And their values should match the current strategy's expectation.
                t5_mask_present_in_npz = "t5_attn_mask" in npz
                apply_t5_mask_key_present_in_npz = "apply_t5_attn_mask" in npz

                if t5_mask_present_in_npz != apply_t5_mask_key_present_in_npz : return False # Should be saved together or not at all
                if apply_t5_mask_key_present_in_npz and npz["apply_t5_attn_mask"] != self.apply_t5_attn_mask: return False
        except Exception as e:
            logger.error(f"Error loading or validating cache file: {npz_path}, Error: {e}")
            return False
        return True

    def load_outputs_npz(self, npz_path: str) -> List[Optional[np.ndarray]]:
        with np.load(npz_path) as data:
            l_pooled = data["l_pooled"] if "l_pooled" in data else None
            t5_out = data["t5_out"]
            txt_ids = data["txt_ids"]
            t5_attn_mask = data["t5_attn_mask"] if "t5_attn_mask" in data else None
        return [l_pooled, t5_out, txt_ids, t5_attn_mask]

    def cache_batch_outputs(
        self, tokenize_strategy: FluxTokenizeStrategy, models: List[Optional[torch.nn.Module]],
        text_encoding_strategy: FluxTextEncodingStrategy, infos: List[train_util.ImageInfo]
    ):
        # if not self.warn_fp8_weights and models[1] is not None:
        #     if hasattr(models[1], 'dtype') and models[1].dtype == torch.float8_e4m3fn:
        #         logger.warning("T5 model is using fp8 weights for caching.")
        #     self.warn_fp8_weights = True

        captions = [info.caption for info in infos]
        tokens_and_masks_batch_dict = tokenize_strategy.tokenize(captions)
        logger.info(f"DEBUG: --- Inside cache_batch_outputs ---")
        logger.info(f"DEBUG: text_encoding_strategy object: {text_encoding_strategy}")
        logger.info(f"DEBUG: text_encoding_strategy type: {type(text_encoding_strategy)}")
        if hasattr(text_encoding_strategy, 'encode_tokens'):
            logger.info(f"DEBUG: text_encoding_strategy.encode_tokens method: {text_encoding_strategy.encode_tokens}")
            try:
                sig = inspect.signature(text_encoding_strategy.encode_tokens)
                logger.info(f"DEBUG: text_encoding_strategy.encode_tokens SIGNATURE: {sig}")
            except Exception as e:
                logger.error(f"DEBUG: Error inspecting signature of text_encoding_strategy.encode_tokens: {e}")
        else:
            logger.warning(f"DEBUG: text_encoding_strategy does NOT have attribute 'encode_tokens'")
        logger.info(f"DEBUG: self.apply_t5_attn_mask (from CachingStrategy): {self.apply_t5_attn_mask}")
        with torch.no_grad():
            encoded_batch_outputs = text_encoding_strategy.encode_tokens(
                tokenize_strategy, models, tokens_and_masks_batch_dict, apply_t5_attn_mask=self.apply_t5_attn_mask
            )

            for i, info in enumerate(infos):
                npz_dict_to_save = {}

                current_l_pooled = encoded_batch_outputs[0]
                if current_l_pooled is not None: # Only save if it exists
                    npz_dict_to_save["l_pooled"] = current_l_pooled[i].cpu().float().numpy()

                npz_dict_to_save["t5_out"] = encoded_batch_outputs[1][i].cpu().float().numpy()
                npz_dict_to_save["txt_ids"] = encoded_batch_outputs[2][i].cpu().numpy()

                current_t5_attn_mask = encoded_batch_outputs[3]
                if current_t5_attn_mask is not None: # Only save if it exists
                    npz_dict_to_save["t5_attn_mask"] = current_t5_attn_mask[i].cpu().numpy()
                    npz_dict_to_save["apply_t5_attn_mask"] = self.apply_t5_attn_mask # Save this only if mask is saved

                if self.cache_to_disk:
                    np.savez(info.text_encoder_outputs_npz, **npz_dict_to_save)
                else:
                    info.text_encoder_outputs = [
                        encoded_batch_outputs[0][i] if encoded_batch_outputs[0] is not None else None,
                        encoded_batch_outputs[1][i],
                        encoded_batch_outputs[2][i],
                        encoded_batch_outputs[3][i] if encoded_batch_outputs[3] is not None else None
                    ]