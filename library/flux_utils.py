import json
import os
from dataclasses import replace 
from typing import List, Optional, Tuple, Union, Dict, Any, TYPE_CHECKING 
from . import flux_models
import einops 
import torch
from accelerate import init_empty_weights 
from safetensors import safe_open
from safetensors.torch import load_file as load_safetensors_file 
from transformers import CLIPConfig, CLIPTextModel, T5Config, T5EncoderModel 

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

from library import flux_models 
from library.utils import load_safetensors 

if TYPE_CHECKING:
    from library.flux_models import Flux 
DEVICE_CPU = "cpu"
MODEL_VERSION_FLUX_V1 = "flux1" 

MODEL_TYPE_FLUX_DEV = "flux_dev"
MODEL_TYPE_FLUX_SCHNELL = "flux_schnell_official" 
MODEL_TYPE_CHROMA = "chroma"
MODEL_TYPE_UNKNOWN = "unknown"

DOUBLE_BLOCKS_PREFIX = "double_blocks."
SINGLE_BLOCKS_PREFIX = "single_blocks."
CHROMA_DISTILLED_GUIDANCE_KEY = "distilled_guidance_layer.in_proj.weight"
SCHNELL_DEV_DOUBLE_MODULATION_KEY_SAMPLE = "double_blocks.0.img_mod.lin.weight"
SCHNELL_DEV_SINGLE_MODULATION_KEY_SAMPLE = "single_blocks.0.modulation.lin.weight"
FLUX_DEV_GUIDANCE_IN_KEY = "guidance_in.in_layer.weight" 

def analyze_checkpoint_state(ckpt_path: str) -> Tuple[str, int, int, Dict[str, Any]]:
    logger.info(f"Analyzing checkpoint state: {ckpt_path}")

    if os.path.isdir(ckpt_path):
        raise ValueError("Directory paths (Diffusers format) not supported. Provide .safetensors file.")

    try:
        with safe_open(ckpt_path, framework="pt", device="cpu") as f:
            keys = set(f.keys()) 
    except Exception as e:
        logger.error(f"Could not open/read keys from safetensors file {ckpt_path}: {e}")
        return MODEL_TYPE_UNKNOWN, 0, 0, {}

    has_distilled_guidance_layer = CHROMA_DISTILLED_GUIDANCE_KEY in keys
    has_internal_modulation = SCHNELL_DEV_DOUBLE_MODULATION_KEY_SAMPLE in keys or \
                              SCHNELL_DEV_SINGLE_MODULATION_KEY_SAMPLE in keys
    has_guidance_embed_input = FLUX_DEV_GUIDANCE_IN_KEY in keys
    has_time_embed = "time_in.in_layer.weight" in keys
    has_vector_embed = "vector_in.in_layer.weight" in keys
    
    metadata = {
        "has_distilled_guidance_layer": has_distilled_guidance_layer,
        "has_internal_modulation": has_internal_modulation,
        "has_guidance_embed_input": has_guidance_embed_input,
        "has_time_embed": has_time_embed,
        "has_vector_embed": has_vector_embed,
    }

    double_block_indices = set()
    single_block_indices = set()

    for key_item in keys: 
        if key_item.startswith(DOUBLE_BLOCKS_PREFIX):
            try:
                idx_str = key_item.split('.')[1]
                if idx_str.isdigit():
                    double_block_indices.add(int(idx_str))
            except IndexError: pass
        elif key_item.startswith(SINGLE_BLOCKS_PREFIX):
            try:
                idx_str = key_item.split('.')[1]
                if idx_str.isdigit():
                    single_block_indices.add(int(idx_str))
            except IndexError: pass
            
    max_double_block_index = max(double_block_indices, default=-1)
    num_double_blocks = max_double_block_index + 1 if max_double_block_index != -1 else 0

    max_single_block_index = max(single_block_indices, default=-1)
    num_single_blocks = max_single_block_index + 1 if max_single_block_index != -1 else 0

    model_type_str = MODEL_TYPE_UNKNOWN

    if (metadata["has_distilled_guidance_layer"] and
        not metadata["has_internal_modulation"] and
        not metadata["has_time_embed"] and
        not metadata["has_vector_embed"]):
        if metadata["has_guidance_embed_input"]:
            model_type_str = MODEL_TYPE_CHROMA
        else:
            if num_double_blocks > 0 or num_single_blocks > 0:
                model_type_str = MODEL_TYPE_CHROMA
                logger.warning(
                    f"Identified as CHROMA for {ckpt_path} based on core features, "
                    f"but 'guidance_in.in_layer.weight' (has_guidance_embed_input) is MISSING. "
                    f"Proceeding as Chroma. Metadata: {metadata}"
                )
    
    elif (model_type_str == MODEL_TYPE_UNKNOWN and 
          metadata["has_internal_modulation"] and
          metadata["has_guidance_embed_input"] and
          metadata["has_time_embed"] and
          metadata["has_vector_embed"] and
          not metadata["has_distilled_guidance_layer"]):
        model_type_str = MODEL_TYPE_FLUX_DEV

    elif (model_type_str == MODEL_TYPE_UNKNOWN and 
          metadata["has_internal_modulation"] and
          metadata["has_time_embed"] and
          metadata["has_vector_embed"] and
          not metadata["has_distilled_guidance_layer"] and
          not metadata["has_guidance_embed_input"]):
        model_type_str = MODEL_TYPE_FLUX_SCHNELL
    
    if model_type_str == MODEL_TYPE_UNKNOWN:
        if num_double_blocks > 0 or num_single_blocks > 0:
             logger.warning(f"Could not definitively identify model type for {ckpt_path}. "
                           f"Final Metadata: {metadata}. Num double: {num_double_blocks}, Num single: {num_single_blocks}. Defaulting to UNKNOWN.")
        else: 
            logger.error(f"Could not identify model type for {ckpt_path}. No known Flux block structures or "
                         "distinguishing keys found. Final Metadata: {metadata}")

    logger.info(
        f"Analyzed {ckpt_path} - Determined Type: {model_type_str}, "
        f"Double Blocks: {num_double_blocks}, Single Blocks: {num_single_blocks}, "
        f"Final Metadata: {metadata}" 
    )
    return model_type_str, num_double_blocks, num_single_blocks, metadata


def load_flow_model(path: str, dtype: torch.dtype = torch.bfloat16, device: str = "cuda", 
                    disable_mmap: bool = False, flux_variant_type: Optional[str] = None):
    
    model_type_str = "chroma" 
    logger.info(f"Forcing model type to '{model_type_str}' for loading.")

    if model_type_str == "chroma":
        logger.info("Loading ChromaModel structure.")
        # Assuming flux_models has get_chroma_model_params and ChromaModel
        params_obj = flux_models.get_chroma_model_params() 
        model = flux_models.Flux(params_obj)
    else:
        # This part is now less relevant if we are only supporting Chroma loading here.
        logger.error(f"Attempting to load non-Chroma model type '{model_type_str}' through a Chroma-focused path. This may not be fully supported by current simplifications.")
        # Fallback to original logic if truly needed for other types, but for now, focus on Chroma
        model_type_str_analyzed, num_double, num_single, analysis_metadata = analyze_checkpoint_state(path)
        if model_type_str_analyzed not in flux_models.model_configs: 
            raise ValueError(f"Unknown model variant: {model_type_str_analyzed}")
        
        # This assumes model_configs and the generic Flux class are still defined for non-Chroma variants
        params_dto_generic = flux_models.model_configs[model_type_str_analyzed].params 
        if isinstance(params_dto_generic, dict): 
            params_dto_generic = flux_models.flux1_dev_params(num_double, num_single) 
        model = flux_models.Flux(params_dto_generic) 
        model_type_str = model_type_str_analyzed


    logger.info(f"Building model variant '{model_type_str}'.")
    
    # Load state dict
    logger.info(f"Loading state dict from {path} to model on device '{DEVICE_CPU}' initially, target dtype for operations: '{str(dtype)}'")
    
    # Load to CPU first, then cast and move. Dtype in load_safetensors here is for final tensor dtype on target device.
    # We want to load the raw data first.
    sd = load_safetensors(path, DEVICE_CPU, disable_mmap=disable_mmap, dtype_load_as_is=True) # Add a flag to load as is

    model_keys = model.state_dict().keys()
    filtered_sd = {}
    unexpected_keys_in_ckpt_for_current_model = []

    for k, v in sd.items():
        if k in model_keys:
            filtered_sd[k] = v
        else:
            unexpected_keys_in_ckpt_for_current_model.append(k)
            
    if unexpected_keys_in_ckpt_for_current_model:
        logger.warning(
            f"Checkpoint for '{model_type_str}' contained {len(unexpected_keys_in_ckpt_for_current_model)} unexpected keys "
            f"not in the current model structure: {unexpected_keys_in_ckpt_for_current_model[:10]}..."
        )

    # Load with strict=False to allow missing LoRA keys etc., and to handle pruned models gracefully.
    info = model.load_state_dict(filtered_sd, strict=False) 
    
    if info.missing_keys:
        # Filter out known acceptable missing keys for pruned models if necessary
        # For Chroma, we expect final_layer.adaLN_modulation to be missing.
        # Other missing keys might indicate a problem.
        truly_missing_keys = [k for k in info.missing_keys if not k.startswith("final_layer.adaLN_modulation.")]
        if truly_missing_keys:
            logger.error(
                f"CRITICAL Missing keys when loading state_dict for '{model_type_str}': {truly_missing_keys}. "
                f"This indicates a potential mismatch between the checkpoint and the model definition."
            )
            # Optionally raise an error here if any truly critical keys are missing
            # raise RuntimeError(f"Critical missing keys for {model_type_str}: {truly_missing_keys}")
        elif info.missing_keys: # Only adaLN keys were missing
            logger.info(f"Known missing keys (expected for pruned Chroma, e.g., final_layer.adaLN): {info.missing_keys}")

    logger.info(f"State_dict for '{model_type_str}' processed. Missing keys (after filtering known acceptable): {len(truly_missing_keys) if 'truly_missing_keys' in locals() else len(info.missing_keys)}. Model will be moved to {device}/{dtype}.")
    
    # Now move and cast the model with loaded (or initialized) weights
    model.to(device=torch.device(device), dtype=dtype)
    
    return model_type_str, model


def load_ae(
    ckpt_path: str, dtype: torch.dtype, device: Union[str, torch.device], disable_mmap: bool = False
) -> flux_models.AutoEncoder: 
    logger.info("Building AutoEncoder")
    try:
        # ... (AE params loading logic remains the same) ...
        if "schnell" in flux_models.model_configs: 
            ae_params = flux_models.model_configs["schnell"].ae_params
        elif "dev" in flux_models.model_configs: 
            ae_params = flux_models.model_configs["dev"].ae_params
        else: 
            ae_params = flux_models._original_configs_for_ae["dev"].ae_params
    except (KeyError, AttributeError) as e:
        logger.error(f"Could not retrieve AE params: {e}")
        raise

    with init_empty_weights(): 
        ae = flux_models.AutoEncoder(ae_params)
    
    ae.to_empty(device=device) # Materialize on target device
    ae.to(dtype=dtype)         # Set dtype

    logger.info(f"Loading AE state dict from {ckpt_path}")
    sd = load_safetensors(ckpt_path, device="cpu", disable_mmap=disable_mmap, dtype=None) 
    
    info = ae.load_state_dict(sd, strict=False) # assign=True not needed as model is not meta
    logger.info(f"Loaded AE: {info}")
    
    # ae.to(device=device, dtype=dtype) # Already done
    return ae


def load_controlnet(
    ckpt_path: Optional[str], # ControlNet might not always be loaded
    main_model_params_dto: flux_models.FluxParams, # Pass params from the main UNet
    dtype: torch.dtype,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    controlnet_depth_mult: float = 1.0 # Multiplier for controlnet depth vs main model
) -> Optional[flux_models.ControlNetFlux]: # Return Optional since it might not be loaded
    
    if not ckpt_path:
        logger.info("No ControlNet checkpoint path provided. Skipping ControlNet loading.")
        return None

    logger.info("Building ControlNetFlux")
    
    # ControlNet often shares many architectural parameters with the main UNet
    # but might have its own specific tweaks or depth.
    # We pass the main model's FluxParams and a depth multiplier.
    controlnet = flux_models.ControlNetFlux(main_model_params_dto, controlnet_depth_mult=controlnet_depth_mult)
    
    # Load weights if path is provided
    logger.info(f"Loading ControlNet state dict from {ckpt_path}")
    # Use DEVICE_CPU for initial loading if defined, else "cpu"
    cpu_device = getattr(flux_models, 'DEVICE_CPU', "cpu") # Check if DEVICE_CPU is in flux_models
    sd = load_safetensors(ckpt_path, device=cpu_device, disable_mmap=disable_mmap, dtype_load_as_is=True)
    
    # ControlNet might have a different prefix if it's a separate model.
    # If weights are directly compatible (e.g. a fine-tuned UNet as ControlNet):
    info = controlnet.load_state_dict(sd, strict=False) 
    logger.info(f"Loaded ControlNet: Missing keys: {info.missing_keys}, Unexpected keys: {info.unexpected_keys}")
    
    controlnet.to(device=torch.device(device), dtype=dtype)
    controlnet.eval() # ControlNets are usually in eval mode during training of other components
    return controlnet


def load_clip_l(
    ckpt_path: Optional[str],
    dtype: torch.dtype,
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
) -> CLIPTextModel:
    logger.info("Building CLIP-L")
    try:
        # ... (config loading logic remains the same) ...
        full_config = CLIPConfig.from_pretrained("openai/clip-vit-large-patch14")
        text_config = full_config.text_config
        if not hasattr(text_config, 'projection_dim') and hasattr(full_config, 'projection_dim'):
            text_config.projection_dim = full_config.projection_dim
    except Exception as e:
        logger.error(f"Failed to load CLIPConfig from pretrained: {e}. Falling back to manual.")
        text_config_dict = {
            "hidden_size": 768, "intermediate_size": 3072, "num_attention_heads": 12,
            "num_hidden_layers": 12, "vocab_size": 49408, "max_position_embeddings": 77,
            "hidden_act": "quick_gelu", "layer_norm_eps": 1e-5, "attention_dropout": 0.0,
            "initializer_range": 0.02, "initializer_factor": 1.0,
            "pad_token_id": 0, "bos_token_id": 49406, "eos_token_id": 49407, 
            "model_type": "clip_text_model", "projection_dim": 768
        }
        from transformers.models.clip.configuration_clip import CLIPTextConfig
        text_config = CLIPTextConfig(**text_config_dict)


    with init_empty_weights():
        clip = CLIPTextModel(config=text_config)

    clip.to_empty(device=device)
    clip.to(dtype=dtype)

    if state_dict is None:
        if ckpt_path is None:
            raise ValueError("ckpt_path cannot be None if state_dict is not provided for load_clip_l")
        logger.info(f"Loading CLIP-L state dict from {ckpt_path}")
        sd = load_safetensors(ckpt_path, device="cpu", disable_mmap=disable_mmap, dtype=None) 
    else:
        sd = state_dict
            
    info = clip.load_state_dict(sd, strict=False)
    logger.info(f"Loaded CLIP-L: {info}")

    # clip.to(device=device, dtype=dtype) # Already done
    return clip


def load_t5xxl(
    ckpt_path: str,
    dtype: Optional[torch.dtype], 
    device: Union[str, torch.device],
    disable_mmap: bool = False,
    state_dict: Optional[dict] = None,
) -> T5EncoderModel:
    # ... (T5_CONFIG_JSON remains the same) ...
    T5_CONFIG_JSON = """
{
  "architectures": [
    "T5EncoderModel"
  ],
  "classifier_dropout": 0.0,
  "d_ff": 10240,
  "d_kv": 64,
  "d_model": 4096,
  "decoder_start_token_id": 0,
  "dense_act_fn": "gelu_new",
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "initializer_factor": 1.0,
  "is_encoder_decoder": true,
  "is_gated_act": true,
  "layer_norm_epsilon": 1e-06,
  "model_type": "t5",
  "num_decoder_layers": 24,
  "num_heads": 64,
  "num_layers": 24,
  "output_past": true,
  "pad_token_id": 0,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.41.2",
  "use_cache": true,
  "vocab_size": 32128
}
"""
    config_dict = json.loads(T5_CONFIG_JSON)
    
    config = T5Config(**config_dict)
    with init_empty_weights():
        t5xxl = T5EncoderModel(config=config)

    target_load_dtype = dtype if dtype is not None else torch.float32 # Default to float32 for materialization if not specified
    t5xxl.to_empty(device=device)
    t5xxl.to(dtype=target_load_dtype)


    if state_dict is None:
        logger.info(f"Loading T5xxl state dict from {ckpt_path}")
        sd = load_safetensors(ckpt_path, device="cpu", disable_mmap=disable_mmap, dtype=None) 
    else:
        sd = state_dict
            
    info = t5xxl.load_state_dict(sd, strict=False)
    logger.info(f"Loaded T5xxl: {info}")

    # t5xxl.to(device=device, dtype=final_dtype) # Already on device and target_load_dtype
    # If dtype was None initially, model is now float32. If a specific dtype was passed, it's that dtype.
    return t5xxl


def get_t5xxl_actual_dtype(t5xxl: T5EncoderModel) -> torch.dtype:
    return t5xxl.encoder.block[0].layer[0].SelfAttention.q.weight.dtype


def prepare_img_ids(batch_size: int, packed_latent_height: int, packed_latent_width: int):
    img_ids = torch.zeros(packed_latent_height, packed_latent_width, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(packed_latent_height)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(packed_latent_width)[None, :]
    img_ids = einops.repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
    return img_ids


def unpack_latents(x: torch.Tensor, packed_latent_height: int, packed_latent_width: int) -> torch.Tensor:
    x = einops.rearrange(x, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=packed_latent_height, w=packed_latent_width, ph=2, pw=2)
    return x


def pack_latents(x: torch.Tensor) -> torch.Tensor:
    x = einops.rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    return x


# region Diffusers

NUM_DOUBLE_BLOCKS = 19
NUM_SINGLE_BLOCKS = 38

BFL_TO_DIFFUSERS_MAP = {
    "time_in.in_layer.weight": ["time_text_embed.timestep_embedder.linear_1.weight"],
    "time_in.in_layer.bias": ["time_text_embed.timestep_embedder.linear_1.bias"],
    "time_in.out_layer.weight": ["time_text_embed.timestep_embedder.linear_2.weight"],
    "time_in.out_layer.bias": ["time_text_embed.timestep_embedder.linear_2.bias"],
    "vector_in.in_layer.weight": ["time_text_embed.text_embedder.linear_1.weight"],
    "vector_in.in_layer.bias": ["time_text_embed.text_embedder.linear_1.bias"],
    "vector_in.out_layer.weight": ["time_text_embed.text_embedder.linear_2.weight"],
    "vector_in.out_layer.bias": ["time_text_embed.text_embedder.linear_2.bias"],
    "guidance_in.in_layer.weight": ["time_text_embed.guidance_embedder.linear_1.weight"],
    "guidance_in.in_layer.bias": ["time_text_embed.guidance_embedder.linear_1.bias"],
    "guidance_in.out_layer.weight": ["time_text_embed.guidance_embedder.linear_2.weight"],
    "guidance_in.out_layer.bias": ["time_text_embed.guidance_embedder.linear_2.bias"],
    "txt_in.weight": ["context_embedder.weight"],
    "txt_in.bias": ["context_embedder.bias"],
    "img_in.weight": ["x_embedder.weight"],
    "img_in.bias": ["x_embedder.bias"],
    "double_blocks.().img_mod.lin.weight": ["norm1.linear.weight"],
    "double_blocks.().img_mod.lin.bias": ["norm1.linear.bias"],
    "double_blocks.().txt_mod.lin.weight": ["norm1_context.linear.weight"],
    "double_blocks.().txt_mod.lin.bias": ["norm1_context.linear.bias"],
    "double_blocks.().img_attn.qkv.weight": ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight"],
    "double_blocks.().img_attn.qkv.bias": ["attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias"],
    "double_blocks.().txt_attn.qkv.weight": ["attn.add_q_proj.weight", "attn.add_k_proj.weight", "attn.add_v_proj.weight"],
    "double_blocks.().txt_attn.qkv.bias": ["attn.add_q_proj.bias", "attn.add_k_proj.bias", "attn.add_v_proj.bias"],
    "double_blocks.().img_attn.norm.query_norm.scale": ["attn.norm_q.weight"],
    "double_blocks.().img_attn.norm.key_norm.scale": ["attn.norm_k.weight"],
    "double_blocks.().txt_attn.norm.query_norm.scale": ["attn.norm_added_q.weight"],
    "double_blocks.().txt_attn.norm.key_norm.scale": ["attn.norm_added_k.weight"],
    "double_blocks.().img_mlp.0.weight": ["ff.net.0.proj.weight"],
    "double_blocks.().img_mlp.0.bias": ["ff.net.0.proj.bias"],
    "double_blocks.().img_mlp.2.weight": ["ff.net.2.weight"],
    "double_blocks.().img_mlp.2.bias": ["ff.net.2.bias"],
    "double_blocks.().txt_mlp.0.weight": ["ff_context.net.0.proj.weight"],
    "double_blocks.().txt_mlp.0.bias": ["ff_context.net.0.proj.bias"],
    "double_blocks.().txt_mlp.2.weight": ["ff_context.net.2.weight"],
    "double_blocks.().txt_mlp.2.bias": ["ff_context.net.2.bias"],
    "double_blocks.().img_attn.proj.weight": ["attn.to_out.0.weight"],
    "double_blocks.().img_attn.proj.bias": ["attn.to_out.0.bias"],
    "double_blocks.().txt_attn.proj.weight": ["attn.to_add_out.weight"],
    "double_blocks.().txt_attn.proj.bias": ["attn.to_add_out.bias"],
    "single_blocks.().modulation.lin.weight": ["norm.linear.weight"],
    "single_blocks.().modulation.lin.bias": ["norm.linear.bias"],
    "single_blocks.().linear1.weight": ["attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight", "proj_mlp.weight"],
    "single_blocks.().linear1.bias": ["attn.to_q.bias", "attn.to_k.bias", "attn.to_v.bias", "proj_mlp.bias"],
    "single_blocks.().linear2.weight": ["proj_out.weight"], 
    "single_blocks.().norm.query_norm.scale": ["attn.norm_q.weight"],
    "single_blocks.().norm.key_norm.scale": ["attn.norm_k.weight"],
    "single_blocks.().linear2.bias": ["proj_out.bias"],
    "final_layer.linear.weight": ["proj_out.weight"],
    "final_layer.linear.bias": ["proj_out.bias"],
    "final_layer.adaLN_modulation.1.weight": ["norm_out.linear.weight"],
    "final_layer.adaLN_modulation.1.bias": ["norm_out.linear.bias"],
}


def make_diffusers_to_bfl_map(num_double_blocks: int, num_single_blocks: int) -> dict[str, tuple[int, str]]:
    diffusers_to_bfl_map = {} 
    for b in range(num_double_blocks):
        for key, weights in BFL_TO_DIFFUSERS_MAP.items():
            if key.startswith("double_blocks."):
                block_prefix = f"transformer_blocks.{b}."
                for i, weight in enumerate(weights):
                    diffusers_to_bfl_map[f"{block_prefix}{weight}"] = (i, key.replace("()", f"{b}"))
    for b in range(num_single_blocks):
        for key, weights in BFL_TO_DIFFUSERS_MAP.items():
            if key.startswith("single_blocks."):
                block_prefix = f"single_transformer_blocks.{b}."
                for i, weight in enumerate(weights):
                    diffusers_to_bfl_map[f"{block_prefix}{weight}"] = (i, key.replace("()", f"{b}"))
    for key, weights in BFL_TO_DIFFUSERS_MAP.items():
        if not (key.startswith("double_blocks.") or key.startswith("single_blocks.")):
            for i, weight in enumerate(weights):
                diffusers_to_bfl_map[weight] = (i, key)
    return diffusers_to_bfl_map


def convert_diffusers_sd_to_bfl(
    diffusers_sd: dict[str, torch.Tensor], num_double_blocks: int = NUM_DOUBLE_BLOCKS, num_single_blocks: int = NUM_SINGLE_BLOCKS
) -> dict[str, torch.Tensor]:
    diffusers_to_bfl_map = make_diffusers_to_bfl_map(num_double_blocks, num_single_blocks)

    flux_sd = {}
    for diffusers_key, tensor in diffusers_sd.items():
        if diffusers_key in diffusers_to_bfl_map:
            index, bfl_key = diffusers_to_bfl_map[diffusers_key]
            if bfl_key not in flux_sd:
                flux_sd[bfl_key] = []
            flux_sd[bfl_key].append((index, tensor))
        else:
            logger.warning(f"Skipping key not found in diffusers_to_bfl_map: {diffusers_key}")

    for key, values in list(flux_sd.items()): 
        if len(values) == 1:
            flux_sd[key] = values[0][1]
        else:
            try:
                flux_sd[key] = torch.cat([value[1] for value in sorted(values, key=lambda x: x[0])])
            except Exception as e:
                logger.error(f"Error concatenating tensors for key {key}: {e}. Values: {values}")
                del flux_sd[key]

    def swap_scale_shift(weight):
        shift, scale = weight.chunk(2, dim=0)
        new_weight = torch.cat([scale, shift], dim=0)
        return new_weight

    if "final_layer.adaLN_modulation.1.weight" in flux_sd:
        flux_sd["final_layer.adaLN_modulation.1.weight"] = swap_scale_shift(flux_sd["final_layer.adaLN_modulation.1.weight"])
    if "final_layer.adaLN_modulation.1.bias" in flux_sd:
        flux_sd["final_layer.adaLN_modulation.1.bias"] = swap_scale_shift(flux_sd["final_layer.adaLN_modulation.1.bias"])

    return flux_sd