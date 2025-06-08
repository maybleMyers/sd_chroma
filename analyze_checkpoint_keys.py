# analyze_checkpoint_keys.py
import torch
import os
from safetensors.torch import load_file as load_safetensors_file # For loading

# Assuming your project structure allows these imports
# Adjust paths if necessary
try:
    from library import flux_models
    from library.flux_models import flux_chroma_params, Flux, Approximator, FeatureFusionDistiller
except ImportError as e:
    print(f"Error importing library modules: {e}")
    print("Please ensure this script is run from a location where 'library' is importable,")
    print("or adjust aimport paths.")
    exit()

# --- Configuration ---
CHECKPOINT_PATH = "/home/mayble/diffusion/chromaforge/models/Stable-diffusion/chroma-unlocked-v32.safetensors" # หรือ v34
# CHECKPOINT_PATH = "/path/to/your/chroma-unlocked-v34-detail-calibrated.safetensors"
DEVICE = "cpu"
DTYPE = torch.bfloat16 # The dtype the model is likely stored/used in

def analyze_keys():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint file not found at {CHECKPOINT_PATH}")
        return

    print(f"Analyzing checkpoint: {CHECKPOINT_PATH}\n")

    # 1. Load the state dictionary from the checkpoint
    try:
        print(f"Loading state_dict from checkpoint...")
        checkpoint_sd = load_safetensors_file(CHECKPOINT_PATH, device=DEVICE)
        print(f"Successfully loaded state_dict with {len(checkpoint_sd)} keys.\n")
    except Exception as e:
        print(f"ERROR: Could not load checkpoint state_dict: {e}")
        return

    checkpoint_keys = set(checkpoint_sd.keys())

    # 2. Instantiate your Flux model configured for Chroma
    print(f"Instantiating flux_models.Flux with flux_chroma_params()...")
    try:
        chroma_flux_params = flux_chroma_params() # Uses guidance_embed=False now
        # Ensure the __init__ of Flux correctly sets up modules based on chroma_flux_params
        # For example, if a module should be an Approximator vs FeatureFusionDistiller based on Chroma's needs.

        # Based on our previous discussion, for Chroma with use_modulation=False:
        # - self.time_in should be None
        # - self.vector_in should be None
        # - self.guidance_in should be nn.Identity (since guidance_embed=False in params)
        # - self.distilled_guidance_layer is an Approximator (this is the main modulator)
        # - self.img_p_enhancer (if you added it) would be a FeatureFusionDistiller or None

        # Let's reflect the structure derived from the official Chroma script and diagram more closely
        # We will create a temporary modified FluxParams and Flux class for this analysis
        # to match the official Chroma architecture more closely.

        temp_chroma_params_for_analysis = flux_models.FluxParams(
            in_channels=64, vec_in_dim=768, context_in_dim=4096, hidden_size=3072,
            mlp_ratio=4.0, num_heads=24, depth=19, depth_single_blocks=38,
            axes_dim=[16, 56, 56], theta=10_000, qkv_bias=True,
            guidance_embed=True, # As per official chroma_params, but it is likely pruned.
                                 # The important part is that an Approximator is expected for modulation.
            use_modulation=False, # This triggers the Approximator path
            use_distilled_guidance_layer=True, # This usually means the main Approximator in official Chroma
            distilled_guidance_dim=5120, # Used by Approximator hidden_dim
            use_time_embed=False,
            use_vector_embed=False,
            double_block_has_main_norms=False,
            approximator_config=flux_models.ApproximatorParams( # This configures the main modulator
                in_dim=64, out_dim_per_mod_vector=3072,
                hidden_dim=5120, n_layers=4, mod_index_length=344
            )
        )
        
        # For analysis, let's ensure the Flux model's main modulator (Approximator)
        # is named 'distilled_guidance_layer' to match official Chroma structure.
        # We'll create a temporary Flux model structure for this.
        # This is a bit of a hack for analysis; the proper fix is in your flux_models.py.

        class TempFluxForAnalysis(flux_models.Flux):
            def __init__(self, params: flux_models.FluxParams):
                super().__init__(params) # Calls original Flux init
                
                # If Chroma, the main modulation Approximator is likely named 'distilled_guidance_layer'
                # and there's no separate 'modulation_approximator'.
                # The original Flux init for Chroma (use_modulation=False, approximator_config=True)
                # creates self.modulation_approximator.
                # Let's assume the checkpoint keys are for 'distilled_guidance_layer' as the Approximator.
                # We need to ensure our model has an attribute with this name that IS an Approximator.
                if not params.use_modulation and params.approximator_config:
                    # In our Flux, this creates self.modulation_approximator
                    # If the checkpoint uses 'distilled_guidance_layer' for these weights:
                    if hasattr(self, 'modulation_approximator') and self.modulation_approximator is not None:
                        print("INFO: For analysis, renaming self.modulation_approximator to self.distilled_guidance_layer to match potential checkpoint keys for the main modulator.")
                        self.distilled_guidance_layer = self.modulation_approximator
                        del self.modulation_approximator # Avoid duplicate if names clash

                        # The original self.distilled_guidance_layer (FeatureFusionDistiller) might not exist
                        # in the pruned Chroma model or might be a different component.
                        # For now, we focus on the main modulator.
                        if hasattr(self, 'img_p_enhancer'): # If we defined it
                             print("INFO: self.img_p_enhancer (FeatureFusionDistiller) exists.")
                        
                    else:
                         print("WARNING: self.modulation_approximator was not created as expected for Chroma config.")


        model_instance = TempFluxForAnalysis(temp_chroma_params_for_analysis)
        # model_instance = flux_models.Flux(chroma_flux_params) # Original instantiation
        
        model_instance.to(dtype=DTYPE, device=DEVICE) # Match checkpoint dtype
        model_keys = set(model_instance.state_dict().keys())
        print(f"Instantiated model has {len(model_keys)} keys.\n")

    except Exception as e:
        print(f"ERROR: Could not instantiate Flux model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Comparisons
    print("--- Key Analysis ---")

    missing_from_checkpoint = model_keys - checkpoint_keys
    if missing_from_checkpoint:
        print(f"\nWARNING: {len(missing_from_checkpoint)} keys in an ideal Chroma model structure are MISSING from the checkpoint:")
        for i, k in enumerate(sorted(list(missing_from_checkpoint))):
            print(f"  {i+1}. {k}")
    else:
        print("\nINFO: All keys expected by the model structure are present in the checkpoint.")

    extra_in_checkpoint = checkpoint_keys - model_keys
    if extra_in_checkpoint:
        print(f"\nWARNING: {len(extra_in_checkpoint)} keys in the checkpoint are UNEXPECTED by the model structure (extra keys):")
        for i, k in enumerate(sorted(list(extra_in_checkpoint))):
            print(f"  {i+1}. {k}")
    else:
        print("\nINFO: No extra keys found in the checkpoint.")

    common_keys = model_keys.intersection(checkpoint_keys)
    print(f"\nINFO: {len(common_keys)} keys are common to both model and checkpoint.")

    # 4. Specifically check for modulation-related prefixes
    print("\n--- Modulation Key Analysis ---")
    
    print("\nCheckpoint keys starting with 'distilled_guidance_layer.':")
    found_distilled_guidance = False
    for k in sorted(list(checkpoint_keys)):
        if k.startswith("distilled_guidance_layer."):
            print(f"  CKPT: {k}")
            found_distilled_guidance = True
    if not found_distilled_guidance:
        print("  None found.")

    print("\nModel keys starting with 'distilled_guidance_layer.' (after potential rename for analysis):")
    # This checks the attribute name in the TempFluxForAnalysis instance
    if hasattr(model_instance, 'distilled_guidance_layer') and model_instance.distilled_guidance_layer is not None:
        print(f"  Model has 'distilled_guidance_layer' attribute. Type: {type(model_instance.distilled_guidance_layer)}")
        for name, _ in model_instance.distilled_guidance_layer.named_parameters():
            print(f"  MODEL: distilled_guidance_layer.{name}")
        if not list(model_instance.distilled_guidance_layer.named_parameters()):
            print("  Model's 'distilled_guidance_layer' has no parameters (e.g., nn.Identity or empty).")
    else:
        print("  Model does not have 'distilled_guidance_layer' attribute or it's None.")


    print("\nCheckpoint keys starting with 'modulation_approximator.':")
    found_mod_approx_ckpt = False
    for k in sorted(list(checkpoint_keys)):
        if k.startswith("modulation_approximator."):
            print(f"  CKPT: {k}")
            found_mod_approx_ckpt = True
    if not found_mod_approx_ckpt:
        print("  None found.")

    print("\nModel keys starting with 'modulation_approximator.' (if it exists before rename):")
    # This checks the attribute name in the original Flux model structure before potential rename
    original_flux_for_keys = flux_models.Flux(chroma_flux_params) # Instantiate original for key names
    if hasattr(original_flux_for_keys, 'modulation_approximator') and original_flux_for_keys.modulation_approximator is not None:
        print(f"  Original Flux model structure has 'modulation_approximator' attribute. Type: {type(original_flux_for_keys.modulation_approximator)}")
        for name, _ in original_flux_for_keys.modulation_approximator.named_parameters():
            print(f"  MODEL_ORIG: modulation_approximator.{name}")
    else:
        print("  Original Flux model structure does not have 'modulation_approximator' or it's None.")

    print("\n--- Final Layer adaLN Modulation Analysis ---")
    print("Checkpoint keys for 'final_layer.adaLN_modulation.':")
    found_final_adaln_ckpt = False
    for k in checkpoint_keys:
        if k.startswith("final_layer.adaLN_modulation."):
            print(f"  CKPT: {k}")
            found_final_adaln_ckpt = True
    if not found_final_adaln_ckpt:
        print("  None found (implies final_layer.adaLN_modulation is pruned).")

    print("\nModel keys for 'final_layer.adaLN_modulation.':")
    # model_instance is TempFluxForAnalysis, which inherits from Flux
    if hasattr(model_instance, 'final_layer') and hasattr(model_instance.final_layer, 'adaLN_modulation'):
        for name, _ in model_instance.final_layer.adaLN_modulation.named_parameters():
            print(f"  MODEL: final_layer.adaLN_modulation.{name}")
        if not list(model_instance.final_layer.adaLN_modulation.named_parameters()):
             print("  Model's 'final_layer.adaLN_modulation' has no parameters.")
    else:
        print("  Model's 'final_layer' does not have 'adaLN_modulation'.")

if __name__ == "__main__":
    analyze_keys()