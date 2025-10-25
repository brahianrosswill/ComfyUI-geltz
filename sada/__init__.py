import torch
import comfy.model_patcher
import comfy.samplers

class SADAAccelerator:
    """Stateful accelerator that tracks sampling trajectory."""
    
    def __init__(self, enable_step: bool, stability_threshold: float, max_consecutive_skips: int = 2):
        self.enable_step = enable_step
        self.stability_threshold = stability_threshold
        self.max_consecutive_skips = max_consecutive_skips
        self.reset()
    
    def reset(self):
        self.x_history = []
        self.eps_history = []
        self.step_count = 0
        self.skipped_steps = 0
        self.total_steps = 0
        self.consecutive_skips = 0
        self.last_skip_step = -10
        
    def should_skip_step(self) -> bool:
        """Check stability criterion from paper Criterion 3.4 with conservative guards."""
        if not self.enable_step or len(self.x_history) < 3 or len(self.eps_history) < 3:
            return False
        
        # Don't skip too many consecutive steps (prevents error accumulation)
        if self.consecutive_skips >= self.max_consecutive_skips:
            return False
        
        # Require gap between skips for fresh computation
        if self.total_steps - self.last_skip_step < 2:
            return False
        
        # Get recent trajectory points
        x_t = self.x_history[-1]
        x_t1 = self.x_history[-2]
        x_t2 = self.x_history[-3]
        
        # Third-order finite difference extrapolation
        x_est = 3 * x_t - 3 * x_t1 + x_t2
        
        # Get noise predictions
        eps_t = self.eps_history[-1]
        eps_t1 = self.eps_history[-2]
        eps_t2 = self.eps_history[-3]
        
        # Second-order difference: Δ²ε_t = ε_t - 2ε_{t-1} + ε_{t-2}
        delta_2_eps = eps_t - 2 * eps_t1 + eps_t2
        
        # Stability criterion: (x_t - x̂_t) · Δ²ε_t < 0
        error = x_t - x_est
        
        # Flatten and compute dot product
        error_flat = error.flatten()
        delta_flat = delta_2_eps.flatten()
        criterion = torch.dot(error_flat, delta_flat)
        
        # Must be anti-aligned (negative)
        if criterion.item() >= 0:
            return False
        
        # Check relative error magnitude
        error_magnitude = torch.norm(error).item()
        x_magnitude = torch.norm(x_t).item()
        relative_error = error_magnitude / (x_magnitude + 1e-8)
        
        # Check noise prediction stability (how much is it changing?)
        eps_change = torch.norm(eps_t - eps_t1).item()
        eps_magnitude = torch.norm(eps_t).item()
        relative_eps_change = eps_change / (eps_magnitude + 1e-8)
        
        # Conservative thresholds
        is_trajectory_stable = relative_error < self.stability_threshold
        is_prediction_stable = relative_eps_change < self.stability_threshold * 0.5
        
        # Both must be stable
        return is_trajectory_stable and is_prediction_stable
    
    def get_cached_prediction(self):
        """Return previous noise prediction for step skipping."""
        if len(self.eps_history) >= 1:
            return self.eps_history[-1]
        return None
    
    def mark_skipped(self):
        """Track that a skip occurred."""
        self.consecutive_skips += 1
        self.last_skip_step = self.total_steps
        
    def mark_computed(self):
        """Track that full computation occurred."""
        self.consecutive_skips = 0


class SADAModelPatch:
    """
    SADA acceleration node for ComfyUI that patches the model to use
    stability-guided adaptive pruning during sampling.
    This version adds **proper teardown/flush** so that when you disable
    the node, any previously patched model is restored and all caches are cleared.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enable_acceleration": ("BOOLEAN", {"default": True}),
                "stability_threshold": ("FLOAT", {
                    "default": 0.15,
                    "min": 0.01,
                    "max": 0.5,
                    "step": 0.01,
                    "display": "slider",
                    "tooltip": "Lower = more aggressive skipping, Higher = more conservative"
                }),
                "min_steps_before_skip": ("INT", {
                    "default": 3,
                    "min": 2,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Minimum steps before checking stability"
                }),
                "max_consecutive_skips": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Safety guard: cap on consecutive skipped steps"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch_model"
    CATEGORY = "model_patches/acceleration"

    @staticmethod
    def _unpatch_if_present(model):
        """
        If the incoming model (or its underlying UNet) was previously patched
        by SADA, restore the original apply_model and clear state.
        """
        # Prefer storing the original on the UNet to survive model cloning
        unet = getattr(model, "model", None)
        if unet is not None and getattr(unet, "_sada_wrapped", False):
            try:
                # Clear accelerator state if present
                acc = getattr(unet, "_sada_accelerator", None)
                if acc is not None:
                    acc.reset()
                # Restore apply_model
                orig = getattr(unet, "_sada_original_apply_model", None)
                if orig is not None:
                    unet.apply_model = orig
                # Drop flags/attrs
                for attr in ["_sada_wrapped", "_sada_original_apply_model", "_sada_accelerator"]:
                    if hasattr(unet, attr):
                        delattr(unet, attr)
                # Also clear convenience attrs on the outer model
                for attr in ["sada_reset", "sada_unpatch", "sada_active"]:
                    if hasattr(model, attr):
                        delattr(model, attr)
                print("[SADA] Unpatched previous wrapper and flushed state.")
            except Exception as e:
                print(f"[SADA] Warning: attempted to unpatch but hit: {e}")

    def patch_model(
        self,
        model,
        enable_acceleration: bool,
        stability_threshold: float,
        min_steps_before_skip: int,
        max_consecutive_skips: int,
    ):
        """
        Applies SADA patch to the model using ComfyUI's model patcher.
        Also guarantees that disabling the node *fully* restores the model.
        """
        # If the model (or its shared UNet) is already wrapped from a previous run,
        # clean it up first to avoid double-wrapping or lingering effects.
        self._unpatch_if_present(model)

        if not enable_acceleration:
            print("[SADA] Acceleration disabled. Ensured any previous patch is removed.")
            return (model,)

        # Clone model to avoid modifying original instance wiring in the graph
        patched_model = model.clone()

        # Create accelerator instance (pass all args)
        accelerator = SADAAccelerator(enable_acceleration, stability_threshold, max_consecutive_skips)

        # Store original apply_model function on the UNet object itself
        unet = patched_model.model
        original_apply_model = unet.apply_model

        def sada_apply_model(x, t, c_concat=None, c_crossattn=None, control=None, transformer_options=None, **kwargs):
            """Wrapped apply_model that implements SADA acceleration."""

            # Track total steps
            accelerator.total_steps += 1

            # Check if we should skip this step
            if accelerator.step_count >= min_steps_before_skip and accelerator.should_skip_step():
                # Skip computation, reuse cached prediction
                cached = accelerator.get_cached_prediction()
                if cached is not None:
                    accelerator.skipped_steps += 1
                    accelerator.mark_skipped()
                    if accelerator.total_steps % 10 == 0:
                        skip_ratio = accelerator.skipped_steps / max(1, accelerator.total_steps)
                        print(f"[SADA] Step {accelerator.total_steps}: Skipped {accelerator.skipped_steps}/{accelerator.total_steps} ({skip_ratio:.1%} acceleration)")
                    return cached

            # Perform full computation
            output = original_apply_model(x, t, c_concat, c_crossattn, control, transformer_options, **kwargs)

            # Update history for stability checking
            accelerator.x_history.append(x.detach().clone())
            accelerator.eps_history.append(output.detach().clone())

            # Keep history bounded
            if len(accelerator.x_history) > 5:
                accelerator.x_history.pop(0)
            if len(accelerator.eps_history) > 5:
                accelerator.eps_history.pop(0)

            accelerator.step_count += 1
            accelerator.mark_computed()

            return output

        # Patch the model (bind on the UNet to ensure single point of truth)
        unet.apply_model = sada_apply_model
        # Mark & keep handles on the UNet so we can unpatch even if the outer
        # ModelPatcher instance isn't the same object on future runs
        unet._sada_wrapped = True
        unet._sada_original_apply_model = original_apply_model
        unet._sada_accelerator = accelerator

        # Add callbacks to reset/unpatch accelerator state
        def reset_callback():
            accelerator.reset()
            print(f"[SADA] Initialized with threshold={stability_threshold}, min_steps={min_steps_before_skip}, max_consecutive_skips={max_consecutive_skips}")

        def unpatch_callback():
            # Allow users or other nodes to explicitly drop the wrapper
            self._unpatch_if_present(patched_model)

        # Expose helpers on the outer model
        patched_model.sada_reset = reset_callback
        patched_model.sada_unpatch = unpatch_callback
        patched_model.sada_active = True

        # Reset now for immediate use
        reset_callback()

        print(f"[SADA] Model patched successfully")
        print(f"  - Stability threshold: {stability_threshold}")
        print(f"  - Min steps before skipping: {min_steps_before_skip}")
        print(f"  - Max consecutive skips: {max_consecutive_skips}")

        return (patched_model,)


class SADAInfo:
    """Optional node to display SADA statistics."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "show"
    CATEGORY = "model_patches/acceleration"

    def show(self, model):
        if hasattr(model, "sada_reset"):
            print("[SADAInfo] SADA is patched on this model.")
        else:
            print("[SADAInfo] SADA is NOT patched on this model.")
        return ()



NODE_CLASS_MAPPINGS = {
    "SADAModelPatch": SADAModelPatch,
    "SADAInfo": SADAInfo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SADAModelPatch": "SADA Model Acceleration",
    "SADAInfo": "SADA Info",
}