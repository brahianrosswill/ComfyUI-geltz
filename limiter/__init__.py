from typing import Literal, Optional, Callable
import torch
import comfy.samplers
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.model_patcher import ModelPatcher
import math  # Import math at the top level


class CFGLimiter(ComfyNodeABC):
    """
    Enhanced guidance limiter with smooth transitions and CFG-adaptive intervals.
    Based on "Applying Guidance in a Limited Interval Improves Sample and Distribution Quality"
    by Kynkäänniemi et al.
    
    Corrected and optimized version.
    """
    
    # Hyperparameters - tuned for optimal quality/efficiency trade-off
    TRANSITION_MODE = "cosine"  # smooth, cosine, linear, or step
    TRANSITION_WIDTH = 0.15  # sigma units for smooth blending at boundaries
    REFERENCE_CFG = 7.0  # CFG value where provided interval is optimal
    ENABLE_CFG_SCALING = True  # automatically adjust interval based on CFG
    
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {}),
                "sigma_start": (IO.FLOAT, {
                    "default": 5.42, 
                    "min": -1.0, 
                    "max": 10000.0, 
                    "step": 0.01, 
                    "round": False,
                    "tooltip": "Upper noise level for guidance (higher = earlier in sampling)"
                }),
                "sigma_end": (IO.FLOAT, {
                    "default": 0.28, 
                    "min": -1.0, 
                    "max": 10000.0, 
                    "step": 0.01, 
                    "round": False,
                    "tooltip": "Lower noise level for guidance (lower = later in sampling)"
                }),
            }
        }

    RETURN_TYPES = (IO.MODEL,)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"
    DESCRIPTION = "Guidance limiter with smooth transitions and automatic CFG adaptation"

    def patch(
        self, 
        model: ModelPatcher, 
        sigma_start: float, 
        sigma_end: float
    ):
        # --- Create helper functions once ---
        
        # Get the core interpolation function (e.g., smoothstep, cosine)
        transition_interpolator = self._get_interpolator(self.TRANSITION_MODE)
        
        # Pre-bind self attributes for the closure
        enable_cfg_scaling = self.ENABLE_CFG_SCALING
        reference_cfg = self.REFERENCE_CFG
        transition_width = self.TRANSITION_WIDTH

        def compute_transition_weight(
            sigma: float, 
            cfg_scale: float,
            base_start: float,
            base_end: float
        ) -> float:
            """
            Calculates the guidance weight (0.0 to 1.0) for a given sigma,
            handling CFG scaling and smooth transitions efficiently.
            """
            current_start = base_start
            current_end = base_end
            
            # Dynamically adjust interval based on CFG scale
            if enable_cfg_scaling and cfg_scale != reference_cfg:
                scale_factor = reference_cfg / max(cfg_scale, 1.0)
                current_start = base_start * scale_factor
                current_end = base_end * scale_factor
            
            # Handle disable cases
            start_disabled = current_start < 0
            end_disabled = current_end < 0
            
            # Calculate weight
            weight = 1.0
            
            # Upper boundary transition
            if not start_disabled:
                if sigma > current_start + transition_width:
                    return 0.0  # Fully off
                elif sigma > current_start:
                    if transition_width > 0:
                        # t goes from 1.0 (at sigma=current_start) to 0.0 (at sigma=current_start + width)
                        t = (current_start + transition_width - sigma) / transition_width
                        weight *= transition_interpolator(t)
                    else:
                        return 0.0  # Step mode, hard cutoff
            
            # Lower boundary transition
            if not end_disabled:
                if sigma <= current_end - transition_width:
                    return 0.0  # Fully off
                elif sigma <= current_end:
                    if transition_width > 0:
                        # t goes from 0.0 (at sigma=current_end - width) to 1.0 (at sigma=current_end)
                        t = (sigma - (current_end - transition_width)) / transition_width
                        weight *= transition_interpolator(t)
                    else:
                        return 0.0  # Step mode, hard cutoff
            
            return weight

        # --- This is the main patch function passed to the sampler ---
        
        def limited_cfg(args):
            x_cfg = args["denoised"]          # Fully guided output
            uncond = args["uncond_denoised"]  # Unconditional output
            sigma = args["sigma"]
            cfg_scale = args.get("cond_scale", 1.0)
            
            # Get guidance weight multiplier (0.0 to 1.0)
            weight = compute_transition_weight(
                sigma[0].item(), 
                cfg_scale, 
                sigma_start, 
                sigma_end
            )
            
            # Fast path: full bypass (return UNCONDITIONAL)
            if weight == 0.0:
                return uncond
            
            # Fast path: full guidance
            if weight == 1.0:
                return x_cfg
            
            # CRITICAL FIX: Interpolate between unconditional and guided result
            return uncond + weight * (x_cfg - uncond)

        m = model.clone()
        m.set_model_sampler_post_cfg_function(limited_cfg)
        return (m,)
    
    @staticmethod
    def _get_interpolator(mode: str) -> Callable[[float], float]:
        """
        Returns a simple interpolation function mapping t=[0, 1] -> [0, 1].
        """
        if mode == "linear":
            # Clamping is handled by the caller, but good practice.
            return lambda x: max(0.0, min(1.0, x))
        
        if mode == "smooth":
            def smooth_step(x: float) -> float:
                x = max(0.0, min(1.0, x))
                return x * x * (3.0 - 2.0 * x)
            return smooth_step

        if mode == "cosine":
            def cosine_interp(x: float) -> float:
                x = max(0.0, min(1.0, x))
                return 0.5 * (1.0 - math.cos(x * math.pi))
            return cosine_interp

        # Default to step function (always return 1.0 within computed range)
        return lambda x: 1.0


NODE_CLASS_MAPPINGS = {
    "CFGLimiter": CFGLimiter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CFGLimiter": "CFG Limiter",
}