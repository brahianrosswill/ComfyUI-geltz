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
            scaled_width = transition_width

            # Dynamically adjust interval based on CFG scale
            if enable_cfg_scaling and cfg_scale != reference_cfg:
                scale_factor = reference_cfg / max(cfg_scale, 1.0)
                current_start = base_start * scale_factor
                current_end = base_end * scale_factor
                scaled_width = transition_width * scale_factor

            # Upper boundary
            if not start_disabled:
                if sigma > current_start + scaled_width:
                    return 0.0
                elif sigma > current_start:
                    if scaled_width > 0:
                        t = (current_start + scaled_width - sigma) / scaled_width
                        weight *= transition_interpolator(t)
                    else:
                        return 0.0

            # Lower boundary
            if not end_disabled:
                if sigma <= current_end - scaled_width:
                    return 0.0
                elif sigma <= current_end:
                    if scaled_width > 0:
                        t = (sigma - (current_end - scaled_width)) / scaled_width
                        weight *= transition_interpolator(t)
                    else:
                        return 0.0
            
            return weight

        # --- This is the main patch function passed to the sampler ---
        
        def limited_cfg(args):  # [~L121]
            x_cfg   = args["denoised"]          # guided (already CFG-applied)
            uncond  = args["uncond_denoised"]
            sigma_v = args["sigma"]
            cfg_scale = float(args.get("cond_scale", 1.0))

            # Robust sigma extraction -> 1D tensor on the right device/dtype
            if torch.is_tensor(sigma_v):
                sigmas = sigma_v.detach().to(device=x_cfg.device, dtype=torch.float32).reshape(-1)
            else:
                sigmas = torch.tensor([float(sigma_v)], device=x_cfg.device, dtype=torch.float32)

            # Compute per-sample scalar weights on CPU (cheap) then send to device
            weights = []
            for s in sigmas.tolist():
                if not math.isfinite(s):
                    weights.append(0.0)  # safer fallback = no guidance
                else:
                    w = compute_transition_weight(s, cfg_scale, sigma_start, sigma_end)
                    # clamp for safety
                    weights.append(0.0 if w <= 0.0 else (1.0 if w >= 1.0 else float(w)))

            w = torch.tensor(weights, device=x_cfg.device, dtype=x_cfg.dtype)
            # Broadcast weight across [N, C, H, W] etc.
            while w.ndim < x_cfg.ndim:
                w = w.view(-1, *([1] * (x_cfg.ndim - 1)))

            # Fast paths
            if torch.all(w == 0):
                return uncond
            if torch.all(w == 1):
                return x_cfg

            # Smooth mix: uncond + w * (x_cfg - uncond)
            # (lerp keeps dtype/device and is numerically stable)
            return torch.lerp(uncond, x_cfg, w)

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