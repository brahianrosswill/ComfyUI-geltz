# cfg_limiter.py
from __future__ import annotations
import torch
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.model_patcher import ModelPatcher

class CFGLimiter(ComfyNodeABC):
    """
    Minimal, vectorized CFG limiter.
    Disables classifier-free guidance outside [sigma_min, sigma_max].
    Set sigma_min or sigma_max to -1 to ignore that bound.
    """

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {}),
                # Defaults match common SD settings; use -1 to disable a bound
                "sigma_max": (IO.FLOAT, {"default": 5.42, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
                "sigma_min": (IO.FLOAT, {"default": 0.28, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False}),
            }
        }

    RETURN_TYPES = (IO.MODEL,)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"

    def patch(self, model: ModelPatcher, sigma_max: float, sigma_min: float):
        # Hoist constants to avoid attribute lookups in the hot path
        _use_max = sigma_max >= 0.0
        _use_min = sigma_min >= 0.0
        _sigma_max = float(sigma_max)
        _sigma_min = float(sigma_min)

        def cfg_limiter(args):
            """
            args:
              - denoised: x_cfg (guided)
              - cond_denoised: x_cond (no CFG)
              - sigma: scalar or (B,) tensor of current noise sigmas
            """
            x_cfg: torch.Tensor = args["denoised"]
            x_cond: torch.Tensor = args["cond_denoised"]
            sigma: torch.Tensor = args["sigma"]

            # Fast path: nothing to do
            if not (_use_max or _use_min):
                return x_cfg

            # Make sigma broadcastable over NCHW...
            # If sigma is scalar -> shape (1,1,1,1)
            # If sigma is (B,)  -> shape (B,1,1,1)
            if sigma.ndim == 0:
                sv = sigma.reshape(1, *([1] * (x_cfg.ndim - 1)))
            else:
                sv = sigma.view(-1, *([1] * (x_cfg.ndim - 1)))

            # Build mask: True where guidance should be disabled (i.e., use cond)
            mask = torch.zeros_like(x_cfg, dtype=torch.bool)
            if _use_max:
                mask |= sv > _sigma_max
            if _use_min:
                mask |= sv <= _sigma_min

            # Select per-sample/per-pixel without Python branching
            return torch.where(mask, x_cond, x_cfg)

        m = model.clone()
        m.set_model_sampler_post_cfg_function(cfg_limiter)
        return (m,)

# Register node
NODE_CLASS_MAPPINGS = {
    "CFG Limiter": CFGLimiter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CFG Limiter": "CFG Limiter",
}
