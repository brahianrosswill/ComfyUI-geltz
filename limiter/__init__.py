# __init__.py
from typing import Callable
import math
import torch
import comfy.samplers
from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
from comfy.model_patcher import ModelPatcher


class CFGLimiter(ComfyNodeABC):
    """
    Guidance limiter with smooth transitions, CFG-adaptive interval, and robust NaN/Inf sanitation.
    Based on "Applying Guidance in a Limited Interval Improves Sample and Distribution Quality"
    (Kynkäänniemi et al.)
    """

    # --- Tunables -------------------------------------------------------------
    TRANSITION_MODE = "cosine"   # "smooth", "cosine", "linear", "step"
    TRANSITION_WIDTH = 0.15      # sigma units for soft fade at both ends
    REFERENCE_CFG = 7.0          # interval is calibrated for this CFG
    ENABLE_CFG_SCALING = True    # scale [start,end,width] with actual CFG
    ENABLE_SANITIZE = True       # replace NaN/Inf in inputs before blending

    # --- Comfy UI plumbing ----------------------------------------------------
    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "model": (IO.MODEL, {}),
                "sigma_start": (IO.FLOAT, {
                    "default": 5.42, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False,
                    "tooltip": "Upper noise level for guidance (earlier in sampling)"
                }),
                "sigma_end": (IO.FLOAT, {
                    "default": 0.28, "min": -1.0, "max": 10000.0, "step": 0.01, "round": False,
                    "tooltip": "Lower noise level for guidance (later in sampling)"
                }),
            }
        }

    RETURN_TYPES = (IO.MODEL,)
    FUNCTION = "patch"
    CATEGORY = "model_patches/unet"
    DESCRIPTION = "Limits CFG to a sigma interval with smooth entry/exit and NaN/Inf-safe blending"

    # --- Helper factories -----------------------------------------------------
    @staticmethod
    def _get_interpolator(mode: str) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Returns f(t) for t in [0,1] → [0,1].
        Operates on tensors; caller clamps t to [0,1].
        """
        if mode == "linear":
            return lambda t: t
        if mode == "smooth":
            # smoothstep: t^2 (3-2t)
            def smooth(t: torch.Tensor) -> torch.Tensor:
                return t * t * (3.0 - 2.0 * t)
            return smooth
        if mode == "cosine":
            def cosine(t: torch.Tensor) -> torch.Tensor:
                # 0.5*(1 - cos(pi*t))
                return 0.5 * (1.0 - torch.cos(t * math.pi))
            return cosine
        # step
        return lambda t: torch.where(t > 0, torch.ones_like(t), torch.zeros_like(t))

    # --- Core weight computation ---------------------------------------------
    @staticmethod
    def _compute_weights(
        sigmas: torch.Tensor,
        cfg_scale: float,
        base_start: float,
        base_end: float,
        transition_width: float,
        reference_cfg: float,
        enable_cfg_scaling: bool,
        interp_fn: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Vectorized weight computation. Returns w in [0,1] per-sample.
        """
        # Ensure 1D tensor
        s = sigmas.reshape(-1).to(device=device, dtype=torch.float32)

        # Scale interval with CFG if requested
        start = float(base_start)
        end = float(base_end)
        width = float(transition_width)

        if enable_cfg_scaling and cfg_scale != reference_cfg:
            # Conservative scaling: shrink interval when CFG increases
            scale = reference_cfg / max(cfg_scale, 1.0)
            start *= scale
            end *= scale
            width *= scale

        # Start/end ordering guard (swap if user passed reversed values)
        if start < end:
            start, end = end, start

        # Width guard
        width = max(0.0, width)

        # Base: fully inside interval → weight=1
        w = torch.ones_like(s, dtype=torch.float32, device=device)

        # Upper boundary (early timesteps, higher sigma)
        if width > 0.0:
            upper_hard = s > (start + width)
            upper_soft = (s > start) & (s <= (start + width))
            # t decreases from 1 → 0 as we approach start from the hard side
            t_upper = ((start + width) - s) / width
            t_upper = torch.clamp(t_upper, 0.0, 1.0)
            w = torch.where(upper_hard, torch.zeros_like(w), w)
            w = torch.where(upper_soft, interp_fn(t_upper), w)
        else:
            w = torch.where(s > start, torch.zeros_like(w), w)

        # Lower boundary (late timesteps, lower sigma)
        if width > 0.0:
            lower_hard = s <= (end - width)
            lower_soft = (s > (end - width)) & (s <= end)
            # t increases 0 → 1 as we enter the soft range, then we *fade out*:
            # multiply by (1 - interp(t)) to go from 1 → 0
            t_lower = (s - (end - width)) / width
            t_lower = torch.clamp(t_lower, 0.0, 1.0)
            w = torch.where(lower_hard, torch.zeros_like(w), w)
            w = torch.where(lower_soft, w * (1.0 - interp_fn(t_lower)), w)
        else:
            w = torch.where(s <= end, torch.zeros_like(w), w)

        # Clamp for absolute safety and cast to model dtype later
        w = torch.clamp(w, 0.0, 1.0).to(dtype=dtype)
        return w

    # --- Sanitation to kill black dots ---------------------------------------
    @staticmethod
    def _sanitize_pair(x_cfg: torch.Tensor, x_uncond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Replace non-finite values by borrowing from the counterpart; if both non-finite, use 0.
        This prevents NaN/Inf speckling in the final image.
        """
        finite_cfg = torch.isfinite(x_cfg)
        finite_unc = torch.isfinite(x_uncond)

        # where guided is bad → take uncond; if uncond also bad → 0
        safe_cfg = torch.where(
            finite_cfg, x_cfg,
            torch.where(finite_unc, x_uncond, torch.zeros_like(x_cfg))
        )
        # where uncond is bad → take guided; if guided also bad → 0
        safe_unc = torch.where(
            finite_unc, x_uncond,
            torch.where(finite_cfg, x_cfg, torch.zeros_like(x_uncond))
        )
        return safe_cfg, safe_unc

    # --- Public: patch model --------------------------------------------------
    def patch(self, model: ModelPatcher, sigma_start: float, sigma_end: float):
        interp_fn = self._get_interpolator(self.TRANSITION_MODE)
        enable_cfg_scaling = self.ENABLE_CFG_SCALING
        reference_cfg = self.REFERENCE_CFG
        transition_width = self.TRANSITION_WIDTH
        sanitize = self.ENABLE_SANITIZE

        def limited_cfg(args):
            # Inputs from sampler
            x_cfg = args["denoised"]            # already CFG-applied
            x_unc = args["uncond_denoised"]
            sigma_v = args["sigma"]
            cfg_scale = float(args.get("cond_scale", 1.0))

            # Optional: sanitize incoming tensors to eliminate NaN/Inf sources
            if sanitize:
                x_cfg, x_unc = self._sanitize_pair(x_cfg, x_unc)

            # Robust sigma extraction → 1D tensor on the right device
            if torch.is_tensor(sigma_v):
                sigmas = sigma_v.detach().to(device=x_cfg.device, dtype=torch.float32).reshape(-1)
            else:
                sigmas = torch.tensor([float(sigma_v)], device=x_cfg.device, dtype=torch.float32)

            # Compute per-sample weights
            w = self._compute_weights(
                sigmas=sigmas,
                cfg_scale=cfg_scale,
                base_start=sigma_start,
                base_end=sigma_end,
                transition_width=transition_width,
                reference_cfg=reference_cfg,
                enable_cfg_scaling=enable_cfg_scaling,
                interp_fn=interp_fn,
                device=x_cfg.device,
                dtype=x_cfg.dtype,
            )

            # Broadcast to match [N, C, H, W, ...]
            while w.ndim < x_cfg.ndim:
                w = w.view(-1, *([1] * (x_cfg.ndim - 1)))

            # Fast paths (also avoids unnecessary ops)
            if torch.all(w == 0):
                return x_unc
            if torch.all(w == 1):
                return x_cfg

            # Stable blend (uncond + w * (cfg - uncond))
            return torch.lerp(x_unc, x_cfg, w)

        m = model.clone()
        m.set_model_sampler_post_cfg_function(limited_cfg)
        return (m,)


NODE_CLASS_MAPPINGS = {
    "CFGLimiter": CFGLimiter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CFGLimiter": "CFG Limiter",
}
