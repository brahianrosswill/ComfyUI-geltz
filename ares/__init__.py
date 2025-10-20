# Adaptive Refined Exponential Solver

from __future__ import annotations
import importlib
import comfy.samplers as _samplers
from comfy.k_diffusion import sampling as _kdiff
import torch

_ARES_STEP = None
try:
    ares_mod = importlib.import_module(".ares", __package__)
except Exception:
    try:
        ares_mod = importlib.import_module("ares")
    except Exception:
        ares_mod = None

if ares_mod is not None and hasattr(ares_mod, "_ares"):
    _ARES_STEP = getattr(ares_mod, "_ares")

@torch.no_grad()
def sample_ares(model, x, sigmas, extra_args=None, callback=None, disable=False,
                c2: float = 0.5, simple_phi_calc: bool = False, momentum: float = 0.0,
                sigma_min: float = 0.28, sigma_max: float = 25.0):
    n = int(sigmas.numel()) if torch.is_tensor(sigmas) else len(sigmas)
    if (_ARES_STEP is None) or (n < 2):
        return _kdiff.sample_euler(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)

    if extra_args is None:
        extra_args = {}

    if torch.is_tensor(sigmas):
        s = sigmas.flatten().to(device=x.device, dtype=torch.float32)
    else:
        s = torch.tensor(sigmas, device=x.device, dtype=torch.float32)
    keep_zero = torch.isclose(s[-1], torch.tensor(0.0, device=s.device))
    s_clamped = s.clone()
    if n > 1:
        s_clamped[:-1] = s_clamped[:-1].clamp(min=sigma_min, max=sigma_max)
    s_clamped[-1] = 0.0 if keep_zero else float(max(sigma_min, min(sigma_max, float(s[-1]))))

    h = 1.0
    vel = None
    x0_est = None

    total = n - 1
    for i in range(total):
        sigma = float(s_clamped[i].item())
        sigma_next = float(s_clamped[i + 1].item())
        if sigma_next == sigma:
            if callback is not None:
                callback({"i": i, "denoised": x0_est if x0_est is not None else x, "x": x})
            continue

        x, vel, x0_est, h = _ARES_STEP(
            model, x, sigma, sigma_next, h,
            c2=c2, extra_args=extra_args, pbar=callback,
            simple_phi_calc=simple_phi_calc, momentum=momentum,
            vel=vel, vel_2=None, time=None
        )

        if callback is not None:
            callback({"i": i, "denoised": x0_est if x0_est is not None else x, "x": x})

    return x

def _register_sampler_name(name: str):
    try:
        lst = _samplers.KSampler.SAMPLERS
        if isinstance(lst, (tuple, set)):
            lst = list(lst)
            _samplers.KSampler.SAMPLERS = lst
        if name not in lst:
            lst.append(name)
    except Exception:
        pass

def _register_into_kdiff(name: str, func):
    setattr(_kdiff, f"sample_{name}", func)

_register_into_kdiff("ares", sample_ares)
_register_sampler_name("ares")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
