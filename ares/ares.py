from __future__ import annotations
import importlib
import torch
import comfy.samplers as _samplers
from comfy.k_diffusion import sampling as _kdiff

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
    extra_args = {} if extra_args is None else extra_args
    s = sigmas.flatten().to(device=x.device, dtype=torch.float32) if torch.is_tensor(sigmas) else torch.tensor(sigmas, device=x.device, dtype=torch.float32)
    keep_zero = torch.isclose(s[-1], torch.tensor(0.0, device=s.device))
    s_clamped = s.clone()
    if n > 1:
        s_clamped[:-1] = s_clamped[:-1].clamp(min=sigma_min, max=sigma_max)
    s_clamped[-1] = 0.0 if keep_zero else float(max(sigma_min, min(sigma_max, float(s[-1]))))
    h, vel, x0_est, total = 1.0, None, None, n - 1
    for i in range(total):
        sigma = float(s_clamped[i].item())
        sigma_next = float(s_clamped[i + 1].item())
        if sigma_next == sigma:
            if callback is not None:
                callback({"i": i, "denoised": x0_est if x0_est is not None else x, "x": x})
            continue
        x, vel, x0_est, h = _ARES_STEP(model, x, sigma, sigma_next, h, c2=c2, extra_args=extra_args, pbar=callback, simple_phi_calc=simple_phi_calc, momentum=momentum, vel=vel, vel_2=None, time=None)
        if callback is not None:
            callback({"i": i, "denoised": x0_est if x0_est is not None else x, "x": x})
    return x

def _rms(z):
    return torch.sqrt(torch.mean(z.float() * z.float()) + 1e-8)

class _RDAProxy:
    def __init__(self, base_model, n_calls, tau=0.02, gamma=1.2, max_stale=2, w=1.0):
        self.m, self.n = base_model, max(1, int(n_calls))
        self.tau, self.gamma, self.max_stale, self.w = float(tau), float(gamma), int(max_stale), float(w)
        self.i, self.prev, self.prev2, self.stale = 0, None, None, 0

    @torch.no_grad()
    def __call__(self, x, sigma, **extra):
        p = min(1.0, max(0.0, self.i / self.n))
        if self.prev is None or self.prev2 is None or self.stale >= self.max_stale:
            y = self.m(x, sigma, **extra)
            self.prev2, self.prev, self.stale, self.i = self.prev, y, 0, self.i + 1
            return y
        thr = self.tau * ((1.0 - p + 1e-3) ** self.gamma)
        d = self.prev - self.prev2
        r, rd = _rms(self.prev), _rms(d)
        rel = rd / max(1e-8, r)
        if rel < thr:
            y = self.prev + self.w * d
            yr = _rms(y)
            if yr > 0 and r > 0:
                y = y * torch.clamp(r / yr, max=2.0).to(y)
            self.prev2, self.prev, self.stale, self.i = self.prev, y, min(self.stale + 1, self.max_stale), self.i + 1
            return y
        y = self.m(x, sigma, **extra)
        self.prev2, self.prev, self.stale, self.i = self.prev, y, 0, self.i + 1
        return y

@torch.no_grad()
def sample_ares_rda(model, x, sigmas, extra_args=None, callback=None, disable=False,
                    c2: float = 0.5, simple_phi_calc: bool = False, momentum: float = 0.0,
                    sigma_min: float = 0.28, sigma_max: float = 25.0,
                    tau: float = 0.02, gamma: float = 1.2, max_stale: int = 2, w: float = 1.0):
    n = int(sigmas.numel()) if torch.is_tensor(sigmas) else len(sigmas)
    if (_ARES_STEP is None) or (n < 2):
        return sample_ares(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable, c2=c2, simple_phi_calc=simple_phi_calc, momentum=momentum, sigma_min=sigma_min, sigma_max=sigma_max)
    extra_args = {} if extra_args is None else extra_args
    s = sigmas.flatten().to(device=x.device, dtype=torch.float32) if torch.is_tensor(sigmas) else torch.tensor(sigmas, device=x.device, dtype=torch.float32)
    keep_zero = torch.isclose(s[-1], torch.tensor(0.0, device=s.device))
    s_clamped = s.clone()
    if n > 1:
        s_clamped[:-1] = s_clamped[:-1].clamp(min=sigma_min, max=sigma_max)
    s_clamped[-1] = 0.0 if keep_zero else float(max(sigma_min, min(sigma_max, float(s[-1]))))
    h, vel, x0_est, total = 1.0, None, None, n - 1
    proxy = _RDAProxy(model, n_calls=total, tau=tau, gamma=gamma, max_stale=max_stale, w=w)
    for i in range(total):
        sigma = float(s_clamped[i].item())
        sigma_next = float(s_clamped[i + 1].item())
        if sigma_next == sigma:
            if callback is not None:
                callback({"i": i, "denoised": x0_est if x0_est is not None else x, "x": x})
            continue
        x, vel, x0_est, h = _ARES_STEP(proxy, x, sigma, sigma_next, h, c2=c2, extra_args=extra_args, pbar=callback, simple_phi_calc=simple_phi_calc, momentum=momentum, vel=vel, vel_2=None, time=None)
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
_register_into_kdiff("ares_rda", sample_ares_rda)
_register_sampler_name("ares_rda")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
