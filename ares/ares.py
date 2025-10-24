from __future__ import annotations
import importlib
import torch
import comfy.samplers as _samplers
from comfy.k_diffusion import sampling as _kdiff

_ARES_STEP = None

def _resolve_ares_step():
    import importlib

    # Try relative first (inside this node pack), then absolute import.
    candidates = [
        (".ares", __package__),
        ("ares", None),
    ]
    # Accept any of these attribute names as the step function.
    attrs = ("_ares", "ares_step", "ARES_STEP")

    for modname, pkg in candidates:
        try:
            mod = importlib.import_module(modname, pkg) if pkg else importlib.import_module(modname)
        except Exception:
            continue
        for attr in attrs:
            fn = getattr(mod, attr, None)
            if callable(fn):
                print(f"[ares] OK: using {mod.__name__}.{attr}")
                return fn

    print("[ares] WARNING: could not locate _ares/ares_step; falling back to k-diffusion euler. ares and ares_rda will behave the same.")
    return None

_ARES_STEP = _resolve_ares_step()

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

class _RDAStepProxy:
    """RDA wrapper that caches _ARES_STEP outputs, not individual model calls."""
    def __init__(self, n_steps, tau=0.15, gamma=0.75, max_stale=3, w=1.0):
        self.n = max(1, int(n_steps))
        self.tau, self.gamma, self.max_stale, self.w = float(tau), float(gamma), int(max_stale), float(w)
        self.i, self.prev_x, self.prev2_x, self.stale = 0, None, None, 0

    @torch.no_grad()
    def step(self, model, x, sigma, sigma_next, h, c2, extra_args, pbar, simple_phi_calc, momentum, vel, vel_2, time):
        """Wrapper around _ARES_STEP with RDA acceleration."""
        p = min(1.0, max(0.0, self.i / self.n))
        
        # First two steps or exceeded max_stale: always compute
        if self.prev_x is None or self.prev2_x is None or self.stale >= self.max_stale:
            x_out, vel_out, x0_est_out, h_out = _ARES_STEP(
                model, x, sigma, sigma_next, h, c2=c2, extra_args=extra_args, 
                pbar=pbar, simple_phi_calc=simple_phi_calc, momentum=momentum, 
                vel=vel, vel_2=vel_2, time=time
            )
            self.prev2_x, self.prev_x = self.prev_x, x_out
            self.stale = 0
            self.i += 1
            return x_out, vel_out, x0_est_out, h_out
        
        # Check if we can extrapolate
        thr = self.tau * ((1.0 - p + 1e-3) ** self.gamma)
        d = self.prev_x - self.prev2_x
        r, rd = _rms(self.prev_x), _rms(d)
        rel = rd / max(1e-8, r)
        
        if rel < thr:
            # Extrapolate from previous steps
            x_out = self.prev_x + self.w * d
            yr = _rms(x_out)
            if yr > 0 and r > 0:
                x_out = x_out * torch.clamp(r / yr, max=2.0).to(x_out)
            
            self.prev2_x, self.prev_x = self.prev_x, x_out
            self.stale = min(self.stale + 1, self.max_stale)
            self.i += 1
            # Return extrapolated x with previous vel, x0_est, h (approximation)
            return x_out, vel, x_out, h  # Use x_out as x0_est approximation
        
        # Compute full step
        x_out, vel_out, x0_est_out, h_out = _ARES_STEP(
            model, x, sigma, sigma_next, h, c2=c2, extra_args=extra_args, 
            pbar=pbar, simple_phi_calc=simple_phi_calc, momentum=momentum, 
            vel=vel, vel_2=vel_2, time=time
        )
        self.prev2_x, self.prev_x = self.prev_x, x_out
        self.stale = 0
        self.i += 1
        return x_out, vel_out, x0_est_out, h_out

@torch.no_grad()
def sample_ares_rda(model, x, sigmas, extra_args=None, callback=None, disable=False,
                    c2: float = 0.5, simple_phi_calc: bool = False, momentum: float = 0.0,
                    sigma_min: float = 0.28, sigma_max: float = 25.0,
                    tau: float = 0.07, gamma: float = 1.0, max_stale: int = 2, w: float = 0.8):
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
    
    # Create RDA proxy that wraps _ARES_STEP calls, not model calls
    rda_proxy = _RDAStepProxy(n_steps=total, tau=tau, gamma=gamma, max_stale=max_stale, w=w)
    
    for i in range(total):
        sigma = float(s_clamped[i].item())
        sigma_next = float(s_clamped[i + 1].item())
        if sigma_next == sigma:
            if callback is not None:
                callback({"i": i, "denoised": x0_est if x0_est is not None else x, "x": x})
            continue
        
        # Call through RDA proxy
        x, vel, x0_est, h = rda_proxy.step(
            model, x, sigma, sigma_next, h, c2=c2, extra_args=extra_args, 
            pbar=callback, simple_phi_calc=simple_phi_calc, momentum=momentum, 
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
_register_into_kdiff("ares_rda", sample_ares_rda)
_register_sampler_name("ares_rda")

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}