from __future__ import annotations
import importlib
import torch
import comfy.samplers as _samplers
from comfy.k_diffusion import sampling as _kdiff

_ARES_STEP = None

def _broadcast_arg_like_batch(arg, x):
    if arg is None:
        return None
    if not torch.is_tensor(arg):
        arg = torch.as_tensor(arg, device=x.device, dtype=x.dtype)
    if arg.ndim == 0:
        arg = arg.repeat(x.shape[0])
    elif arg.ndim == 1 and arg.shape[0] != x.shape[0]:
        arg = arg[: x.shape[0]] if arg.shape[0] > x.shape[0] else arg.repeat(int((x.shape[0] + arg.shape[0] - 1) // arg.shape[0]))[: x.shape[0]]
    return arg.to(device=x.device)

@torch.no_grad()
def _ares_builtin(model, x, sigma, sigma_next, h, c2=0.5, extra_args=None, pbar=None,
                  simple_phi_calc=False, momentum=0.0, vel=None, vel_2=None, time=None):
    """Built-in ARES step implementation with second-order correction."""
    extra_args = {} if extra_args is None else dict(extra_args)
    device = x.device
    B = x.shape[0]

    # Ensure per-sample sigma shapes
    s = torch.as_tensor(sigma if torch.is_tensor(sigma) else float(sigma), device=device, dtype=x.dtype)
    sn = torch.as_tensor(sigma_next if torch.is_tensor(sigma_next) else float(sigma_next), device=device, dtype=x.dtype)
    
    if s.ndim == 0:
        s = s.repeat(B)
    if sn.ndim == 0:
        sn = sn.repeat(B)

    # Normalize common scheduler/model args
    if "timestep" in extra_args:
        extra_args["timestep"] = _broadcast_arg_like_batch(extra_args["timestep"], x)
    if "sigmas" in extra_args:
        extra_args["sigmas"] = _broadcast_arg_like_batch(extra_args["sigmas"], x)
    if "sigma" in extra_args:
        extra_args["sigma"] = _broadcast_arg_like_batch(extra_args["sigma"], x)

    # Get first denoised prediction (x0) from model
    denoised = model(x, s, **extra_args)
    
    # Compute derivative: d = (x - denoised) / sigma
    s_view = s.view(B, *([1] * (x.ndim - 1)))
    sn_view = sn.view(B, *([1] * (x.ndim - 1)))
    d = (x - denoised) / s_view.clamp(min=1e-10)
    
    # Step size
    dt = (sn - s).view(B, *([1] * (x.ndim - 1)))
    
    # First-order Euler step
    x_euler = x + dt * d
    
    # Second-order correction (ARES improvement over Euler)
    # Take a half-step, evaluate derivative there, and use it for correction
    if simple_phi_calc or torch.abs(dt).max() < 1e-8:
        # Skip second-order correction for very small steps
        x_next = x_euler
    else:
        # Midpoint for second derivative estimate
        s_mid = s + c2 * (sn - s)
        s_mid_view = s_mid.view(B, *([1] * (x.ndim - 1)))
        x_mid = x + c2 * dt * d
        
        # Update extra_args for midpoint evaluation if needed
        extra_args_mid = extra_args.copy()
        if "sigma" in extra_args_mid:
            extra_args_mid["sigma"] = s_mid
        
        # Get denoised prediction at midpoint
        denoised_mid = model(x_mid, s_mid, **extra_args_mid)
        d_mid = (x_mid - denoised_mid) / s_mid_view.clamp(min=1e-10)
        
        # Second-order correction: use midpoint derivative for better accuracy
        # This is the key difference from plain Euler
        x_next = x + dt * d_mid
    
    # Apply momentum if requested
    if momentum != 0.0 and vel is not None:
        vel_new = x_next - x
        x_next = x_next + momentum * vel_new
        vel_out = vel_new
    else:
        vel_out = x_next - x if vel is not None else None
    
    # Estimate h for next step (average step size)
    h_out = float(torch.mean(torch.abs(dt)).item())
    
    return x_next, vel_out, denoised, h_out

def _resolve_ares_step():
    """Try to find external ARES implementation, fall back to built-in."""
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
                print(f"[ares] OK: using {attr} from {modname}")
                return fn

    # If no external module found, use built-in
    print("[ares] Using built-in ARES step")
    return _ares_builtin

# Initialize ARES step function (now after _ares_builtin is defined)
_ARES_STEP = _resolve_ares_step()

@torch.no_grad()
def sample_ares(model, x, sigmas, extra_args=None, callback=None, disable=False,
                c2: float = 0.5, simple_phi_calc: bool = False, momentum: float = 0.0,
                sigma_min: float = 0.0, sigma_max: float = float('inf')):
    """ARES sampler - Adaptive Residual Euler Solver."""
    n = int(sigmas.numel()) if torch.is_tensor(sigmas) else len(sigmas)
    if (_ARES_STEP is None) or (n < 2):
        return _kdiff.sample_euler(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)
    
    extra_args = {} if extra_args is None else extra_args
    
    # Handle sigmas
    if torch.is_tensor(sigmas):
        sigmas = sigmas.to(device=x.device, dtype=torch.float32)
    else:
        sigmas = torch.tensor(sigmas, device=x.device, dtype=torch.float32)
    
    n = len(sigmas) - 1
    if n < 1:
        return x
    
    # Clamp sigmas to valid range (except final zero)
    sigmas_clamped = sigmas.clone()
    if sigma_max != float('inf') or sigma_min > 0:
        for i in range(n):
            sigmas_clamped[i] = sigmas_clamped[i].clamp(min=sigma_min, max=sigma_max)
    
    h, vel, denoised = 1.0, None, None
    
    for i in range(n):
        sigma = float(sigmas_clamped[i])
        sigma_next = float(sigmas_clamped[i + 1])
        
        if abs(sigma - sigma_next) < 1e-10:
            if callback is not None:
                callback({"i": i, "denoised": denoised if denoised is not None else x, "x": x, "sigma": sigma})
            continue
        
        x, vel, denoised, h = _ARES_STEP(
            model, x, sigma, sigma_next, h, 
            c2=c2, extra_args=extra_args, pbar=callback,
            simple_phi_calc=simple_phi_calc, momentum=momentum, 
            vel=vel, vel_2=None, time=None
        )
        
        if callback is not None:
            callback({"i": i, "denoised": denoised, "x": x, "sigma": sigma_next})
    
    return x

def _rms(z):
    """Root mean square with numerical stability."""
    return torch.sqrt(torch.mean(z.float() ** 2) + 1e-10)

class _RDAStepProxy:
    """RDA wrapper that caches ARES step outputs for acceleration."""
    def __init__(self, n_steps, tau=0.15, gamma=0.75, max_stale=3, w=1.0):
        self.n = max(1, int(n_steps))
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.max_stale = int(max_stale)
        self.w = float(w)
        
        self.i = 0
        self.prev_x = None
        self.prev2_x = None
        self.prev_denoised = None
        self.stale = 0
        self.skipped = 0  # Track number of skipped evaluations

    @torch.no_grad()
    def step(self, model, x, sigma, sigma_next, h, c2, extra_args, pbar, 
             simple_phi_calc, momentum, vel, vel_2, time):
        """Wrapper around _ARES_STEP with RDA acceleration."""
        p = min(1.0, max(0.0, self.i / max(1, self.n)))
        
        # Always compute for first two steps or when stale limit reached
        if self.prev_x is None or self.prev2_x is None or self.stale >= self.max_stale:
            x_out, vel_out, denoised, h_out = _ARES_STEP(
                model, x, sigma, sigma_next, h, c2=c2, extra_args=extra_args, 
                pbar=pbar, simple_phi_calc=simple_phi_calc, momentum=momentum, 
                vel=vel, vel_2=vel_2, time=time
            )
            self.prev2_x = self.prev_x
            self.prev_x = x_out
            self.prev_denoised = denoised
            self.stale = 0
            self.i += 1
            return x_out, vel_out, denoised, h_out
        
        # Check if we can extrapolate
        # Threshold decreases as we progress through sampling
        thr = self.tau * ((1.0 - p + 1e-3) ** self.gamma)
        d = self.prev_x - self.prev2_x
        rel = _rms(d) / max(1e-10, _rms(self.prev_x))
        
        if rel < thr:
            # Extrapolate: x_new ≈ x_prev + w * (x_prev - x_prev2)
            x_out = self.prev_x + self.w * d
            
            # Scale correction to prevent explosion
            prev_scale = _rms(self.prev_x)
            curr_scale = _rms(x_out)
            if curr_scale > 1e-10 and prev_scale > 1e-10:
                scale_factor = torch.clamp(prev_scale / curr_scale, max=2.0).to(x_out)
                x_out = x_out * scale_factor
            
            self.prev2_x = self.prev_x
            self.prev_x = x_out
            self.stale = min(self.stale + 1, self.max_stale)
            self.i += 1
            
            # Reuse previous denoised and vel as approximation
            return x_out, vel, self.prev_denoised, h
        
        # Compute full step
        x_out, vel_out, denoised, h_out = _ARES_STEP(
            model, x, sigma, sigma_next, h, c2=c2, extra_args=extra_args, 
            pbar=pbar, simple_phi_calc=simple_phi_calc, momentum=momentum, 
            vel=vel, vel_2=vel_2, time=time
        )
        self.prev2_x = self.prev_x
        self.prev_x = x_out
        self.prev_denoised = denoised
        self.stale = 0
        self.i += 1
        return x_out, vel_out, denoised, h_out

def sample_ares_rda(
    model,
    x,
    sigmas,
    *,
    extra_args=None,
    callback=None,
    disable=False,
    c2=None,
    simple_phi_calc=False,
    momentum=0.0,
    sigma_min=None,
    sigma_max=None,
    tau=0.25,
    gamma=0.6,
    max_stale=3,
    w=1.0,
):
    # import here to avoid package-relative import issues at module import time
    try:
        from .rda import wrap_with_rda as _wrap_rda
    except Exception:
        from rda import wrap_with_rda as _wrap_rda  # fallback for flat-layout

    model_wrapped = _wrap_rda(model, sigmas, tau=tau, gamma=gamma, max_stale=max_stale, w=w)

    # delegate everything else to the proven ARES loop
    return sample_ares(
        model_wrapped,
        x,
        sigmas,
        extra_args=extra_args,
        callback=callback,
        disable=disable,
        c2=c2,
        simple_phi_calc=simple_phi_calc,
        momentum=momentum,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )

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