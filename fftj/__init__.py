# FFT Jitter

import math
import torch

class FFTJ:
    def __init__(self, strength=0.9):
        self.strength = float(max(0.0, min(1.0, strength)))
        self.cutoff = 0.1

    def _seq_dim(self, x):
        return 2 if x.dim() == 4 else 1 if x.dim() == 3 else None

    def _hp_window(self, steps, device):
        i = torch.arange(steps, device=device, dtype=torch.float32)
        start = int(self.cutoff * steps)
        denom = max(1, steps - start)
        t = torch.clamp((i - start) / denom, 0.0, 1.0)
        return torch.sin(0.5 * math.pi * t) ** 2

    def _sched(self, extra):
        if not isinstance(extra, dict):
            return 1.0
        if "timestep" in extra:
            ts = float(extra["timestep"])
            r = 1.0 - max(0.0, min(1000.0, ts)) / 1000.0
            return 0.10 + 0.90 * 4.0 * r * (1.0 - r)
        return 1.0

    def _gate(self, s):
        g = 0.75 * self.strength * s
        return float(max(0.0, min(1.0, g)))

    def _jitter(self, x, extra):
        if self.strength <= 0.0:
            return x
        dim = self._seq_dim(x)
        if dim is None or x.size(dim) < 4:
            return x
        y = x.float()
        n = y.size(dim)
        yf = torch.fft.rfft(y, dim=dim)
        steps = yf.shape[dim]
        hp = self._hp_window(steps, y.device)
        view = [1] * yf.dim()
        view[dim] = steps
        hp = hp.view(view)
        s = self._sched(extra)
        amp = 1.0 + (0.50 * self.strength * s) * hp
        phi_std = 1.25 * self.strength * s
        shape = list(yf.shape)
        shape[dim] = 1
        phi = phi_std * hp * torch.randn(shape, device=y.device, dtype=torch.float32)
        mult = torch.polar(torch.ones_like(yf.real), phi)
        yj = torch.fft.irfft(yf * amp * mult, n=n, dim=dim)
        var_eps = 1e-8
        u0 = torch.mean(y, dim=dim, keepdim=True)
        u1 = torch.mean(yj, dim=dim, keepdim=True)
        yj = yj - u1 + u0
        v0 = torch.var(y, dim=dim, unbiased=False, keepdim=True)
        v1 = torch.var(yj, dim=dim, unbiased=False, keepdim=True)
        scale = torch.sqrt((v0 + var_eps) / (v1 + var_eps))
        yj = yj * scale
        g = self._gate(s)
        out = (y * (1.0 - g) + yj * g).to(x.dtype)
        return out

    def patch(self, q, k, v, extra_options=None):
        is_cross = bool(extra_options.get("is_cross", False)) if isinstance(extra_options, dict) else False
        if self.strength <= 0.0 or is_cross:
            return q, k, v
        v = self._jitter(v, extra_options)
        return q, k, v

class FFTJModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})}}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model/patches"
    OUTPUT_NODE = False
    def apply(self, model, strength):
        algo = FFTJ(strength)
        def _hook(q, k, v, extra_options): return algo.patch(q, k, v, extra_options)
        m = model.clone()
        m.set_model_attn1_patch(_hook)
        if hasattr(m, "set_model_attn2_patch"):
            m.set_model_attn2_patch(_hook)
        return (m,)

NODE_CLASS_MAPPINGS = {"FFTJModelPatch": FFTJModelPatch}
NODE_DISPLAY_NAME_MAPPINGS = {"FFTJModelPatch": "FFT Jitter"}
