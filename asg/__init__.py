# Attention Shuffle Guidance

import math
import torch
import torch.nn.functional as F

def _safe(x):
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

class _SDPAPerturb:
    def __init__(self, s, window=16):
        self.s = float(s)
        self.window = int(max(1, window))
        self._orig = None

    def __enter__(self):
        self._orig = F.scaled_dot_product_attention
        s = self.s
        w = self.window
        orig = self._orig

        def _window_idx(L, win, device):
            if L <= 1 or win <= 1:
                return torch.arange(L, device=device)
            idx_parts = []
            for start in range(0, L, win):
                end = min(start + win, L)
                seg = torch.randperm(end - start, device=device) + start
                idx_parts.append(seg)
            return torch.cat(idx_parts, dim=0)

        def patched(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
            if s <= 0.0:
                return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
            L = v.shape[-2]
            idx = _window_idx(L, min(w, L), v.device)
            v_shuffled = v.index_select(-2, idx)
            v_pert = (1.0 - s) * v + s * v_shuffled
            v_pert = _safe(v_pert)
            if q.is_cuda:
                with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=True):
                    return orig(q, k, v_pert, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
            return orig(q, k, v_pert, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)

        F.scaled_dot_product_attention = patched
        return self

    def __exit__(self, exc_type, exc, tb):
        F.scaled_dot_product_attention = self._orig


def _coerce_params(params):
    x = params.get("input", None)
    t = params.get("timestep", None)
    c = params.get("c", {})
    if x is None or t is None:
        raise ValueError("Missing required keys: input and timestep")
    if not isinstance(c, dict):
        c = {}
    if not torch.is_tensor(t):
        t = torch.as_tensor(t, dtype=torch.float32, device=x.device)
    return {"input": x, "timestep": t, "c": c}


def _eps(unet_apply, params):
    p = _coerce_params(params)
    with torch.no_grad():
        out = unet_apply(x=p["input"], t=p["timestep"], **p["c"])
    return _safe(out)


def _rescale_delta(eps, delta, rescale):
    if rescale <= 0.0:
        return _safe(delta)
    eps = _safe(eps)
    delta = _safe(delta)
    eps_std = eps.std(unbiased=False)
    delta_std = delta.std(unbiased=False)
    scale_factor = (eps_std / (delta_std + 1e-12)) * float(rescale)
    scale_factor = torch.nan_to_num(scale_factor, nan=0.0, posinf=1.0, neginf=0.0)
    scale_factor = float(min(1.0, max(0.0, scale_factor.item() if torch.is_tensor(scale_factor) else scale_factor)))
    return _safe(delta * scale_factor)


def _proj_out(delta, eps):
    delta = _safe(delta)
    eps = _safe(eps)
    dims = tuple(range(1, delta.ndim))
    num = (delta * eps).sum(dim=dims, keepdim=True)
    den = (eps * eps).sum(dim=dims, keepdim=True).clamp_min(1e-12)
    proj = delta - (num / den) * eps
    return _safe(proj)


def _rms(x):
    x = _safe(x).float()
    dims = tuple(range(1, x.ndim))
    return _safe(x.pow(2).mean(dim=dims, keepdim=True).sqrt())


def _rms_clamp(delta, ref, tau=1.0):
    ref_r = _rms(ref)
    del_r = _rms(delta)
    gain = (float(tau) * ref_r) / (del_r + 1e-12)
    gain = torch.nan_to_num(gain, nan=0.0, posinf=1.0, neginf=0.0)
    gain = gain.clamp(max=1.0)
    return _safe(delta * gain)


def _apply_asg(unet_apply, params, s, rescale):
    s = float(s)
    if not math.isfinite(s):
        s = 0.0
    rescale = float(rescale)
    if not math.isfinite(rescale):
        rescale = 0.0
    s = max(0.0, s)
    rescale = max(0.0, rescale)

    base = _eps(unet_apply, params)
    with _SDPAPerturb(s, window=16):
        guided = _eps(unet_apply, params)

    delta = _safe(base - guided)
    delta = _rescale_delta(base, delta, rescale)
    delta = _proj_out(delta, base)
    delta = _rms_clamp(delta, base, tau=1.0)
    out = _safe(base + s * delta)
    return out


class _ASGWrapper:
    def __init__(self, strength, rescale):
        self.s = float(strength)
        self.rescale = float(rescale)

    def __call__(self, unet_apply, params):
        return _apply_asg(unet_apply, params, self.s, self.rescale)


class AttentionShuffleGuidanceModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.01}),
                "rescale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/patches"

    def patch(self, model, strength, rescale):
        m = model.clone()
        m.set_model_unet_function_wrapper(_ASGWrapper(strength, rescale))
        return (m,)


NODE_CLASS_MAPPINGS = {
    "AttentionShuffleGuidanceModelPatch": AttentionShuffleGuidanceModelPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AttentionShuffleGuidanceModelPatch": "Attention Shuffle Guidance",
}
