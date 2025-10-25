import math
from collections import OrderedDict
import torch
import torch.nn.functional as F


def _safe(x):
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


_GLOBAL_INDEX_CACHE = OrderedDict()
_GLOBAL_INDEX_CACHE_CAP = 64


def _device_key(t):
    d = t.device
    if d.type == "cuda":
        return ("cuda", d.index)
    return (d.type, None)


def _dynamic_window(L):
    return int(max(1, round(L ** 0.5)))


def _cached_window_idx(L, win, seed, device):
    key = ("idx", int(L), int(win), int(seed), _device_key(torch.empty(0, device=device)))
    if key in _GLOBAL_INDEX_CACHE:
        _GLOBAL_INDEX_CACHE.move_to_end(key)
        return _GLOBAL_INDEX_CACHE[key]
    if L <= 1 or win <= 1:
        idx = torch.arange(L, device=device)
    else:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        parts = []
        for start in range(0, L, win):
            end = min(start + win, L)
            seg = torch.randperm(end - start, generator=g, device=device) + start
            parts.append(seg)
        idx = torch.cat(parts, dim=0)
    _GLOBAL_INDEX_CACHE[key] = idx
    if len(_GLOBAL_INDEX_CACHE) > _GLOBAL_INDEX_CACHE_CAP:
        _GLOBAL_INDEX_CACHE.popitem(last=False)
    return idx


class _SDPAPerturb:
    def __init__(self, s, window=0, seed=42):
        self.s = float(s)
        self.window = int(window)
        self.seed = int(seed)
        self._orig = None

    def __enter__(self):
        self._orig = F.scaled_dot_product_attention
        s = self.s
        win_cfg = self.window
        seed = self.seed
        orig = self._orig

        def patched(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
            if s <= 0.0:
                return orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
            L = v.shape[-2]
            w = win_cfg if win_cfg > 0 else _dynamic_window(L)
            idx = _cached_window_idx(L, min(w, L), seed, v.device)
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


def _proj_out(delta, eps):
    delta = _safe(delta)
    eps = _safe(eps)
    dims = tuple(range(1, delta.ndim))
    num = (delta * eps).sum(dim=dims, keepdim=True)
    den = (eps * eps).sum(dim=dims, keepdim=True).clamp_min(1e-12)
    return _safe(delta - (num / den) * eps)


def _rms(x):
    x = _safe(x).float()
    dims = tuple(range(1, x.ndim))
    return _safe(x.pow(2).mean(dim=dims, keepdim=True).sqrt())


def _rms_clamp(delta, ref, tau=0.7):
    ref_r = _rms(ref)
    del_r = _rms(delta)
    gain = (float(tau) * ref_r) / (del_r + 1e-12)
    gain = torch.nan_to_num(gain, nan=0.0, posinf=1.0, neginf=0.0).clamp(max=1.0)
    return _safe(delta * gain)


def _per_channel_rms(x):
    x = _safe(x).float()
    if x.ndim < 2:
        return torch.ones_like(x)
    dims = tuple(range(2, x.ndim))
    return _safe(x.pow(2).mean(dim=dims, keepdim=True).sqrt())


def _rescale_delta_advanced(eps, delta, rescale):
    if rescale <= 0.0:
        return _safe(delta)
    eps = _safe(eps)
    delta = _safe(delta)
    eps_r = _per_channel_rms(eps)
    del_r = _per_channel_rms(delta)
    sf = (eps_r / (del_r + 1e-12)) * float(rescale)
    sf = torch.nan_to_num(sf, nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)
    return _safe(delta * sf)


def _timestep_ratio(t):
    tt = torch.as_tensor(t, dtype=torch.float32)
    m = tt.detach().abs().max()
    if not torch.isfinite(m) or m <= 0:
        return 0.0
    r = (tt / m).mean().item()
    r = 0.0 if not math.isfinite(r) else r
    return float(max(0.0, min(1.0, r)))


def _apply_asg(unet_apply, params, s, rescale, seed, window):
    s = 0.0 if (not math.isfinite(s) or s < 0.0) else float(s)
    rescale = 0.0 if (not math.isfinite(rescale) or rescale < 0.0) else float(rescale)

    base = _eps(unet_apply, params)
    if s == 0.0:
        return base

    # Step ratio and effective strength
    ratio = _timestep_ratio(params.get("timestep", 0.0))
    s_eff = s * (ratio ** 0.7)
    if s_eff <= 0.0:
        return base

    # Derive a stable-but-changing tag from the current diffusion step to decorrelate shuffles
    try:
        tt = torch.as_tensor(params.get("timestep", 0.0), dtype=torch.float32)
        step_tag = int(tt.detach().abs().max().item())
    except Exception:
        step_tag = 0

    # Number of ASG evaluations
    N_min, N_max = 1, 4
    Nf = N_min + (N_max - N_min) * (1.0 - ratio)
    N = int(max(N_min, min(N_max, round(Nf))))

    mean = None
    for i in range(N):
        # Seed now varies across i AND across diffusion steps
        si = int(seed + step_tag * 9973 + i * 2654435761)
        with _SDPAPerturb(s_eff, window=window, seed=si):
            y = _eps(unet_apply, params)
        if mean is None:
            mean = y
        else:
            k = float(i + 1)
            mean = mean + (y - mean) / k

    guided = mean if mean is not None else base
    delta = _safe(base - guided)
    delta = _rescale_delta_advanced(base, delta, rescale)
    delta = _proj_out(delta, base)
    delta = _rms_clamp(delta, base, tau=0.7)
    return _safe(base + s_eff * delta)

class _ASGWrapper:
    def __init__(self, strength, rescale, seed, window):
        self.s = float(strength)
        self.rescale = float(rescale)
        self.seed = int(seed)
        self.window = int(window)

    def __call__(self, unet_apply, params):
        return _apply_asg(unet_apply, params, self.s, self.rescale, self.seed, self.window)


class AttentionShuffleGuidanceModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "strength": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05}),
                "rescale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                # 0 => dynamic sqrt(L); 1 => no shuffle; >=L => global shuffle
                "window": ("INT", {"default": 4, "min": 0, "max": 1_000_000}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/patches"

    def patch(self, model, strength, rescale, seed, window):
        m = model.clone()
        m.set_model_unet_function_wrapper(_ASGWrapper(strength, rescale, seed, window))
        return (m,)



NODE_CLASS_MAPPINGS = {
    "AttentionShuffleGuidanceModelPatch": AttentionShuffleGuidanceModelPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AttentionShuffleGuidanceModelPatch": "Attention Shuffle Guidance",
}
