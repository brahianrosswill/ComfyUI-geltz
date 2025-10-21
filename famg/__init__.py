# Flow-Aligned Mask Guidance

import math
import torch
import torch.nn.functional as F

_mask_cache = {}

def _long_range_soft_bias(n, r_frac, device, dtype):
    s = int(math.sqrt(n))
    if s * s != n:
        return None
    key = (n, float(r_frac), 6.0)
    if key not in _mask_cache:
        yy, xx = torch.meshgrid(torch.arange(s), torch.arange(s), indexing="ij")
        coords = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=1).float()
        d = torch.cdist(coords, coords, p=2)
        r = max(1.0, r_frac * float(s))
        pen = ((torch.clamp(r - d, min=0.0) / r) ** 2) * 6.0
        bias = -pen
        _mask_cache[key] = bias.unsqueeze(0)
    return _mask_cache[key].to(device=device, dtype=dtype)

class _SoftmaxHook:
    def __init__(self, lam=0.6, temp=1.4):
        self.lam = lam
        self.temp = temp
        self._orig = None

    def __enter__(self):
        self._orig = F.softmax
        def hooked(input, dim=None, _stacklevel=3, dtype=None):
            if dim == -1 and input.dim() == 3 and input.shape[-1] == input.shape[-2]:
                x = input / self.temp
                A = self._orig(x, dim=dim, dtype=dtype)
                N = A.shape[-1]
                flat = A.reshape(-1, N, N)
                eps = 1e-12
                H = -(flat * (flat.clamp_min(eps).log())).sum(dim=-1)
                H = H / math.log(max(N, 2))
                thresh = torch.quantile(H, 0.5, dim=1, keepdim=True)
                M = (H >= thresh).to(flat.dtype).unsqueeze(-1)
                eye = torch.eye(N, device=flat.device, dtype=flat.dtype).unsqueeze(0).expand(flat.size(0), -1, -1)
                flat = (1.0 - self.lam) * flat + self.lam * (M * eye + (1.0 - M) * flat)
                return flat.reshape_as(A)
            return self._orig(input, dim=dim, dtype=dtype)
        F.softmax = hooked
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._orig is not None:
            F.softmax = self._orig
        self._orig = None

def _norm2(t):
    return torch.sqrt((t.float() * t.float()).sum() + 1e-12)

def _orth(delta, base):
    b2 = (base.float() * base.float()).sum()
    if b2 <= 0:
        return delta
    proj = (delta.float() * base.float()).sum() / b2
    return delta - proj * base

def _clip_tr(delta):
    x = delta.float().flatten()
    med = x.median()
    mad = (x - med).abs().median() + 1e-12
    lo = med - 3.0 * mad
    hi = med + 3.0 * mad
    return delta.clamp(min=lo.item(), max=hi.item())

def _hann2d(h, w, device, dtype):
    y = torch.hann_window(h, periodic=False, device=device, dtype=dtype)
    x = torch.hann_window(w, periodic=False, device=device, dtype=dtype)
    return y[:, None] * x[None, :]

def _phase_corr_shift(prev, curr):
    B, _, H, W = curr.shape
    shifts = []
    eps = 1e-8
    for b in range(B):
        A = prev[b, 0]
        Bc = curr[b, 0]
        FA = torch.fft.rfftn(A)
        FB = torch.fft.rfftn(Bc)
        R = FA * torch.conj(FB)
        R = R / (R.abs() + eps)
        r = torch.fft.irfftn(R, s=A.shape)
        idx = torch.argmax(r)
        iy = int(idx // W)
        ix = int(idx % W)
        if iy > H // 2:
            dy = iy - H
        else:
            dy = iy
        if ix > W // 2:
            dx = ix - W
        else:
            dx = ix
        shifts.append((dy, dx))
    dy = torch.tensor([s[0] for s in shifts], device=curr.device, dtype=curr.dtype)
    dx = torch.tensor([s[1] for s in shifts], device=curr.device, dtype=curr.dtype)
    return dy, dx

def _build_sliding_mask(H, W, k, s, offy, offx, device, dtype):
    win = _hann2d(k, k, device, dtype)
    acc = torch.zeros((H, W), device=device, dtype=dtype)
    start_y = int(math.floor(offy)) - k
    start_x = int(math.floor(offx)) - k
    y = start_y
    while y < H:
        x = start_x
        while x < W:
            y0 = y
            x0 = x
            ya = max(0, y0)
            yb = min(H, y0 + k)
            xa = max(0, x0)
            xb = min(W, x0 + k)
            if ya < yb and xa < xb:
                wy0 = ya - y0
                wx0 = xa - x0
                acc[ya:yb, xa:xb] += win[wy0:wy0 + (yb - ya), wx0:wx0 + (xb - xa)]
            x += s
        y += s
    ssum = acc.sum().clamp_min(1e-8)
    acc = acc * (float(H * W) / float(ssum))
    return acc[None, None, :, :]

class _State:
    def __init__(self, beta=0.7):
        self.beta = beta
        self.m = None
        self.shape = None
        self.prev_low = None
        self.offy = 0.0
        self.offx = 0.0
        self.k = None
        self.s = None
        self.delta_o = None

    def reset(self):
        self.m = None
        self.shape = None
        self.prev_low = None
        self.offy = 0.0
        self.offx = 0.0
        self.k = None
        self.s = None
        self.delta_o = None

class FAMG_Patch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "modelpatch"

    def apply(self, model, strength):
        radius_frac = 0.25
        window_frac = 0.75
        overlap = 0.5
        flow_gain = 1.0
        m = model.clone()
        lam = 0.6
        temp = 1.4
        beta = 0.7
        kappa = 1.2
        smin, smax = 0.2, 25.0
        tau = 0.35
        state = _State(beta=beta)

        def attn_override(default_attn, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
            if q.shape[1] != k.shape[1]:
                return default_attn(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
            n = q.shape[1]
            bias = _long_range_soft_bias(n, radius_frac, q.device, q.dtype)
            if bias is None:
                return default_attn(q, k, v, heads, mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)
            msk = bias if mask is None else mask + bias
            return default_attn(q, k, v, heads, mask=msk, attn_precision=attn_precision, skip_reshape=skip_reshape, skip_output_reshape=skip_output_reshape, **kwargs)

        def make_mask_and_update_flow(x_full):
            B, C, H, W = x_full.shape
            if state.k is None or state.s is None or state.shape != (B, C, H, W):
                k = max(8, int(window_frac * min(H, W)))
                s = max(1, int((1.0 - overlap) * k))
                state.k = k
                state.s = s
                state.delta_o = (s // 2, s // 2)
                state.offy = 0.0
                state.offx = 0.0
                state.shape = (B, C, H, W)
                state.prev_low = None
            scale = 8
            h = max(8, H // scale)
            w = max(8, W // scale)
            low = torch.nn.functional.interpolate(x_full.float().mean(dim=1, keepdim=True), size=(h, w), mode="bilinear", align_corners=False)
            dy = torch.tensor(0.0, device=x_full.device, dtype=x_full.dtype)
            dx = torch.tensor(0.0, device=x_full.device, dtype=x_full.dtype)
            if state.prev_low is not None and state.prev_low.shape[-2:] == (h, w):
                dy_l, dx_l = _phase_corr_shift(state.prev_low, low)
                dy = dy_l.mean() * (H / h) * float(flow_gain)
                dx = dx_l.mean() * (W / w) * float(flow_gain)
            state.prev_low = low.detach()
            state.offy = (state.offy + state.delta_o[0] + float(dy.item())) % max(state.s, 1)
            state.offx = (state.offx + state.delta_o[1] + float(dx.item())) % max(state.s, 1)
            mask = _build_sliding_mask(H, W, state.k, state.s, state.offy, state.offx, x_full.device, x_full.dtype)
            return mask

        def unet_wrapper(apply_model_method, options_dict):
            c = options_dict.get("c", {})
            x = options_dict["input"]
            t = options_dict["timestep"]
            if float(strength) <= 0.0:
                state.reset()
                return apply_model_method(x, t, **c)
            to = c.get("transformer_options", {})
            mask = make_mask_and_update_flow(x)
            c0 = dict(c)
            c0["transformer_options"] = dict(to)
            eps0 = apply_model_method(x, t, **c0)
            eps0 = torch.nan_to_num(eps0, nan=0.0, posinf=0.0, neginf=0.0)
            c1 = dict(c)
            to1 = dict(to)
            to1["optimized_attention_override"] = attn_override
            c1["transformer_options"] = to1
            with _SoftmaxHook(lam=lam, temp=temp):
                epslr = apply_model_method(x, t, **c1)
            epslr = torch.nan_to_num(epslr, nan=0.0, posinf=0.0, neginf=0.0)
            b = eps0.shape[0]
            u = eps0.float().flatten(1)
            v = epslr.float().flatten(1)
            n0 = u.norm(dim=1).clamp_min(1e-8)
            nv = v.norm(dim=1).clamp_min(1e-8)
            dot = (u * v).sum(1)
            a = dot / (n0.square())
            cos = (dot / (n0 * nv)).clamp(-1.0, 1.0)
            delta = epslr - a.view(b, 1, 1, 1).to(eps0.dtype) * eps0
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            dn = delta.float().flatten(1).norm(dim=1).clamp_min(1e-8)
            s_ang = torch.sqrt((1.0 - cos.square()).clamp_min(0.0))
            s_clip = (tau * n0 / dn).clamp_max(1.0)
            w_lrmg = (float(strength) * s_ang * s_clip).view(b, 1, 1, 1).to(eps0.dtype)
            delta = _orth(delta, eps0)
            delta = _clip_tr(delta)
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
            r = (_norm2(delta) / (_norm2(eps0) + 1e-12)).item()
            if not math.isfinite(r):
                r = 0.0
            s_pag = max(smin, min(smax, kappa * r))
            if state.m is None or state.m.shape != delta.shape or not torch.isfinite(state.m).all():
                state.m = delta.detach()
            else:
                state.m = state.beta * state.m + (1.0 - state.beta) * delta.detach()
            w_map = torch.nan_to_num(w_lrmg * mask.to(eps0.dtype), nan=0.0, posinf=0.0, neginf=0.0)
            out = eps0 + w_map * delta + s_pag * state.m
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            return out

        m.set_model_unet_function_wrapper(unet_wrapper)
        return (m,)

NODE_CLASS_MAPPINGS = {
    "FAMG_Patch": FAMG_Patch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FAMG_Patch": "Flow-Aligned Mask Guidance",
}
