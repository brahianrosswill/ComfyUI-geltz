# Quantile Match Scaling

import torch
import torch.fft as fft
from comfy_api.latest import io

def _winsor(x, p=99.9):
    if p is None or p <= 0 or p >= 100:
        return x
    x = x.float()
    lo = torch.quantile(x, (100 - p) / 200.0)
    hi = torch.quantile(x, 1.0 - (100 - p) / 200.0)
    return x.clamp(lo, hi)

def _ifft_real(xf):
    x = fft.ifft2(xf, norm="ortho")
    return x.real

def _freq_masks(h, w, low_cut, high_cut, device):
    yy = torch.fft.fftfreq(h, d=1.0).to(device)
    xx = torch.fft.fftfreq(w, d=1.0).to(device)
    fy, fx = torch.meshgrid(yy, xx, indexing="ij")
    r = torch.sqrt(fx * fx + fy * fy)
    mn = float(min(h, w))
    lc = float(low_cut) / mn
    hc = float(high_cut) / mn
    low = (r <= lc).float()
    mid = ((r > lc) & (r <= hc)).float()
    high = (r > hc).float()
    return low, mid, high

def _bandpass_spatial(x, mask):
    B, C, H, W = x.shape
    x32 = x.float()
    xf = fft.fft2(x32, norm="ortho")
    xf = xf * mask.view(1, 1, H, W)
    return _ifft_real(xf)

def _safe_quantile_computation(x, quantiles, fallback_value=0.0):
    try:
        if torch.numel(x) == 0 or torch.isnan(x).any():
            return [fallback_value] * len(quantiles)
        out = []
        flat = x.flatten()
        for q in quantiles:
            try:
                v = torch.quantile(flat, q)
                if torch.isnan(v) or torch.isinf(v):
                    out.append(fallback_value)
                else:
                    out.append(v)
            except:
                out.append(fallback_value)
        return out
    except:
        return [fallback_value] * len(quantiles)

def _band_quantiles_safe_from_fft(xf, masks, winsor_p, quantiles):
    B, C, H, W = xf.shape
    results = []
    for m in masks:
        xb = _ifft_real(xf * m.view(1, 1, H, W))
        xb = _winsor(xb, winsor_p)
        results.append(_safe_quantile_computation(xb, quantiles))
    return results

def _band_quantiles_safe(x, masks, winsor_p, quantiles):
    results = []
    for m in masks:
        xb = _bandpass_spatial(x, m)
        xb = _winsor(xb, winsor_p)
        results.append(_safe_quantile_computation(xb, quantiles))
    return results

def _fit_linear_map(src_quantiles, tgt_quantiles):
    src = torch.stack([q if torch.is_tensor(q) else torch.tensor(q, device=tgt_quantiles[0].device if len(tgt_quantiles)>0 and torch.is_tensor(tgt_quantiles[0]) else 'cpu') for q in src_quantiles])
    tgt = torch.stack([q if torch.is_tensor(q) else torch.tensor(q, device=src.device) for q in tgt_quantiles])
    n = src.numel()
    sum_x = torch.sum(src)
    sum_y = torch.sum(tgt)
    sum_xy = torch.sum(src * tgt)
    sum_x2 = torch.sum(src * src)
    denom = n * sum_x2 - sum_x * sum_x
    if torch.abs(denom) < 1e-12:
        a = src.new_tensor(1.0)
        b = src.new_tensor(0.0)
    else:
        a = (n * sum_xy - sum_x * sum_y) / denom
        b = (sum_y - a * sum_x) / n
    return a, b

def _select_indices_for_qs(qs_tensor):
    q25 = (qs_tensor - 0.25).abs().argmin()
    q50 = (qs_tensor - 0.50).abs().argmin()
    q75 = (qs_tensor - 0.75).abs().argmin()
    return int(q25), int(q50), int(q75)

def _fit_robust_map_from_qs(src_vals, tgt_vals, qs, eps=1e-12):
    qs_t = torch.tensor(qs, dtype=torch.float32, device=(src_vals[0].device if torch.is_tensor(src_vals[0]) else 'cpu'))
    src = torch.stack([v if torch.is_tensor(v) else torch.tensor(v, device=qs_t.device) for v in src_vals]).to(qs_t)
    tgt = torch.stack([v if torch.is_tensor(v) else torch.tensor(v, device=qs_t.device) for v in tgt_vals]).to(qs_t)
    i25, i50, i75 = _select_indices_for_qs(qs_t)
    iqr_s = src[i75] - src[i25]
    iqr_t = tgt[i75] - tgt[i25]
    safe = torch.where(iqr_s.abs() < eps, torch.tensor(1.0, device=src.device, dtype=src.dtype), iqr_s)
    a = iqr_t / safe
    b = tgt[i50] - a * src[i50]
    return a, b

def _apply_qms(g, a_low, b_low, a_mid, b_mid, a_high, b_high, masks):
    B, C, H, W = g.shape
    g32 = g.float()
    gf = fft.fft2(g32, norm="ortho")
    low, mid, high = masks
    low = low.view(1, 1, H, W).to(gf.dtype).to(gf.device)
    mid = mid.view(1, 1, H, W).to(gf.dtype).to(gf.device)
    high = high.view(1, 1, H, W).to(gf.dtype).to(gf.device)
    a_low = a_low.to(dtype=g32.dtype, device=g32.device)
    b_low = b_low.to(dtype=g32.dtype, device=g32.device)
    a_mid = a_mid.to(dtype=g32.dtype, device=g32.device)
    b_mid = b_mid.to(dtype=g32.dtype, device=g32.device)
    a_high = a_high.to(dtype=g32.dtype, device=g32.device)
    b_high = b_high.to(dtype=g32.dtype, device=g32.device)
    band_scale = a_low.view(1,1,1,1).to(gf.dtype) * low + a_mid.view(1,1,1,1).to(gf.dtype) * mid + a_high.view(1,1,1,1).to(gf.dtype) * high
    gf_scaled = gf * band_scale
    g_scaled = _ifft_real(gf_scaled) + (b_low + b_mid + b_high).view(1,1,1,1)
    return g_scaled.to(g.dtype)

def _adaptive_ema_params(iteration, total_iterations, base_rho=0.8, base_r=1.2):
    progress = iteration / max(1, total_iterations)
    rho = base_rho * (0.5 + 0.5 * (1 - progress))
    r = base_r * (0.8 + 0.4 * progress)
    return rho, r

def _adaptive_freq_cutoffs(h, w, content_complexity=None):
    base_cutoff_low = max(1, min(h, w) // 16)
    base_cutoff_high = max(base_cutoff_low + 1, min(h, w) // 4)
    if content_complexity is not None:
        c = 0.5 + content_complexity
        base_cutoff_low = int(base_cutoff_low * c)
        base_cutoff_high = int(base_cutoff_high * (2 - c))
    return base_cutoff_low, base_cutoff_high

def _adaptive_quantiles(x, num_quantiles=7):
    x_flat = x.flatten()
    m = torch.mean(x_flat)
    s = torch.std(x_flat)
    if s == 0:
        base = torch.linspace(0.1, 0.9, num_quantiles)
    else:
        skewness = torch.mean(((x_flat - m) / (s + 1e-12)) ** 3)
        if abs(skewness) < 0.1:
            base = torch.linspace(0.1, 0.9, num_quantiles)
        elif skewness > 0:
            base = torch.linspace(0.05, 0.85, num_quantiles)
        else:
            base = torch.linspace(0.15, 0.95, num_quantiles)
    extra = torch.tensor([0.25, 0.5, 0.75])
    all_q = torch.cat([base, extra]).clamp(1e-4, 1 - 1e-4)
    all_q = torch.unique(all_q).sort().values
    return [float(v.item()) for v in all_q]

def _dynamic_rescale(cfg_value, base_rescale=0.75):
    import math
    k = 0.6
    x0 = 5.0
    s = 1.0 / (1.0 + math.exp(-k * (float(cfg_value) - x0)))
    return min(base_rescale + 0.2 * s, 0.95)

def _safe_rms(x):
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x.pow(2).mean().sqrt()

class QMS(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QMS",
            display_name="Quantile Match Scaling",
            category="_for_testing",
            description="Tames CFG overdrive by matching per-band quantile distributions to the conditional.",
            inputs=[
                io.Model.Input("model"),
                io.Float.Input("rescale", default=0.75, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Model.Output(),
            ],
            is_experimental=False,
        )

    @classmethod
    def execute(cls, model, rescale):
        state = {
            "ema_a_low": None, "ema_b_low": None,
            "ema_a_mid": None, "ema_b_mid": None,
            "ema_a_high": None, "ema_b_high": None,
            "iter": 0, "total": None,
            "mask_key": None, "masks": None
        }
        def custom_pre_cfg(args):
            conds_out = args["conds_out"]
            if len(conds_out) <= 1 or None in args["conds"][:2]:
                return conds_out
            cond = conds_out[0]
            uncond = conds_out[1]
            if cond is None or uncond is None:
                return conds_out
            B, C, H, W = cond.shape
            if H < 16 or W < 16:
                return conds_out
            w = float(args.get("cfg", 1.0))
            g = cond - uncond
            device = cond.device
            if state["total"] is None:
                total = None
                sigmas = args.get("sigmas", None)
                if isinstance(sigmas, (list, tuple)):
                    total = len(sigmas)
                total = total if total is not None else int(args.get("steps", 0)) or 50
                state["total"] = total
            rho, r = _adaptive_ema_params(state["iter"], state["total"])
            low_cut, high_cut = _adaptive_freq_cutoffs(H, W, None)
            key = (H, W, low_cut, high_cut, device)
            if state["mask_key"] != key:
                state["masks"] = _freq_masks(H, W, low_cut, high_cut, device)
                state["mask_key"] = key
            masks = state["masks"]
            xcfg_temp = uncond + w * g
            qs = _adaptive_quantiles(cond, num_quantiles=7)
            xf_cond = fft.fft2(cond.float(), norm="ortho")
            xf_xcfg = fft.fft2(xcfg_temp.float(), norm="ortho")
            cond_q = _band_quantiles_safe_from_fft(xf_cond, masks, 99.9, qs)
            xcfg_q = _band_quantiles_safe_from_fft(xf_xcfg, masks, 99.9, qs)
            a_low0, b_low0 = _fit_robust_map_from_qs(xcfg_q[0], cond_q[0], qs)
            a_mid0, b_mid0 = _fit_robust_map_from_qs(xcfg_q[1], cond_q[1], qs)
            a_high0, b_high0 = _fit_robust_map_from_qs(xcfg_q[2], cond_q[2], qs)
            one = torch.tensor(1.0, device=cond.device, dtype=cond.dtype)
            zero = torch.tensor(0.0, device=cond.device, dtype=cond.dtype)
            rescale_eff = _dynamic_rescale(w, base_rescale=rescale)
            a_low = one + rescale_eff * (a_low0.to(device=cond.device, dtype=cond.dtype) - one)
            b_low = zero + rescale_eff * b_low0.to(device=cond.device, dtype=cond.dtype)
            a_mid = one + rescale_eff * (a_mid0.to(device=cond.device, dtype=cond.dtype) - one)
            b_mid = zero + rescale_eff * b_mid0.to(device=cond.device, dtype=cond.dtype)
            a_high = one + rescale_eff * (a_high0.to(device=cond.device, dtype=cond.dtype) - one)
            b_high = zero + rescale_eff * b_high0.to(device=cond.device, dtype=cond.dtype)
            if state["ema_a_low"] is None:
                a_low = a_low.clamp_min(0.2)
                a_mid = a_mid.clamp_min(0.2)
                a_high = a_high.clamp_min(0.2)
                state["ema_a_low"], state["ema_b_low"] = a_low.detach(), b_low.detach()
                state["ema_a_mid"], state["ema_b_mid"] = a_mid.detach(), b_mid.detach()
                state["ema_a_high"], state["ema_b_high"] = a_high.detach(), b_high.detach()
            else:
                state["ema_a_low"] = rho * state["ema_a_low"] + (1 - rho) * a_low
                state["ema_b_low"] = rho * state["ema_b_low"] + (1 - rho) * b_low
                state["ema_a_mid"] = rho * state["ema_a_mid"] + (1 - rho) * a_mid
                state["ema_b_mid"] = rho * state["ema_b_mid"] + (1 - rho) * b_mid
                state["ema_a_high"] = rho * state["ema_a_high"] + (1 - rho) * a_high
                state["ema_b_high"] = rho * state["ema_b_high"] + (1 - rho) * b_high
                a_low = torch.clamp(a_low, state["ema_a_low"] / r, state["ema_a_low"] * r)
                b_low = torch.clamp(b_low, state["ema_b_low"] / r, state["ema_b_low"] * r)
                a_mid = torch.clamp(a_mid, state["ema_a_mid"] / r, state["ema_a_mid"] * r)
                b_mid = torch.clamp(b_mid, state["ema_b_mid"] / r, state["ema_b_mid"] * r)
                a_high = torch.clamp(a_high, state["ema_a_high"] / r, state["ema_a_high"] * r)
                b_high = torch.clamp(b_high, state["ema_b_high"] / r, state["ema_b_high"] * r)
            g_scaled = _apply_qms(g, a_low, b_low, a_mid, b_mid, a_high, b_high, masks)
            base_rms = _safe_rms(g)
            scaled_rms = _safe_rms(g_scaled)
            cap = 1.0 + 0.5 * rescale_eff
            scale = torch.clamp(cap * base_rms / (scaled_rms + 1e-12), max=1.0)
            g_scaled = g_scaled * scale
            g_scaled = torch.nan_to_num(g_scaled, nan=0.0, posinf=0.0, neginf=0.0)
            cond_new = uncond + g_scaled
            cond_new = torch.nan_to_num(cond_new, nan=0.0, posinf=0.0, neginf=0.0).to(cond.dtype)
            state["iter"] += 1
            return [cond_new, uncond] + conds_out[2:]
        m = model.clone()
        m.set_model_sampler_pre_cfg_function(custom_pre_cfg)
        return io.NodeOutput(m)

NODE_CLASS_MAPPINGS = {"QMS": QMS}
NODE_DISPLAY_NAME_MAPPINGS = {"QMS": "Quantile Match Scaling"}
