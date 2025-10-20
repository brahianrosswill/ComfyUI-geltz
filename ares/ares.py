from __future__ import annotations
import torch

@torch.no_grad()
def _sigma_batch(sigma, x):
    if torch.is_tensor(sigma):
        if sigma.numel() == 1:
            val = float(sigma.item())
        else:
            s = sigma.to(device=x.device, dtype=x.dtype)
            if s.shape == (x.shape[0],):
                return s
            val = float(s.flatten()[0].item())
    else:
        val = float(sigma)
    return torch.full((x.shape[0],), val, device=x.device, dtype=x.dtype)

def _expand_sigma(s_b, x):
    return s_b.view(s_b.shape[0], *([1] * (x.ndim - 1)))

def _alpha_sigma_bar(s_b):
    one = torch.ones_like(s_b)
    denom = torch.sqrt(one + s_b * s_b)
    a = one / denom
    sbar = s_b / denom
    return a, sbar

@torch.no_grad()
def _pred_to_eps(x, s_b, pred, pred_type="auto"):
    s = _expand_sigma(s_b, x)
    if pred_type == "eps":
        eps = pred
        x0 = x - s * eps
        return eps, x0
    if pred_type == "x0":
        x0 = pred
        eps = (x - x0) / (s + 1e-12)
        return eps, x0
    if pred_type == "v":
        a, sbar = _alpha_sigma_bar(s_b)
        a = _expand_sigma(a, x)
        sbar = _expand_sigma(sbar, x)
        x0 = a * x - sbar * pred
        eps = (x - x0) / (s + 1e-12)
        return eps, x0
    eps_eps = pred
    x0_eps = x - s * eps_eps
    eps_x0 = (x - pred) / (s + 1e-12)
    x0_x0 = pred
    a_b, sbar_b = _alpha_sigma_bar(s_b)
    a = _expand_sigma(a_b, x)
    sbar = _expand_sigma(sbar_b, x)
    x0_v = a * x - sbar * pred
    eps_v = (x - x0_v) / (s + 1e-12)
    def _score(z):
        zf = z.view(z.shape[0], -1)
        std = zf.std(dim=1).clamp_min(1e-6)
        return (std.log()).abs().mean()
    scores = torch.stack([_score(eps_eps), _score(eps_x0), _score(eps_v)])
    idx = int(torch.argmin(scores).item())
    if idx == 0:
        return eps_eps, x0_eps
    if idx == 1:
        return eps_x0, x0_x0
    return eps_v, x0_v

def _dt_from_raw(x, sigma, sigma_next):
    if torch.is_tensor(sigma):
        if sigma.numel() == 1:
            s0 = float(sigma.item())
        else:
            s0 = float(sigma.flatten()[0].item())
    else:
        s0 = float(sigma)
    if torch.is_tensor(sigma_next):
        if sigma_next.numel() == 1:
            s1 = float(sigma_next.item())
        else:
            s1 = float(sigma_next.flatten()[0].item())
    else:
        s1 = float(sigma_next)
    dt = torch.tensor(s1 - s0, device=x.device, dtype=x.dtype)
    return dt.view(1, *([1] * (x.ndim - 1))).expand(x.shape[0], *([1] * (x.ndim - 1)))

@torch.no_grad()
def _heun_step(model, x, sigma_eval, sigma_next_eval, dt_raw, *, extra_args=None, momentum=0.0, vel=None):
    if extra_args is None:
        extra_args = {}
    s_cur_b = _sigma_batch(sigma_eval, x).clamp_min(0.0)
    s_nxt_b = _sigma_batch(sigma_next_eval, x).clamp_min(0.0)
    dt = dt_raw
    near_zero_dt = torch.all(dt.abs() <= 1e-12)
    if near_zero_dt:
        dt = _expand_sigma(s_nxt_b - s_cur_b, x)
        same_eval = torch.all(dt.abs() <= 1e-12)
        if same_eval:
            sign = torch.sign(_expand_sigma(s_cur_b, x)).clamp_min(0.0) * -1.0 + (s_nxt_b > s_cur_b).to(x.dtype)
            dt = sign + 0.0
            dt = dt * (1e-5 + s_cur_b.abs() * 1e-5)
    out = model(
