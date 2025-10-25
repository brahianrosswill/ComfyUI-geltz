import math, hashlib, torch
from collections import OrderedDict

try:
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass

class LRU:
    def __init__(self, capacity): self.capacity, self.od = int(capacity), OrderedDict()
    def get(self, key):
        v = self.od.get(key)
        if v is not None: self.od.move_to_end(key)
        return v
    def put(self, key, value):
        self.od[key] = value; self.od.move_to_end(key)
        if len(self.od) > self.capacity: self.od.popitem(last=False)

def _tri_smooth(v, iters=2):
    for _ in range(iters): v = (v + torch.roll(v, 1, 0) + torch.roll(v, -1, 0)) / 3.0
    return v

def _smooth_tokens(x, iters=1):
    for _ in range(iters): x = (x + torch.roll(x, 1, 1) + torch.roll(x, -1, 1)) / 3.0
    return x

def _dev_key(dev): return f"{dev.type}:{getattr(dev, 'index', None)}"

def _stable_seed(base_seed, *parts):
    m = hashlib.sha256(); m.update(str(int(base_seed)).encode())
    for p in parts: m.update(str(p).encode())
    return int.from_bytes(m.digest()[:8], "little")

class TWS:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}), "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})}}
    RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY = ("MODEL",), ("model",), "patch", "model_patches"

    def patch(self, model, intensity, seed):
        m, proj_cache, sketch_proj_cache, perm_cache = model.clone(), LRU(64), LRU(32), LRU(128)

        def _orth_proj(df, dt, dev):
            key = (df, dt, _dev_key(dev)); P = proj_cache.get(key)
            if P is None:
                gen = torch.Generator(device=dev); gen.manual_seed(_stable_seed(seed, df, dt, "proj", _dev_key(dev)))
                P = torch.empty(df, dt, device=dev, dtype=torch.float32); torch.nn.init.orthogonal_(P); proj_cache.put(key, P)
            return P

        def _sketch_proj(d, dev):
            key = (d, 64, _dev_key(dev)); R = sketch_proj_cache.get(key)
            if R is None:
                gen = torch.Generator(device=dev); gen.manual_seed(_stable_seed(seed, d, 64, "sketch", _dev_key(dev)))
                R = torch.empty(d, 64, device=dev, dtype=torch.float32); torch.nn.init.orthogonal_(R); sketch_proj_cache.put(key, R)
            return R

        def baseline_attn(q, k, sample_max_q=512, sample_max_k=512):
            if q.dim() == 4: q = q.reshape(q.shape[0] * q.shape[1], q.shape[2], q.shape[3])
            if k.dim() == 4: k = k.reshape(k.shape[0] * k.shape[1], k.shape[2], k.shape[3])
            dq, dk = q.shape[-1], k.shape[-1]
            if dq == dk: q2, k2 = q, k
            elif dk > dq: P = _orth_proj(dk, dq, k.device); q2, k2 = q, torch.matmul(k.float(), P).to(k.dtype)
            else: P = _orth_proj(dq, dk, q.device); q2, k2 = torch.matmul(q.float(), P).to(q.dtype), k
            B, Tq = q2.shape[0], q2.shape[1]; S = k2.shape[1]
            if Tq > sample_max_q:
                step = max(1, Tq // sample_max_q); idx_q = torch.arange(0, Tq, step, device=q2.device)[:sample_max_q]; q2 = q2.index_select(1, idx_q)
            if S > sample_max_k:
                stepk = max(1, S // sample_max_k); idx_k = torch.arange(0, S, stepk, device=k2.device)[:sample_max_k]; k2 = k2.index_select(1, idx_k)
            d = q2.shape[-1]
            with torch.autocast("cuda", enabled=q2.is_cuda):
                logits = (q2 @ k2.transpose(1, 2)) / math.sqrt(max(int(d), 1)); probs = logits.softmax(dim=-1)
            return probs

        def kl_per_head(P1, P0):
            eps = 1e-8; P0c, P1c = P0.clamp_min(eps).float(), P1.clamp_min(eps).float()
            return (P1c * (P1c.log() - P0c.log())).sum(dim=-1).mean(dim=-1)

        def bsearch_alpha_vec_mix_k(q, k_ref, A0, k_new, max_a_vec, kl_cap, steps=6):
            dev, B = q.device, q.shape[0]; lo, hi = torch.zeros(B, device=dev), max_a_vec.to(torch.float32).clamp(0.0, 1.0)
            for _ in range(steps):
                mid = 0.5 * (lo + hi)
                k_try = (1.0 - mid.view(B, 1, 1)) * k_ref + mid.view(B, 1, 1) * k_new
                A1 = baseline_attn(q, k_try, sample_max_q=q.shape[1], sample_max_k=k_try.shape[1])
                too_high = kl_per_head(A1, A0) > float(kl_cap); hi, lo = torch.where(too_high, mid, hi), torch.where(~too_high, mid, lo)
            return lo.clamp(0.0, 1.0)

        def bsearch_alpha_vec_mix_q(q_ref, k_fixed, A0, q_new, max_a_vec, kl_cap, steps=6):
            dev, B = q_ref.device, q_ref.shape[0]; lo, hi = torch.zeros(B, device=dev), max_a_vec.to(torch.float32).clamp(0.0, 1.0)
            for _ in range(steps):
                mid = 0.5 * (lo + hi)
                q_try = (1.0 - mid.view(B, 1, 1)) * q_ref + mid.view(B, 1, 1) * q_new
                A1 = baseline_attn(q_try, k_fixed, sample_max_q=q_try.shape[1], sample_max_k=k_fixed.shape[1])
                too_high = kl_per_head(A1, A0) > float(kl_cap); hi, lo = torch.where(too_high, mid, hi), torch.where(~too_high, mid, lo)
            return lo.clamp(0.0, 1.0)

        def rms(x, dim=-1, keepdim=True, eps=1e-6): return x.float().pow(2).mean(dim=dim, keepdim=keepdim).add(eps).sqrt()
        def rms_match(x_new, x_ref, eps=1e-6): s = (rms(x_ref, eps=eps) / rms(x_new, eps=eps)).to(x_new.dtype); return x_new * s
        def clamp_rms(delta, factor=2.5, eps=1e-6):
            r = rms(delta, dim=-1, keepdim=True, eps=eps).squeeze(-1); med = r.median(dim=1, keepdim=True).values
            scale = (factor * med).clamp_min(eps) / r; return (delta.float() * scale.clamp_max(1.0).unsqueeze(-1)).to(delta.dtype)

        def orthogonal_noise(x, g, strength, tag):
            B, S, D = x.shape; dev = x.device; gen = torch.Generator(device=dev); gen.manual_seed(_stable_seed(seed, D, S, tag, _dev_key(dev)))
            n = torch.randn(x.shape, dtype=torch.float32, device=dev, generator=gen); xf = x.float()
            proj = (n * xf).sum(-1, keepdim=True) / xf.norm(dim=-1, keepdim=True).clamp_min(1e-6).pow(2)
            n_ortho = (n - proj * xf) / (n - proj * xf).norm(dim=-1, keepdim=True).clamp_min(1e-6)
            s = strength * (g.view(B, S, 1).float().add(1e-8).sqrt()); return (xf + s * n_ortho).to(x.dtype)

        def alpha_base_from_entropy(A, gamma=1.0, eps=1e-8):
            Ac = A.clamp_min(eps); H = -(Ac * Ac.log()).sum(dim=-1).mean(dim=-1)
            hmin, hmax = H.amin(dim=0, keepdim=True), H.amax(dim=0, keepdim=True).clamp_min(H.amin(dim=0, keepdim=True) + 1e-6)
            u = ((H - hmin) / (hmax - hmin)).clamp(0, 1); return (1.0 - u + eps).pow(gamma)

        def importance_from_A(A):
            I = (A.mean(dim=1) / A.mean(dim=1).sum(dim=-1, keepdim=True).clamp_min(1e-6)); return I, float(I.amax(dim=-1).mean().item())

        def importance_queries_from_A(A):
            R = A.max(dim=-1).values; return R / R.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        def _perm_indices(S, intensity, phase, dev, cfg_fac):
            ib, pb, cf = round(100 * float(intensity)), round(100 * float(phase)), round(100 * float(cfg_fac))
            key = (S, ib, pb, cf, _dev_key(dev)); p = perm_cache.get(key)
            if p is not None: return p
            gen = torch.Generator(device=dev); gen.manual_seed(_stable_seed(seed, S, ib, pb, cf, "perm", _dev_key(dev)))
            sigma = max(1.0, float(S) * (0.20 + 0.70 * (float(intensity) ** 0.75)) * (phase ** 0.5) * (1.0 - 0.5 * float(cfg_fac)))
            noise = torch.randn(S, device=dev, dtype=torch.float32, generator=gen) * sigma; noise = _tri_smooth(noise, 1 + int(2 * float(cfg_fac)))
            idx = torch.argsort(torch.arange(S, device=dev, dtype=torch.float32) + noise, dim=0); perm_cache.put(key, idx); return idx

        def _shuffle_tokens(k, v, intensity, phase, cfg_fac):
            idx = _perm_indices(k.shape[1], intensity, phase, k.device, cfg_fac); return k.index_select(1, idx), v.index_select(1, idx)

        # Allow more heads to be affected
        def _entropy_head_mask_percentile(A, intensity):
            eps = 1e-12; H = -(A.clamp_min(eps) * A.clamp_min(eps).log()).sum(dim=-1).mean(dim=-1)
            q = 1.0 - (0.25 + 0.65 * float(intensity)); thr = torch.quantile(H.float(), q); return H <= thr

        def tws(q, k, v, extra):
            if q.dim() == 4: q = q.reshape(q.shape[0] * q.shape[1], q.shape[2], q.shape[3])
            if k.dim() == 4: k = k.reshape(k.shape[0] * k.shape[1], k.shape[2], k.shape[3])
            if v.dim() == 4: v = v.reshape(v.shape[0] * v.shape[1], v.shape[2], v.shape[3])
            cfg_scale = float(extra.get("cfg_scale", 7.0)); I = max(0.0, min(1.0, float(intensity) / (1.0 + 0.15 * (cfg_scale - 7.0))))
            cfg_fac = max(0.0, min(1.0, (cfg_scale - 7.0) / 5.0))
            A0 = baseline_attn(q, k, sample_max_q=q.shape[1], sample_max_k=k.shape[1])
            B, S = k.shape[0], k.shape[1]
            mask = _entropy_head_mask_percentile(A0, I); 
            if not mask.any(): mask = torch.ones_like(mask, dtype=torch.bool)
            idx = mask.nonzero(as_tuple=True)[0]
            q_sel, k_sel, v_sel = q.index_select(0, idx), k.index_select(0, idx), v.index_select(0, idx)
            t = float(extra.get("timestep", 0)); T = float(extra.get("total_timesteps", max(t, 1))); phase = 1.0 - (t / T)
            A0_sel = baseline_attn(q_sel, k_sel, sample_max_q=q_sel.shape[1], sample_max_k=k_sel.shape[1])
            a_base_vec = alpha_base_from_entropy(A0_sel); I_k, conc = importance_from_A(A0_sel)
            p = min(0.55, (0.20 + 0.50 * I) * (1.0 - 0.20 * cfg_fac)); k_tokens = max(1, int(math.ceil(p * S)))
            tau = 0.25 + 0.75 * (1.0 - I) + 0.50 * cfg_fac
            gate_k_w = (I_k / tau).softmax(dim=-1) * (k_tokens / float(S)) * S
            gate_k_w = _smooth_tokens(gate_k_w.unsqueeze(-1), 2 + int(3 * cfg_fac)).squeeze(-1).clamp(0.0, 1.0)
            k_perm, v_perm = _shuffle_tokens(k_sel, v_sel, I, phase, cfg_fac)
            k_mix, v_mix = rms_match(k_perm, k_sel), rms_match(v_perm, v_sel)
            noise_k_strength = 0.25 * I * (0.5 + 0.5 * conc) * (phase ** 0.25) * (1.0 - 0.50 * cfg_fac)
            noise_v_strength = 0.18 * I * (0.5 + 0.5 * conc) * (phase ** 0.25) * (1.0 - 0.50 * cfg_fac)
            k_mix, v_mix = orthogonal_noise(k_mix, I_k, noise_k_strength, "k"), orthogonal_noise(v_mix, I_k, noise_v_strength, "v")
            kl_cap = 2.5 * (0.10 + 0.25 * I) * (phase ** 0.5) * (1.0 - 0.3 * cfg_fac)
            a_k_max = min(0.90, 0.45 + 0.25 * I) * (1.0 - 0.50 * cfg_fac)
            cap_scale = torch.ones_like(a_base_vec)
            a_k_vec = bsearch_alpha_vec_mix_k(q_sel, k_sel, A0_sel, k_mix, cap_scale * a_k_max, kl_cap, 6)
            a_v_base = 1.0 * ((0.8 + 0.2 * I) * (0.6 + 0.4 * conc))
            a_v_vec = (a_v_base * cap_scale * (a_k_vec / (cap_scale * a_k_max + 1e-6))).clamp(0.0, 1.0)
            dk, dv = clamp_rms(k_mix - k_sel, 4.5), clamp_rms(v_mix - v_sel, 3.5)
            dk, dv = _smooth_tokens(dk, 1 + int(3 * cfg_fac)), _smooth_tokens(dv, 2 + int(3 * cfg_fac))
            a_k_tok = _smooth_tokens((a_k_vec.view(-1, 1) * gate_k_w).unsqueeze(-1).to(k_sel.dtype), 1 + int(2 * cfg_fac))
            a_v_tok = _smooth_tokens((a_v_vec.view(-1, 1) * gate_k_w).unsqueeze(-1).to(v_sel.dtype), 1 + int(2 * cfg_fac))
            k_new_sel = rms_match(k_sel + a_k_tok * dk, k_sel); v_new_sel = rms_match(v_sel + a_v_tok * dv, v_sel)
            A0_fullQ = baseline_attn(q_sel, k_sel, sample_max_q=q_sel.shape[1], sample_max_k=k_sel.shape[1])
            I_q = importance_queries_from_A(A0_fullQ); Tq = q_sel.shape[1]; q_tokens = max(1, int(math.ceil(p * Tq)))
            top_vals_q, top_idx_q = I_q.topk(q_tokens, dim=-1, largest=True, sorted=False)
            gate_q_w = torch.zeros_like(I_q, dtype=torch.float32); gate_q_w.scatter_(1, top_idx_q, 1.0)
            gate_q_w = _smooth_tokens(gate_q_w.unsqueeze(-1), 1 + int(2 * cfg_fac)).squeeze(-1).clamp(0.0, 1.0)
            q_perm, _ = _shuffle_tokens(q_sel, q_sel, I * 1.0, phase, cfg_fac); q_mix = rms_match(q_perm, q_sel)
            q_mix = orthogonal_noise(q_mix, I_q, 0.35 * noise_k_strength, "q")
            kl_cap_q, a_q_max = 0.9 * kl_cap, 0.9 * a_k_max
            a_q_vec = torch.zeros(q_sel.shape[0], device=q_sel.device) if float(I_k.amax().item()) < 0.2 else bsearch_alpha_vec_mix_q(q_sel, k_sel, A0_fullQ, q_mix, cap_scale * a_q_max, kl_cap_q, 5)
            dq = clamp_rms(q_mix - q_sel, 4.0)
            a_q_tok = _smooth_tokens((0.90 * a_q_vec.view(-1, 1) * gate_q_w).unsqueeze(-1).to(q_sel.dtype), 1 + int(2 * cfg_fac))
            q_new_sel = (q_sel + a_q_tok * dq).to(q_sel.dtype)
            q_out, k_out, v_out = q.clone(), k.clone(), v.clone()
            q_out.index_copy_(0, idx, q_new_sel.to(q_out.dtype)); k_out.index_copy_(0, idx, k_new_sel.to(k_out.dtype)); v_out.index_copy_(0, idx, v_new_sel.to(v_out.dtype))
            return q_out, k_out, v_out

        m.set_model_attn2_patch(tws); return (m,)

NODE_CLASS_MAPPINGS = {"TWS": TWS}
NODE_DISPLAY_NAME_MAPPINGS = {"TWS": "Token-Weighted Shuffle"}
