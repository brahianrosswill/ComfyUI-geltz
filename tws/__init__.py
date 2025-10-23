import torch
import math
import hashlib
from collections import OrderedDict

try:
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    pass

class LRU:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.od = OrderedDict()
    def get(self, key):
        v = self.od.get(key)
        if v is not None:
            self.od.move_to_end(key)
        return v
    def put(self, key, value):
        self.od[key] = value
        self.od.move_to_end(key)
        if len(self.od) > self.capacity:
            self.od.popitem(last=False)

def _tri_smooth(v, iters=2):
    for _ in range(iters):
        v = (v + torch.roll(v, 1, 0) + torch.roll(v, -1, 0)) / 3.0
    return v

def _smooth_tokens(x, iters=1):
    for _ in range(iters):
        x = (x + torch.roll(x, 1, 1) + torch.roll(x, -1, 1)) / 3.0
    return x

def _dev_key(dev):
    idx = getattr(dev, "index", None)
    return f"{dev.type}:{idx}"

def _stable_seed(base_seed, *parts):
    m = hashlib.sha256()
    m.update(str(int(base_seed)).encode())
    for p in parts:
        m.update(str(p).encode())
    return int.from_bytes(m.digest()[:8], "little")

class TWS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "model_patches"

    def patch(self, model, intensity, seed):
        m = model.clone()
        proj_cache = LRU(64)
        sketch_proj_cache = LRU(32)
        perm_cache = LRU(128)

        def _orth_proj(df, dt, dev):
            key = (df, dt, _dev_key(dev))
            P = proj_cache.get(key)
            if P is None:
                gen = torch.Generator(device=dev)
                gen.manual_seed(_stable_seed(seed, df, dt, "proj", _dev_key(dev)))
                P = torch.empty(df, dt, device=dev, dtype=torch.float32)
                torch.nn.init.orthogonal_(P)
                proj_cache.put(key, P)
            return P

        def _sketch_proj(d, dev):
            key = (d, 64, _dev_key(dev))
            R = sketch_proj_cache.get(key)
            if R is None:
                gen = torch.Generator(device=dev)
                gen.manual_seed(_stable_seed(seed, d, 64, "sketch", _dev_key(dev)))
                R = torch.empty(d, 64, device=dev, dtype=torch.float32)
                torch.nn.init.orthogonal_(R)
                sketch_proj_cache.put(key, R)
            return R

        def baseline_attn(q, k, sample_max_q=512, sample_max_k=512):
            if q.dim() == 4:
                q = q.reshape(q.shape[0] * q.shape[1], q.shape[2], q.shape[3])
            if k.dim() == 4:
                k = k.reshape(k.shape[0] * k.shape[1], k.shape[2], k.shape[3])
            dq = q.shape[-1]
            dk = k.shape[-1]
            if dq == dk:
                q2 = q
                k2 = k
            elif dk > dq:
                P = _orth_proj(dk, dq, k.device)
                k2 = torch.matmul(k.float(), P).to(k.dtype)
                q2 = q
            else:
                P = _orth_proj(dq, dk, q.device)
                q2 = torch.matmul(q.float(), P).to(q.dtype)
                k2 = k
            B, Tq, _ = q2.shape
            S = k2.shape[1]
            if Tq > sample_max_q:
                step = max(1, int(Tq // sample_max_q))
                idx_q = torch.arange(0, Tq, step, device=q2.device)
                if idx_q.numel() > sample_max_q:
                    idx_q = idx_q[:sample_max_q]
                q2 = q2.index_select(1, idx_q)
            if S > sample_max_k:
                stepk = max(1, int(S // sample_max_k))
                idx_k = torch.arange(0, S, stepk, device=k2.device)
                if idx_k.numel() > sample_max_k:
                    idx_k = idx_k[:sample_max_k]
                k2 = k2.index_select(1, idx_k)
            d = q2.shape[-1]
            with torch.autocast("cuda", enabled=q2.is_cuda):
                logits = q2 @ k2.transpose(1, 2)
                logits = logits / math.sqrt(max(int(d), 1))
                probs = logits.softmax(dim=-1)
            return probs

        def kl_per_head(P1, P0):
            eps = 1e-8
            P0c = P0.clamp_min(eps).float()
            P1c = P1.clamp_min(eps).float()
            KL = (P1c * (P1c.log() - P0c.log())).sum(dim=-1).mean(dim=-1)
            return KL

        def bsearch_alpha_vec_mix_k(q, k_ref, A0, k_new, max_a_vec, kl_cap, steps=6):
            dev = q.device
            B = q.shape[0]
            lo = torch.zeros(B, device=dev, dtype=torch.float32)
            hi = max_a_vec.to(torch.float32).clamp(0.0, 1.0)
            for _ in range(steps):
                mid = 0.5 * (lo + hi)
                k_try = (1.0 - mid.view(B, 1, 1)) * k_ref + mid.view(B, 1, 1) * k_new
                A1 = baseline_attn(q, k_try, sample_max_q=q.shape[1], sample_max_k=k_try.shape[1])
                kl = kl_per_head(A1, A0)
                too_high = kl > float(kl_cap)
                hi = torch.where(too_high, mid, hi)
                lo = torch.where(~too_high, mid, lo)
            return lo.clamp(0.0, 1.0)

        def bsearch_alpha_vec_mix_q(q_ref, k_fixed, A0, q_new, max_a_vec, kl_cap, steps=6):
            dev = q_ref.device
            B = q_ref.shape[0]
            lo = torch.zeros(B, device=dev, dtype=torch.float32)
            hi = max_a_vec.to(torch.float32).clamp(0.0, 1.0)
            for _ in range(steps):
                mid = 0.5 * (lo + hi)
                q_try = (1.0 - mid.view(B, 1, 1)) * q_ref + mid.view(B, 1, 1) * q_new
                A1 = baseline_attn(q_try, k_fixed, sample_max_q=q_try.shape[1], sample_max_k=k_fixed.shape[1])
                kl = kl_per_head(A1, A0)
                too_high = kl > float(kl_cap)
                hi = torch.where(too_high, mid, hi)
                lo = torch.where(~too_high, mid, lo)
            return lo.clamp(0.0, 1.0)

        def rms(x, dim=-1, keepdim=True, eps=1e-6):
            return (x.float().pow(2).mean(dim=dim, keepdim=keepdim).add(eps).sqrt())

        def rms_match(x_new, x_ref, eps=1e-6):
            r_ref = rms(x_ref, dim=-1, keepdim=True, eps=eps)
            r_new = rms(x_new, dim=-1, keepdim=True, eps=eps)
            s = (r_ref / r_new).to(x_new.dtype)
            return x_new * s

        def clamp_rms(delta, factor=2.5, eps=1e-6):
            r = rms(delta, dim=-1, keepdim=True, eps=eps).squeeze(-1)
            med = r.median(dim=1, keepdim=True).values
            tau = (factor * med).clamp_min(eps)
            scale = (tau / r).clamp_max(1.0).unsqueeze(-1)
            return (delta.float() * scale).to(delta.dtype)

        def orthogonal_noise(x, g, strength, tag):
            B, S, D = x.shape
            dev = x.device
            gen = torch.Generator(device=dev)
            gen.manual_seed(_stable_seed(seed, D, S, tag, _dev_key(dev)))
            n = torch.randn(x.shape, dtype=torch.float32, device=dev, generator=gen)
            x_f = x.float()
            proj = (n * x_f).sum(-1, keepdim=True) / x_f.norm(dim=-1, keepdim=True).clamp_min(1e-6).pow(2)
            n_ortho = n - proj * x_f
            n_ortho = n_ortho / n_ortho.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            s = strength * (g.view(B, S, 1).float().add(1e-8).sqrt())
            out = x_f + s * n_ortho
            return out.to(x.dtype)

        def alpha_base_from_entropy(A, gamma=1.0, eps=1e-8):
            Ac = A.clamp_min(eps)
            H = -(Ac * Ac.log()).sum(dim=-1).mean(dim=-1)
            hmin = H.amin(dim=0, keepdim=True)
            hmax = H.amax(dim=0, keepdim=True).clamp_min(hmin + 1e-6)
            u = ((H - hmin) / (hmax - hmin)).clamp(0, 1)
            z = 1.0 - u
            return (z + eps).pow(gamma)

        def importance_from_A(A):
            I = A.mean(dim=1)
            I = I / I.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            c = I.amax(dim=-1).mean()
            return I, float(c.item())

        def importance_queries_from_A(A):
            R = A.max(dim=-1).values
            R = R / R.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            return R

        def _perm_indices(S, intensity, phase, dev, cfg_fac):
            ib = round(100 * float(intensity))
            pb = round(100 * float(phase))
            cf = round(100 * float(cfg_fac))
            key = (S, ib, pb, cf, _dev_key(dev))
            p = perm_cache.get(key)
            if p is not None:
                return p
            gen = torch.Generator(device=dev)
            gen.manual_seed(_stable_seed(seed, S, ib, pb, cf, "perm", _dev_key(dev)))
            sigma = max(1.0, float(S) * (0.06 + 0.22 * float(intensity)) * (phase ** 0.5) * (1.0 - 0.6 * float(cfg_fac)))
            noise = torch.randn(S, device=dev, dtype=torch.float32, generator=gen) * sigma
            iters = 1 + int(2 * float(cfg_fac))
            noise = _tri_smooth(noise, iters)
            idx = torch.argsort(torch.arange(S, device=dev, dtype=torch.float32) + noise, dim=0)
            perm_cache.put(key, idx)
            return idx

        def _shuffle_tokens(k, v, intensity, phase, cfg_fac):
            B, S, D = k.shape
            dev = k.device
            idx = _perm_indices(S, intensity, phase, dev, cfg_fac)
            k_perm = k.index_select(1, idx)
            v_perm = v.index_select(1, idx)
            return k_perm, v_perm

        def _entropy_head_mask_percentile(A, intensity):
            eps = 1e-12
            Ac = A.clamp_min(eps)
            H = -(Ac * Ac.log()).sum(dim=-1).mean(dim=-1)
            q = 1.0 - (0.15 + 0.55 * float(intensity))
            thr = torch.quantile(H.float(), q)
            return H <= thr

        def tws(q, k, v, extra):
            if q.dim() == 4:
                q = q.reshape(q.shape[0] * q.shape[1], q.shape[2], q.shape[3])
            if k.dim() == 4:
                k = k.reshape(k.shape[0] * k.shape[1], k.shape[2], k.shape[3])
            if v.dim() == 4:
                v = v.reshape(v.shape[0] * v.shape[1], v.shape[2], v.shape[3])

            cfg_scale = float(extra.get("cfg_scale", 7.0))
            I = float(intensity) / (1.0 + 0.15 * (cfg_scale - 7.0))
            I = max(0.0, min(1.0, I))
            cfg_scale = float(extra.get("cfg_scale", 7.0))
            I = float(intensity) / (1.0 + 0.15 * (cfg_scale - 7.0))
            I = max(0.0, min(1.0, I))
            cfg_fac = max(0.0, min(1.0, (cfg_scale - 7.0) / 5.0))

            A0 = baseline_attn(q, k, sample_max_q=q.shape[1], sample_max_k=k.shape[1])
            B, S, _ = k.shape
            mask = _entropy_head_mask_percentile(A0, I)
            if not mask.any():
                mask = torch.ones_like(mask, dtype=torch.bool)
            idx = mask.nonzero(as_tuple=True)[0]

            q_sel = q.index_select(0, idx)
            k_sel = k.index_select(0, idx)
            v_sel = v.index_select(0, idx)

            t = float(extra.get("timestep", 0))
            T = float(extra.get("total_timesteps", max(t, 1)))
            phase = 1.0 - (t / T)

            A0_sel = baseline_attn(q_sel, k_sel, sample_max_q=q_sel.shape[1], sample_max_k=k_sel.shape[1])
            a_base_vec = alpha_base_from_entropy(A0_sel)
            I_k, conc = importance_from_A(A0_sel)

            p = (0.12 + 0.22 * I) * (1.0 - 0.25 * cfg_fac)

            k_tokens = max(1, int(math.ceil(p * S)))
            tau = 0.25 + 0.75 * (1.0 - I) + 0.50 * cfg_fac
            gate_k_w = (I_k / tau).softmax(dim=-1)
            gate_k_w = gate_k_w * (k_tokens / float(S)) * S
            gate_k_w = _smooth_tokens(gate_k_w.unsqueeze(-1), iters=2 + int(3 * cfg_fac)).squeeze(-1).clamp(0.0, 1.0)

            k_perm, v_perm = _shuffle_tokens(k_sel, v_sel, I, phase, cfg_fac)
            k_mix = rms_match(k_perm, k_sel)
            v_mix = rms_match(v_perm, v_sel)

            noise_k_strength = 0.10 * I * (0.5 + 0.5 * conc) * (phase ** 0.25) * (1.0 - 0.70 * cfg_fac)
            noise_v_strength = 0.06 * I * (0.5 + 0.5 * conc) * (phase ** 0.25) * (1.0 - 0.70 * cfg_fac)

            k_mix = orthogonal_noise(k_mix, I_k, noise_k_strength, "k")
            v_mix = orthogonal_noise(v_mix, I_k, noise_v_strength, "v")

            kl_cap = 1.6 * (0.10 + 0.25 * I) * (phase ** 0.5) * (1.0 - 0.3 * cfg_fac)
            a_k_max = min(0.70, 0.45 + 0.25 * I) * (1.0 - 0.50 * cfg_fac)
            cap_scale = a_base_vec.clamp_min(0.08).clamp_max(1.0).to(torch.float32)
            a_k_vec = bsearch_alpha_vec_mix_k(q_sel, k_sel, A0_sel, k_mix, cap_scale * a_k_max, kl_cap, steps=6)

            a_v_base = 0.75 * ((0.6 + 0.4 * I) * (0.4 + 0.6 * conc))
            a_v_vec = (a_v_base * cap_scale * (a_k_vec / (cap_scale * a_k_max + 1e-6))).clamp(0.0, 1.0)

            dk = clamp_rms(k_mix - k_sel, factor=2.2)
            dv = clamp_rms(v_mix - v_sel, factor=1.4)
            dk = _smooth_tokens(dk, iters=1 + int(3 * cfg_fac))
            dv = _smooth_tokens(dv, iters=2 + int(3 * cfg_fac))
            dv = dv - dv.mean(dim=1, keepdim=True)

            a_k_tok = (a_k_vec.view(-1, 1) * gate_k_w).unsqueeze(-1).to(k_sel.dtype)
            a_v_tok = (a_v_vec.view(-1, 1) * gate_k_w).unsqueeze(-1).to(v_sel.dtype)
            a_k_tok = _smooth_tokens(a_k_tok, iters=1 + int(2 * cfg_fac))
            a_v_tok = _smooth_tokens(a_v_tok, iters=1 + int(2 * cfg_fac))

            k_new_sel = rms_match(k_sel + a_k_tok * dk, k_sel)
            v_new_sel = rms_match(v_sel + a_v_tok * dv, v_sel)

            A0_fullQ = baseline_attn(q_sel, k_sel, sample_max_q=q_sel.shape[1], sample_max_k=k_sel.shape[1])
            I_q = importance_queries_from_A(A0_fullQ)
            Tq = q_sel.shape[1]
            q_tokens = max(1, int(math.ceil(p * Tq)))
            top_vals_q, top_idx_q = I_q.topk(q_tokens, dim=-1, largest=True, sorted=False)
            gate_q = torch.zeros_like(I_q, dtype=torch.bool)
            gate_q.scatter_(1, top_idx_q, True)

            q_perm, _ = _shuffle_tokens(q_sel, q_sel, I * 0.6, phase, cfg_fac)
            q_mix = rms_match(q_perm, q_sel)
            q_mix = orthogonal_noise(q_mix, I_q, 0.35 * noise_k_strength, "q")

            kl_cap_q = 0.40 * kl_cap
            a_q_max = 0.45 * a_k_max
            if conc < 0.2:
                a_q_vec = torch.zeros(q_sel.shape[0], device=q_sel.device, dtype=torch.float32)
            else:
                a_q_vec = bsearch_alpha_vec_mix_q(q_sel, k_sel, A0_fullQ, q_mix, cap_scale * a_q_max, kl_cap_q, steps=5)

            gate_q_w = torch.zeros_like(I_q, dtype=torch.float32)
            gate_q_w.scatter_(1, top_idx_q, 1.0)
            gate_q_w = _smooth_tokens(gate_q_w.unsqueeze(-1), iters=1 + int(2 * cfg_fac)).squeeze(-1).clamp(0.0, 1.0)

            dq = clamp_rms(q_mix - q_sel)
            a_q_tok = (0.60 * a_q_vec.view(-1, 1) * gate_q_w).unsqueeze(-1).to(q_sel.dtype)
            a_q_tok = _smooth_tokens(a_q_tok, iters=1 + int(2 * cfg_fac))
            q_new_sel = (q_sel + a_q_tok * dq).to(q_sel.dtype)

            q_out = q.clone()
            k_out = k.clone()
            v_out = v.clone()
            q_out.index_copy_(0, idx, q_new_sel.to(q_out.dtype))
            k_out.index_copy_(0, idx, k_new_sel.to(k_out.dtype))
            v_out.index_copy_(0, idx, v_new_sel.to(v_out.dtype))
            return q_out, k_out, v_out

        m.set_model_attn2_patch(tws)
        return (m,)

NODE_CLASS_MAPPINGS = {"TWS": TWS}
NODE_DISPLAY_NAME_MAPPINGS = {"TWS": "Token-Weighted Shuffle"}
