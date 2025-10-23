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
                "intensity": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05}),
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
        mix_cache = LRU(64)

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

        def _sketch_proj(d, k, dev):
            key = (d, 64, _dev_key(dev))
            R = sketch_proj_cache.get(key)
            if R is None:
                gen = torch.Generator(device=dev)
                gen.manual_seed(_stable_seed(seed, d, 64, "sketch", _dev_key(dev)))
                R = torch.empty(d, 64, device=dev, dtype=torch.float32)
                torch.nn.init.orthogonal_(R)
                sketch_proj_cache.put(key, R)
            return R

        def baseline_attn(q, k, sample_max_q=384, sample_max_k=512):
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

        def kl_guard(P1, P0):
            eps = 1e-8
            P0c = P0.clamp_min(eps).float()
            P1c = P1.clamp_min(eps).float()
            KL = (P1c * (P1c.log() - P0c.log())).sum(dim=-1).mean(dim=(-1, -2))
            return KL.mean()

        def bsearch_alpha_k(q, k, A0, k_mix, a_max, kl_cap, steps=7, tol=5e-4, kl_tol=2e-4):
            lo = 0.0
            hi = float(max(0.0, min(1.0, a_max)))
            best = 0.0
            cap = A0.new_tensor(float(kl_cap))
            for _ in range(steps):
                if hi - lo <= tol:
                    break
                mid = 0.5 * (lo + hi)
                k_try = (1.0 - mid) * k + mid * k_mix
                A1 = baseline_attn(q, k_try)
                kl = kl_guard(A1, A0)
                if (kl - cap).abs() <= kl_tol:
                    best = mid
                    break
                if kl > cap:
                    hi = mid
                else:
                    best = mid
                    lo = mid
            return best

        def rescale_match(x_new, x_ref, eps=1e-6):
            mu0 = x_ref.mean(dim=1, keepdim=True)
            std0 = x_ref.std(dim=1, keepdim=True).clamp_min(eps)
            mu1 = x_new.mean(dim=1, keepdim=True)
            std1 = x_new.std(dim=1, keepdim=True).clamp_min(eps)
            return (x_new - mu1) * (std0 / std1) + mu0

        def orthogonal_noise(x, g, strength, seed_tag):
            B, S, D = x.shape
            dev = x.device
            gen = torch.Generator(device=dev)
            gen.manual_seed(_stable_seed(seed, D, S, seed_tag, _dev_key(dev)))
            n = torch.randn(x.shape, dtype=x.dtype, device=dev, generator=gen)
            proj = (n * x).sum(-1, keepdim=True) / (x.norm(dim=-1, keepdim=True).clamp_min(1e-6) ** 2)
            n_ortho = n - proj * x
            n_ortho = n_ortho / n_ortho.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            s = strength * g.view(B, S, 1)
            return x + s * n_ortho

        def alpha_from_entropy(A, intensity):
            eps = 1e-12
            Ac = A.clamp_min(eps)
            H = -(Ac * Ac.log()).sum(dim=-1)
            h = H.mean(dim=(-1, -2), keepdim=True)
            hmin = H.amin(dim=(-1, -2), keepdim=True)
            hmax = H.amax(dim=(-1, -2), keepdim=True).clamp_min(hmin + 1e-6)
            z = ((h - hmin) / (hmax - hmin)).squeeze()
            z = torch.nan_to_num(z, nan=0.0).clamp(0, 1)
            a = z.mean() * float(intensity)
            return float(a.item())

        def importance_from_A(A):
            I = A.mean(dim=1)
            I = I / I.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            c = I.amax(dim=-1).mean()
            return I, float(c.item())

        def _content_key(k, intensity):
            B, S, D = k.shape
            dev = k.device
            kn = k.float()
            kn = kn / kn.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            R = _sketch_proj(D, k, dev)
            s = torch.matmul(kn, R) if kn.dtype != torch.float16 else torch.matmul(kn.float(), R)
            s = s.mean(dim=1).mean(dim=0)
            q = torch.round(s * 256.0).to(torch.int16).tolist()
            return (S, round(float(intensity), 3), _dev_key(dev), tuple(q))

        def _mix_tokens_banded_topk(k, v, intensity):
            B, S, D = k.shape
            dev = k.device
            key = _content_key(k, intensity)
            cached = mix_cache.get(key)
            if cached is not None:
                idx_top, w_top = cached
                topk = idx_top.shape[-1]
                k_exp2 = k.unsqueeze(2).expand(-1, -1, topk, -1)
                v_exp2 = v.unsqueeze(2).expand(-1, -1, topk, -1)
                idx4k = idx_top.unsqueeze(-1).expand(-1, -1, -1, D)
                idx4v = idx_top.unsqueeze(-1).expand(-1, -1, -1, v.shape[-1])
                gathered_k = torch.gather(k_exp2, 1, idx4k)
                gathered_v = torch.gather(v_exp2, 1, idx4v)
                k_mix = (w_top.unsqueeze(-1) * gathered_k).sum(dim=2)
                v_mix = (w_top.unsqueeze(-1) * gathered_v).sum(dim=2)
                return k_mix, v_mix
            kn = k.float()
            kn = kn / kn.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            r_base = 4 + int(S * (0.01 + 0.04 * float(intensity)))
            r = int(max(2, min(S // 4, r_base)))
            W = 2 * r + 1
            offs = torch.arange(-r, r + 1, device=dev)
            center = torch.arange(S, device=dev).unsqueeze(1)
            neigh = center + offs.unsqueeze(0)
            mask = (neigh >= 0) & (neigh < S)
            neigh_clamped = neigh.clamp(0, S - 1)
            idx_exp = neigh_clamped.unsqueeze(0).expand(B, -1, -1)
            kn_exp = kn.unsqueeze(2).expand(-1, -1, W, -1)
            gather_idx = idx_exp.unsqueeze(-1).expand(-1, -1, -1, D)
            neigh_vecs = torch.gather(kn_exp, 1, gather_idx)
            sims = (kn_exp * neigh_vecs).sum(-1)
            sims = sims.masked_fill(~mask.unsqueeze(0), float("-inf"))
            dist_w = 0.25 + 0.75 * float(intensity)
            bias = -(offs.abs().float() / max(float(S), 1.0)) * dist_w
            sims = sims + bias.view(1, 1, W)
            sims = sims.masked_fill(offs.view(1, 1, W) == 0, float("-inf"))
            gen = torch.Generator(device=dev)
            gen.manual_seed(_stable_seed(seed, "mix", S, D, _dev_key(dev)))
            noise = torch.randn(sims.shape, dtype=sims.dtype, device=sims.device, generator=gen) * (0.02 + 0.08 * float(intensity))
            sims = sims + noise
            temp = max(0.25, 0.7 - 0.25 * float(intensity))
            topk_val = min(W, 8 + int(8 * float(intensity)))
            vals, idx_inW = sims.topk(k=topk_val, dim=-1)
            w_top = (vals / temp).softmax(dim=-1)
            BS = B * S
            idx_exp_flat = idx_exp.reshape(BS, W)
            idx_inW_flat = idx_inW.reshape(BS, topk_val)
            idx_top_flat = torch.gather(idx_exp_flat, 1, idx_inW_flat)
            idx_top = idx_top_flat.reshape(B, S, topk_val)
            k_exp2 = k.unsqueeze(2).expand(-1, -1, topk_val, -1)
            v_exp2 = v.unsqueeze(2).expand(-1, -1, topk_val, -1)
            idx4k = idx_top.unsqueeze(-1).expand(-1, -1, -1, D)
            idx4v = idx_top.unsqueeze(-1).expand(-1, -1, -1, v.shape[-1])
            gathered_k = torch.gather(k_exp2, 1, idx4k)
            gathered_v = torch.gather(v_exp2, 1, idx4v)
            k_mix = (w_top.unsqueeze(-1) * gathered_k).sum(dim=2)
            v_mix = (w_top.unsqueeze(-1) * gathered_v).sum(dim=2)
            mix_cache.put(key, (idx_top, w_top))
            return k_mix, v_mix

        def _entropy_head_mask(A, thr):
            eps = 1e-12
            Ac = A.clamp_min(eps)
            H = -(Ac * Ac.log()).sum(dim=-1).mean(dim=-1)
            hmin = H.amin(dim=0, keepdim=True)
            hmax = H.amax(dim=0, keepdim=True).clamp_min(hmin + 1e-6)
            z = (H - hmin) / (hmax - hmin)
            z = torch.nan_to_num(z, nan=0.0).clamp(0, 1)
            return z >= thr

        def tws(q, k, v, extra):
            if q.dim() == 4:
                q = q.reshape(q.shape[0] * q.shape[1], q.shape[2], q.shape[3])
            if k.dim() == 4:
                k = k.reshape(k.shape[0] * k.shape[1], k.shape[2], k.shape[3])
            if v.dim() == 4:
                v = v.reshape(v.shape[0] * v.shape[1], v.shape[2], v.shape[3])
            A0 = baseline_attn(q, k)
            a_base = alpha_from_entropy(A0, intensity)
            B, S, _ = k.shape
            thr = 0.35 + 0.4 * float(intensity)
            mask = _entropy_head_mask(A0, thr)
            if not mask.any():
                return q, k, v
            idx = mask.nonzero(as_tuple=True)[0]
            k_sel = k.index_select(0, idx)
            v_sel = v.index_select(0, idx)
            q_sel = q.index_select(0, idx)
            k_mix_sel, v_mix_sel = _mix_tokens_banded_topk(k_sel, v_sel, intensity)
            k_mix_sel = rescale_match(k_mix_sel, k_sel)
            v_mix_sel = rescale_match(v_mix_sel, v_sel)
            I, conc = importance_from_A(A0.index_select(0, idx))
            noise_k_strength = 0.15 * float(intensity) * (0.5 + 0.5 * conc)
            noise_v_strength = 0.25 * float(intensity) * (0.5 + 0.5 * conc)
            k_mix_sel = orthogonal_noise(k_mix_sel, I, noise_k_strength, "k")
            v_mix_sel = orthogonal_noise(v_mix_sel, I, noise_v_strength, "v")
            kl_cap = 0.05 + 0.25 * float(intensity)
            a_k_max = min(0.95, 0.35 + 0.55 * float(intensity))
            A0_sel = baseline_attn(q_sel, k_sel)
            a_k = bsearch_alpha_k(q_sel, k_sel, A0_sel, k_mix_sel, a_k_max * max(0.05, a_base), kl_cap, steps=5)
            a_v = max(0.0, min(1.0, (0.5 + 0.5 * conc) * (0.55 + 0.45 * float(intensity)) * max(0.1, a_base)))
            k_mix_sel = k_mix_sel.to(k.dtype)
            v_mix_sel = v_mix_sel.to(v.dtype)
            k_sel = k_sel.to(k.dtype)
            v_sel = v_sel.to(v.dtype)
            k_out = k.clone()
            v_out = v.clone()
            k_out.index_copy_(0, idx, ((1.0 - a_k) * k_sel + a_k * k_mix_sel).to(k_out.dtype))
            v_out.index_copy_(0, idx, ((1.0 - a_v) * v_sel + a_v * v_mix_sel).to(v_out.dtype))
            return q, k_out, v_out

        m.set_model_attn2_patch(tws)
        return (m,)

NODE_CLASS_MAPPINGS = {"TWS": TWS}
NODE_DISPLAY_NAME_MAPPINGS = {"TWS": "Token-Weighted Shuffle"}
    