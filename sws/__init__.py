import math
import hashlib
import torch
from collections import OrderedDict


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


class SWS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "patch"
    CATEGORY = "model_patches"

    def patch(self, model, intensity, seed):
        m = model.clone()
        smax, smin, alpha, wnd_max = 25.0, 0.28, 4.0, 8
        log_smin, log_smax = math.log(smin), math.log(smax)
        state, proj_cache, perm_cache = {"calls": 0, "cycle": 64}, LRU(64), LRU(256)
        sig_denom = 1.0 / (1.0 + math.exp(-alpha * 0.5))

        def get_u_and_w(extra):
            sig = idx = cur = tot = None
            if isinstance(extra, dict):
                if extra.get("sigma") is not None:
                    sig = extra["sigma"]
                elif extra.get("sigmas") is not None:
                    idx = extra.get("sigmas_index", extra.get("step", extra.get("t_index")))
                    if idx is not None:
                        try:
                            sig = float(extra["sigmas"][int(idx)])
                        except:
                            sig = None
                if sig is None:
                    for k in ["sigmas_step", "step", "t_index", "k_step", "model_step", "curr_iter", "timestep"]:
                        v = extra.get(k)
                        if v is not None:
                            try:
                                cur = float(v)
                                break
                            except:
                                pass
                    for k in ["sigmas_total", "steps", "total_steps", "num_steps", "max_steps"]:
                        v = extra.get(k)
                        if v is not None:
                            try:
                                tot = float(v)
                                break
                            except:
                                pass
            if sig is not None:
                s = float(max(min(sig, smax), smin))
                u = max(0.0, min(1.0, (math.log(s) - log_smin) / (log_smax - log_smin)))
            elif cur is not None and tot is not None and tot > 0:
                u = max(0.0, min(1.0, cur / tot))
            else:
                state["calls"] += 1
                u = (state["calls"] % state["cycle"]) / float(state["cycle"])
            w = (1.0 / (1.0 + math.exp(-alpha * (u - 0.5)))) / sig_denom
            return u, w

        def infer_hw(T, extra):
            H = W = None
            if isinstance(extra, dict):
                if extra.get("h") is not None and extra.get("w") is not None:
                    H, W = int(extra["h"]), int(extra["w"])
                elif isinstance(extra.get("hw"), (tuple, list)) and len(extra["hw"]) == 2:
                    H, W = int(extra["hw"][0]), int(extra["hw"][1])
                elif isinstance(extra.get("spatial"), (tuple, list)) and len(extra["spatial"]) == 2:
                    H, W = int(extra["spatial"][0]), int(extra["spatial"][1])
            if H is None or W is None or H * W != T:
                s = int(math.sqrt(T))
                if s * s == T:
                    H, W = s, s
                else:
                    W = int(round(math.sqrt(T)))
                    H = max(1, T // max(1, W))
                    if H * W < T:
                        H = min(H + 1, T)
                    if H * W != T:
                        H, W = 1, T
            return H, W

        def proj_mat(df, dt, dev):
            key = (df, dt, _dev_key(dev))
            P = proj_cache.get(key)
            if P is None:
                gen = torch.Generator(device=dev)
                gen.manual_seed(_stable_seed(seed, df, dt, _dev_key(dev)))
                M = torch.randn(df, dt, device=dev, dtype=torch.float32, generator=gen)
                U, S, Vh = torch.linalg.svd(M, full_matrices=False)
                P = U[:, :dt] if df >= dt else Vh[:dt, :].T
                if P.shape != (df, dt):
                    P = torch.nn.functional.pad(P, (0, max(0, dt - P.shape[1]))) if P.shape[1] < dt else P[:, :dt]
                proj_cache.put(key, P)
            return P

        def baseline_attn(q, k, sample_state=None, sample_max_q=384, sample_max_k=384):
            if q.dim() == 4:
                q = q.reshape(q.shape[0] * q.shape[1], q.shape[2], q.shape[3])
            if k.dim() == 4:
                k = k.reshape(k.shape[0] * k.shape[1], k.shape[2], k.shape[3])
            dq, dk = q.shape[-1], k.shape[-1]
            if dq == dk:
                q2, k2 = q, k
            elif dk > dq:
                P = proj_mat(dk, dq, k.device)
                k2, q2 = torch.matmul(k.float(), P).to(k.dtype), q
            else:
                P = proj_mat(dq, dk, q.device)
                q2, k2 = torch.matmul(q.float(), P).to(q.dtype), k
            B, Tq, D = q2.shape
            S = k2.shape[1]
            ss = {} if sample_state is None else dict(sample_state)
            if Tq > sample_max_q:
                if "idx_q" not in ss:
                    step = max(1, int(Tq // sample_max_q))
                    idx_q = torch.arange(0, Tq, step, device=q2.device)
                    if idx_q.numel() > sample_max_q:
                        idx_q = idx_q[:sample_max_q]
                    ss["idx_q"] = idx_q
                q2 = q2.index_select(1, ss["idx_q"])
            if S > sample_max_k:
                if "idx_k" not in ss:
                    step = max(1, int(S // sample_max_k))
                    idx_k = torch.arange(0, S, step, device=k2.device)
                    if idx_k.numel() > sample_max_k:
                        idx_k = idx_k[:sample_max_k]
                    ss["idx_k"] = idx_k
                k2 = k2.index_select(1, ss["idx_k"])
            d = q2.shape[-1]
            logits = torch.einsum("btd,bsd->bts", q2, k2) / math.sqrt(max(int(d), 1))
            probs = logits.softmax(dim=-1)
            return logits, probs, ss

        def entropy_alpha_from_probs(probs, u, intensity):
            eps = 1e-12
            A = probs.clamp_min(eps)
            H = -(A * A.log()).sum(dim=-1)
            Hm = H.mean(dim=(-1, -2), keepdim=True)
            Hmin = H.amin(dim=(-1, -2), keepdim=True)
            Hmax = H.amax(dim=(-1, -2), keepdim=True).clamp_min(Hmin + 1e-6)
            Hn = ((Hm - Hmin) / (Hmax - Hmin)).squeeze().nan_to_num(0.0).clamp(0, 1).mean()
            w_u = 4.0 * float(u) * (1.0 - float(u))
            a = Hn * probs.new_tensor(float(intensity) * w_u)
            return a.clamp(0.0, 1.0)

        def kl_guard(P1, P0):
            eps = 1e-8
            P0c, P1c = P0.clamp_min(eps).float(), P1.clamp_min(eps).float()
            KL = (P1c * (P1c.log() - P0c.log())).sum(dim=-1).mean(dim=(-1, -2))
            return KL.mean()

        def bsearch_alpha_k(q, k, A0, k_perm, a_max, kl_cap, sample_state, steps=7, tol=1e-3, kl_tol=1e-4):
            lo, hi, best = 0.0, float(max(0.0, min(1.0, a_max))), 0.0
            kl_cap_t = A0.new_tensor(float(kl_cap))
            for _ in range(steps):
                if hi - lo <= tol:
                    break
                mid = 0.5 * (lo + hi)
                k_try = (1.0 - mid) * k + mid * k_perm
                _, A1, ss = baseline_attn(q, k_try, sample_state)
                kl_val = kl_guard(A1, A0)
                if (kl_val - kl_cap_t).abs() <= kl_tol:
                    best = mid
                    break
                if kl_val > kl_cap_t:
                    hi = mid
                else:
                    best, lo = mid, mid
            return best

        def pick_window(H, W, u, wm):
            target = max(2, int(round(max(2.0, wm * max(0.1, 1.0 - u)))))
            candidates = [s for s in [8, 7, 6, 5, 4, 3, 2] if s <= wm and s <= target]
            for s in candidates:
                if H % s == 0 and W % s == 0:
                    return s
            for s in [4, 3, 2]:
                if s <= wm and H % s == 0 and W % s == 0:
                    return s
            return 1

        def window_perm_indices(H, W, s, device):
            key = (H, W, s, _dev_key(device))
            idx = perm_cache.get(key)
            if idx is not None:
                return idx
            I = torch.arange(H * W, device=device).reshape(H, W)
            nH, nW = H // s, W // s
            W0 = I.reshape(nH, s, nW, s).permute(0, 2, 1, 3).reshape(nH * nW, s * s)
            shift = max(1, (s * s) // 4)
            p = (torch.arange(s * s, device=device) + shift) % (s * s)
            W1 = W0.index_select(1, p)
            idx = W1.reshape(nH, nW, s, s).permute(0, 2, 1, 3).reshape(H, W).reshape(-1)
            perm_cache.put(key, idx)
            return idx

        def apply_idx(X, idx):
            B, T, D = X.shape
            idx_b = idx.unsqueeze(0).expand(B, -1)
            return X.gather(1, idx_b.unsqueeze(-1).expand(B, -1, D))

        def sws(q, k, v, extra):
            if q.dim() == 4:
                q = q.reshape(q.shape[0] * q.shape[1], q.shape[2], q.shape[3])
            if k.dim() == 4:
                k = k.reshape(k.shape[0] * k.shape[1], k.shape[2], k.shape[3])
            if v.dim() == 4:
                v = v.reshape(v.shape[0] * v.shape[1], v.shape[2], v.shape[3])
            u, _ = get_u_and_w(extra)
            tau = 1.0 + (1.0 - u)
            scale = (1.0 / float(tau)) ** 0.5
            q, k = q * scale, k * scale
            _, A0, ss = baseline_attn(q, k, None)
            a = float(entropy_alpha_from_probs(A0, u, intensity).item())
            if a <= 0.0:
                return q, k, v
            Tk, Tv = k.shape[1], v.shape[1]
            Hk, Wk = infer_hw(Tk, extra)
            Hv, Wv = infer_hw(Tv, extra)
            sk, sv = pick_window(Hk, Wk, u, wnd_max), pick_window(Hv, Wv, u, wnd_max)
            k_perm = apply_idx(k, window_perm_indices(Hk, Wk, sk, k.device)) if sk > 1 else k
            v_perm = apply_idx(v, window_perm_indices(Hv, Wv, sv, v.device)) if sv > 1 else v
            kl_cap = 0.08 * (1.0 - u) + 0.01
            a_k_max = 0.25 * a
            a_v = max(0.0, min(1.0, 0.6 * a * (0.5 + 0.5 * (1.0 - u))))
            a_k = 0.0 if u >= 0.75 else bsearch_alpha_k(q, k, A0, k_perm, a_k_max, kl_cap, ss, 7, 1e-3, 1e-4)
            k_final = (1.0 - a_k) * k + a_k * k_perm
            v_final = (1.0 - a_v) * v + a_v * v_perm
            return q, k_final, v_final

        m.set_model_attn2_patch(sws)
        return (m,)


NODE_CLASS_MAPPINGS = {"SWS": SWS}
NODE_DISPLAY_NAME_MAPPINGS = {"SWS": "Sigma-Weighted Shuffle"}
