import torch
import torch.nn.functional as F

class LocalLaplacianFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "levels": ("INT",   {"default": 5, "min": 3, "max": 8,  "step": 1,   "display": "slider"}),
                # unified sigma: drives spatial scale and range sensitivity together
                "sigma":  ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "detail_boost": ("FLOAT", {"default": 1.25, "min": 0.0, "max": 3.0, "step": 0.05, "display": "slider"}),
                "gamma":  ("FLOAT", {"default": 1.00, "min": 0.5, "max": 2.5, "step": 0.10, "display": "slider"}),
                "strength": ("FLOAT", {"default": 1.00, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_llf"
    CATEGORY = "image/filters"

    def apply_llf(self, image, levels, sigma, detail_boost, gamma, strength):
        # ----- helpers (nested => no leakage, guaranteed return) -------------
        def _to_bchw(img_bhwc):
            x = img_bhwc.permute(0, 3, 1, 2).contiguous()
            return x.clamp(0.0, 1.0).to(torch.float32)

        def _to_bhwc(x_bchw):
            return x_bchw.permute(0, 2, 3, 1).contiguous()

        def _luma(x):  # supports 1 or 3 channels
            if x.shape[1] == 1:
                return x
            r, g, b = x[:, :1], x[:, 1:2], x[:, 2:3]
            return 0.299 * r + 0.587 * g + 0.114 * b

        def _gauss1d(s, device, dtype, radius=None):
            s = max(float(s), 0.1)
            r = int(max(1, round(3.0 * s))) if radius is None else int(max(0, int(radius)))
            x = torch.arange(-r, r + 1, device=device, dtype=dtype)
            k = torch.exp(-(x * x) / (2.0 * s * s))
            k = k / (k.sum() + 1e-12)
            return k.view(1, 1, -1)  # (1,1,K)

        def _blur(img, s):
            # separable Gaussian with reflect padding; clamp pad ≤ size-1
            b, c, h, w = img.shape
            r0 = int(max(0, round(3.0 * max(float(s), 0.1))))
            r_x = min(r0, max(0, w - 1))
            r_y = min(r0, max(0, h - 1))
            if h < 2 or w < 2 or (r_x == 0 and r_y == 0):
                return img

            kx1d = _gauss1d(s, img.device, img.dtype, r_x)
            ky1d = _gauss1d(s, img.device, img.dtype, r_y)

            kx = kx1d.view(1, 1, 1, -1).repeat(c, 1, 1, 1)  # (c,1,1,Kx)
            ky = ky1d.view(1, 1, -1, 1).repeat(c, 1, 1, 1)  # (c,1,Ky,1)
            px = min(kx.shape[-1] // 2, max(0, w - 1))
            py = min(ky.shape[-2] // 2, max(0, h - 1))

            x = img
            if px > 0:
                x = F.pad(x, (px, px, 0, 0), mode="reflect")
            x = F.conv2d(x, kx, groups=c)
            if py > 0:
                x = F.pad(x, (0, 0, py, py), mode="reflect")
            x = F.conv2d(x, ky, groups=c)
            return x

        def _down(x):
            # anti-aliased downsample: blur then half-size (min size 1)
            x = _blur(x, 1.0)
            b, c, h, w = x.shape
            nh, nw = max(1, h // 2), max(1, w // 2)
            return F.interpolate(x, size=(nh, nw), mode="bilinear", align_corners=False)

        def _up(x, size_like):
            x = F.interpolate(x, size=size_like, mode="bilinear", align_corners=False)
            return _blur(x, 1.0)

        def _remap_detail(d, lvl, sigma_r, gamma_local):
            # Local Laplacian remap: compress large edges, keep/boost small details
            s = sigma_r * (2.0 ** lvl)
            w = torch.exp(- (torch.abs(d) / (s + 1e-6)) ** gamma_local)
            return detail_boost * d * w

        # ----- parameter tying: single sigma -> spatial + range ----------------
        # Spatial base σ in [0.6, 3.0], step in [1.6, 2.6], range σ_r in [0.04, 0.40]
        sigma_base = 0.6 + 2.4 * float(sigma)
        sigma_step = 1.6 + 1.0 * float(sigma)
        sigma_r    = 0.04 + 0.36 * float(sigma)

        # ----- pipeline -------------------------------------------------------
        x = _to_bchw(image)
        y = _luma(x)  # (B,1,H,W)

        # Gaussian pyramid (luminance)
        G, sigmas = [y], []
        for lvl in range(levels - 1):
            s_lvl = sigma_base * (sigma_step ** lvl)
            sigmas.append(s_lvl)
            G.append(_down(_blur(G[-1], s_lvl)))
        sigmas.append(sigma_base * (sigma_step ** (levels - 1)))

        # Laplacian pyramid
        L = []
        for lvl in range(levels - 1):
            up = _up(_blur(G[lvl + 1], sigmas[lvl]), G[lvl].shape[-2:])
            L.append(G[lvl] - up)
        L.append(G[-1])  # coarsest residual

        # Remap Laplacian bands
        Lm = [ _remap_detail(L[lvl], lvl, sigma_r, gamma) for lvl in range(levels - 1) ]
        Lm.append(L[-1])

        # Reconstruct luminance
        y_rec = Lm[-1]
        for lvl in reversed(range(levels - 1)):
            y_rec = _up(y_rec, Lm[lvl].shape[-2:]) + Lm[lvl]

        # Re-inject luminance into RGB/gray
        eps = 1e-6
        ratio = (y_rec / (y + eps)).clamp(0.0, 4.0)
        out_bchw = (x * ratio).clamp(0.0, 1.0)

        out = image * (1.0 - strength) + _to_bhwc(out_bchw) * strength
        return (out,)

NODE_CLASS_MAPPINGS = {
    "LocalLaplacianFilter": LocalLaplacianFilter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LocalLaplacianFilter": "Local Laplacian Filter",
}
