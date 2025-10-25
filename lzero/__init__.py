import torch
import torch.nn.functional as F

class L0SmoothingFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # λ (sparsity on gradients): higher => flatter/cleaner regions
                "lam": ("FLOAT", {"default": 0.02, "min": 0.001, "max": 0.25, "step": 0.001, "display": "slider"}),
                # β0 (initial augmented Lagrangian weight)
                "beta_init": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 16.0, "step": 0.5, "display": "slider"}),
                # β multiplier per outer step
                "kappa": ("FLOAT", {"default": 2.0, "min": 1.2, "max": 4.0, "step": 0.1, "display": "slider"}),
                # # of outer iterations (β updates)
                "iterations": ("INT", {"default": 4, "min": 1, "max": 10, "step": 1, "display": "slider"}),
                # Blend with original for convenience (like your Kuwahara node)
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_l0"
    CATEGORY = "image/filters"

    def apply_l0(self, image, lam, beta_init, kappa, iterations, strength):
        # Convert to BCHW internal
        u = image.permute(0, 3, 1, 2).contiguous()
        u = u.to(torch.float32)  # robust for FFT
        f = u

        b, c, h, w = u.shape
        device = u.device
        pi = torch.pi

        # Frequency response for forward-difference Laplacian (periodic BC)
        wx = 2.0 * pi * torch.fft.fftfreq(w, d=1.0, device=device)  # (w,)
        wy = 2.0 * pi * torch.fft.fftfreq(h, d=1.0, device=device)  # (h,)
        sx = (2.0 - 2.0 * torch.cos(wx)).view(1, 1, 1, w)           # |Dx|^2
        sy = (2.0 - 2.0 * torch.cos(wy)).view(1, 1, h, 1)           # |Dy|^2
        lap_fft = sx + sy                                            # shape (1,1,h,w), real

        def grad(img):
            # forward differences with periodic wrap
            gx = torch.roll(img, shifts=-1, dims=3) - img  # along W
            gy = torch.roll(img, shifts=-1, dims=2) - img  # along H
            return gx, gy

        def div(px, py):
            # backward divergence (adjoint of forward differences)
            dx = px - torch.roll(px, shifts=1, dims=3)
            dy = py - torch.roll(py, shifts=1, dims=2)
            return dx + dy

        beta = beta_init
        for _ in range(int(iterations)):
            gx, gy = grad(u)
            mag2 = gx * gx + gy * gy
            thresh = lam / beta

            # Hard L0 shrinkage on gradient magnitude
            mask = (mag2 < thresh)
            px = gx.clone()
            py = gy.clone()
            px[mask] = 0.0
            py[mask] = 0.0

            # Solve (I - βΔ) u = f + β div(p,q) in Fourier domain
            rhs = f + beta * div(px, py)
            RHS = torch.fft.fft2(rhs)
            denom = (1.0 + beta * lap_fft).to(RHS.dtype)  # broadcast over (b,c,h,w)

            U = RHS / denom
            u = torch.fft.ifft2(U).real

            beta = beta * kappa

        out = u.permute(0, 2, 3, 1).contiguous()  # back to BHWC
        out = image * (1.0 - strength) + out * strength
        return (out,)

NODE_CLASS_MAPPINGS = {
    "L0SmoothingFilter": L0SmoothingFilter,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "L0SmoothingFilter": "L0 Gradient Minimization",
}
