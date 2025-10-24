import math
import torch
import torch.nn.functional as F

try:
    import comfy.model_management as mm
except Exception:
    class _MM:
        @staticmethod
        def get_torch_device():
            return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        @staticmethod
        def should_use_fp16():
            return False

        @staticmethod
        def should_use_bf16():
            return False

    mm = _MM()


def _pick_out_dtype():
    try:
        if hasattr(mm, "should_use_bf16") and mm.should_use_bf16():
            return torch.bfloat16
        if hasattr(mm, "should_use_fp16") and mm.should_use_fp16():
            return torch.float16
    except Exception:
        pass
    return torch.float32


def _unit_gauss_all(x, eps=1e-6):
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    std = x.std(dim=(1, 2, 3), keepdim=True).clamp_min(eps)
    return (x - mean) / std


def _unit_gauss_channel(x, eps=1e-6):
    mean = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True).clamp_min(eps)
    return (x - mean) / std


def _scharr_kernels(device, dtype):
    gx = torch.tensor(
        [[3.0, 0.0, -3.0], [10.0, 0.0, -10.0], [3.0, 0.0, -3.0]],
        device=device,
        dtype=dtype,
    ) / 16.0
    gy = torch.tensor(
        [[3.0, 10.0, 3.0], [0.0, 0.0, 0.0], [-3.0, -10.0, -3.0]],
        device=device,
        dtype=dtype,
    ) / 16.0
    return gx, gy


def _sobel_kernels(device, dtype):
    gx = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        device=device,
        dtype=dtype,
    ) / 4.0
    gy = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        device=device,
        dtype=dtype,
    ) / 4.0
    return gx, gy


def _gradients_depthwise(x, use_scharr=True):
    _, c, _, _ = x.shape
    device, dtype = x.device, x.dtype
    gx, gy = (_scharr_kernels(device, dtype) if use_scharr else _sobel_kernels(device, dtype))
    gx = gx.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    gy = gy.view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    px = F.conv2d(x, gx, padding=1, groups=c)
    py = F.conv2d(x, gy, padding=1, groups=c)
    return px, py


def _gaussian_kernel1d(ks, sigma, device, dtype):
    ax = torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2
    g = torch.exp(-0.5 * (ax / sigma) ** 2)
    return g / g.sum().clamp_min(1e-12)


def _gaussian_blur_sep_depthwise(x, sigma):
    _, c, _, _ = x.shape
    ks = int(max(3, math.ceil(sigma * 6.0)))
    if ks % 2 == 0:
        ks += 1
    g = _gaussian_kernel1d(ks, max(1e-6, float(sigma)), x.device, x.dtype)
    kx = g.view(1, 1, 1, ks).repeat(c, 1, 1, 1)
    ky = g.view(1, 1, ks, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(x, kx, padding=(0, ks // 2), groups=c)
    x = F.conv2d(x, ky, padding=(ks // 2, 0), groups=c)
    return x


def _fft2(x):
    return torch.fft.rfft2(x, norm="ortho")


def _ifft2(X, s):
    return torch.fft.irfft2(X, s=s, norm="ortho")


def _radial_r(h, w, device, dtype):
    fy = torch.fft.fftfreq(h, d=1.0).to(device=device, dtype=dtype)
    fx = torch.fft.rfftfreq(w, d=1.0).to(device=device, dtype=dtype)
    yy, xx = torch.meshgrid(fy, fx, indexing="ij")
    r = torch.sqrt(yy * yy + xx * xx)
    return r / r.max().clamp_min(1e-12)


def _high_low_energy_fraction(x):
    b, _, h, w = x.shape
    X = _fft2(x)
    amp2 = (X.real ** 2 + X.imag ** 2)
    r = _radial_r(h, w, x.device, x.dtype)
    low = r <= 0.15
    high = r >= 0.35
    El = (amp2 * low).sum(dim=(-2, -1)).mean(dim=1).mean()
    Eh = (amp2 * high).sum(dim=(-2, -1)).mean(dim=1).mean()
    Et = amp2.sum(dim=(-2, -1)).mean(dim=1).mean().clamp_min(1e-6)
    hf_frac = (Eh / Et).clamp(0.0, 1.0)
    lf_frac = (El / Et).clamp(0.0, 1.0)
    return hf_frac, lf_frac


def _orientation_coherence(px, py, eps=1e-6):
    Jxx = (px * px).mean(dim=(1, 2, 3))
    Jyy = (py * py).mean(dim=(1, 2, 3))
    Jxy = (px * py).mean(dim=(1, 2, 3))
    num = (Jxx - Jyy) ** 2 + 4.0 * (Jxy ** 2)
    den = (Jxx + Jyy).clamp_min(eps) ** 2
    return (num / den).clamp(0.0, 1.0).mean()


def _kurtosis_penalty(x):
    xz = _unit_gauss_channel(x)
    m4 = (xz ** 4).mean()
    exk = m4 - 3.0
    return -exk.abs()


def _spectral_shape(x, beta=0.5, mix=0.5):
    if mix <= 0.0:
        return x
    _, _, h, w = x.shape
    X = _fft2(x)
    r = _radial_r(h, w, x.device, x.dtype)
    eps = 1e-6
    t = (r.clamp_min(eps)) ** beta
    s = (1.0 - mix) + mix * t
    s = s.unsqueeze(0).unsqueeze(0)
    X = X * s
    y = _ifft2(X, s=(h, w))
    return _unit_gauss_channel(y)


def _precondition_dither(x, h, w):
    sigma = max(1.0, min(h, w) * 0.04)
    x_blur = _gaussian_blur_sep_depthwise(x, sigma=sigma)
    return x + 0.12 * (x - x_blur)


def _score(x, use_scharr=True):
    x32 = x.to(torch.float32)
    gx, gy = _gradients_depthwise(x32, use_scharr=use_scharr)
    edge = torch.sqrt(gx * gx + gy * gy + 1e-9).mean()
    hf, _ = _high_low_energy_fraction(x32)
    coh = _orientation_coherence(gx, gy)
    kurt = _kurtosis_penalty(x32)
    a, b, c, d = 1.0, 0.55, 0.25, 0.35
    return a * edge + b * hf + c * kurt - d * coh


def _make_generator(device, seed):
    dev_type = device.type if isinstance(device, torch.device) else str(device)
    g = torch.Generator(device=device if dev_type in ("cuda", "mps") else "cpu")
    if seed is None or seed < 0:
        seed = int(torch.seed() % (2**31 - 1))
    g.manual_seed(int(seed))
    return g


def _randn_like_shape(shape, device, g):
    gen_device = g.device if hasattr(g, "device") else torch.device("cpu")
    x = torch.randn(shape, device=gen_device, dtype=torch.float32, generator=g)
    return x.to(device=device, non_blocking=True) if gen_device != device else x


def _dil(shape, seed, iters=2, eta=0.05, out_dtype=None, out_device=None):
    b, c, h, w = shape
    device = out_device if out_device is not None else mm.get_torch_device()
    dtype = out_dtype if out_dtype is not None else _pick_out_dtype()
    g = _make_generator(device, seed)
    x = _randn_like_shape((b, c, h, w), device, g)
    x = _precondition_dither(x, h, w)
    x = _unit_gauss_all(x)
    with torch.inference_mode(False):
        for _ in range(int(max(0, iters))):
            x = x.detach().requires_grad_(True)
            (grad,) = torch.autograd.grad(_score(x, use_scharr=False), x, create_graph=False, retain_graph=False)
            x = (x + float(eta) * grad).detach()
            x = _unit_gauss_all(x)
    return x.to(device=device, dtype=dtype, non_blocking=True)


def _dil2(
    shape,
    seed,
    iters=2,
    eta=0.05,
    beta=0.5,
    spectral_mix=0.5,
    use_scharr=True,
    out_dtype=None,
    out_device=None,
):
    b, c, h, w = shape
    device = out_device if out_device is not None else mm.get_torch_device()
    dtype = out_dtype if out_dtype is not None else _pick_out_dtype()
    base_seed = int(seed if (seed is not None and seed >= 0) else torch.seed())
    seeds = [base_seed + i * 1315423911 for i in range(c)]
    xs = []
    for i in range(c):
        gc = _make_generator(device, seeds[i])
        xs.append(_randn_like_shape((b, 1, h, w), device, gc))
    x = torch.cat(xs, dim=1)
    x = _spectral_shape(x, beta=beta, mix=spectral_mix)
    x = _precondition_dither(x, h, w)
    x = _unit_gauss_channel(x)
    with torch.inference_mode(False):
        for _ in range(int(max(0, iters))):
            x = x.detach().requires_grad_(True)
            (grad,) = torch.autograd.grad(_score(x, use_scharr=use_scharr), x, create_graph=False, retain_graph=False)
            x = (x + float(eta) * grad).detach()
            x = _spectral_shape(x, beta=beta, mix=spectral_mix)
            x = _unit_gauss_channel(x)
    return x.to(device=device, dtype=dtype, non_blocking=True)


class DIL_EmptyLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1, "step": 1, "display": "number", "hidden": True}),
                "iters": ("INT", {"default": 2, "min": 0, "max": 8, "step": 1, "display": "slider", "hidden": True}),
                "eta": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.3, "step": 0.005, "display": "slider", "hidden": True}),
                "channels": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1, "hidden": True}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "make"
    CATEGORY = "latent"

    @staticmethod
    def _latent_wh(width, height):
        w = max(8, int(width)) // 8
        h = max(8, int(height)) // 8
        return w, h

    def make(self, width, height, batch_size=1, seed=-1, iters=2, eta=0.05, channels=4):
        device = mm.get_torch_device()
        out_dtype = _pick_out_dtype()
        wl, hl = self._latent_wh(width, height)
        shape = (int(batch_size), int(channels), hl, wl)
        with torch.inference_mode(False):
            x = _dil(shape, seed=seed, iters=iters, eta=eta, out_dtype=out_dtype, out_device=device)
        return ({"samples": x},)


class DIL2_EmptyLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31 - 1, "step": 1, "display": "number"}),
                "iters": ("INT", {"default": 2, "min": 0, "max": 8, "step": 1, "display": "slider"}),
                "eta": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.3, "step": 0.005, "display": "slider"}),
                "channels": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "beta": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 2.0, "step": 0.05, "display": "slider"}),
                "spectral_mix": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
                "use_scharr": ("BOOL", {"default": True}),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "make"
    CATEGORY = "latent"

    @staticmethod
    def _latent_wh(width, height):
        w = max(8, int(width)) // 8
        h = max(8, int(height)) // 8
        return w, h

    def make(
        self,
        width,
        height,
        batch_size=1,
        seed=-1,
        iters=2,
        eta=0.05,
        channels=4,
        beta=0.5,
        spectral_mix=0.5,
        use_scharr=True,
    ):
        device = mm.get_torch_device()
        out_dtype = _pick_out_dtype()
        wl, hl = self._latent_wh(width, height)
        shape = (int(batch_size), int(channels), hl, wl)
        with torch.inference_mode(False):
            x = _dil2(
                shape,
                seed=seed,
                iters=iters,
                eta=eta,
                beta=beta,
                spectral_mix=spectral_mix,
                use_scharr=use_scharr,
                out_dtype=out_dtype,
                out_device=device,
            )
        return ({"samples": x},)


NODE_CLASS_MAPPINGS = {
    "DIL_EmptyLatent": DIL_EmptyLatent,
    "DIL2_EmptyLatent": DIL2_EmptyLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DIL_EmptyLatent": "Dithered Isotropic Latent",
    "DIL2_EmptyLatent": "Dithered Isotropic Latent (Spectral)",
}
