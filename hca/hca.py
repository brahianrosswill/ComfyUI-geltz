import math
import gc
import torch

def hca_scheduler(model_sampling, steps: int, gamma: float = 0.75) -> torch.FloatTensor:
    with torch.no_grad():
        s = model_sampling
        steps = int(max(1, steps))

        sigma_min = float(getattr(s, "sigma_min", 0.0))
        sigma_max = float(getattr(s, "sigma_max", 0.0))
        device = getattr(s, "device", None)
        if device is None:
            model = getattr(s, "model", None)
            try:
                device = next(model.parameters()).device if model is not None else torch.device("cpu")
            except Exception:
                device = torch.device("cpu")

        if steps <= 1:
            return torch.tensor([float(sigma_max), 0.0], dtype=torch.float32, device=device)

        n = steps

        u = torch.linspace(0.0, 1.0, n, dtype=torch.float32, device=device)

        w = ((1.0 - torch.cos(math.pi * u)) * 0.5).pow(float(gamma))

        atan_min = math.atan(sigma_min)
        atan_max = math.atan(sigma_max)
        interp_atan = (1.0 - w) * atan_max + w * atan_min

        sigmas_n = interp_atan.tan()

        sigmas = torch.zeros(n + 1, dtype=torch.float32, device=device)
        sigmas[:-1] = sigmas_n

        sig0_limit = torch.tensor(float(sigma_max), dtype=sigmas.dtype, device=device)
        sigmas[0] = torch.minimum(sigmas[0], sig0_limit)

        eps = 1e-6
        for i in range(1, sigmas.shape[0] - 1):
            if sigmas[i] >= sigmas[i - 1]:
                dec = max(eps, eps * float(sigmas[i - 1].item()))
                sigmas[i] = max(0.0, float(sigmas[i - 1].item()) - dec)

        sigmas = torch.clamp(sigmas, min=0.0)
        sigmas[-1] = 0.0

        sigmas = sigmas.to(torch.float32).contiguous().clone().detach()
        sigmas.requires_grad_(False)
        return sigmas

def csu_flush_sigmas(*tensors: torch.Tensor) -> None:
    for t in tensors:
        if torch.is_tensor(t):
            if t.grad is not None:
                t.grad = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()