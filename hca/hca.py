import math, gc, torch

def hca_scheduler(model_sampling, steps: int, gamma: float = 0.75) -> torch.FloatTensor:
    with torch.no_grad():
        s = model_sampling
        steps = int(max(1, steps))
        sigma_min, sigma_max = float(getattr(s, "sigma_min", 0.0)), float(getattr(s, "sigma_max", 0.0))
        device = getattr(s, "device", None)
        if device is None:
            model = getattr(s, "model", None)
            try: device = next(model.parameters()).device if model is not None else torch.device("cpu")
            except Exception: device = torch.device("cpu")
        if steps <= 1: return torch.tensor([sigma_max, 0.0], dtype=torch.float32, device=device)
        n = steps
        u = torch.linspace(0.0, 1.0, n, dtype=torch.float32, device=device)
        w = ((1.0 - torch.cos(math.pi * u)) * 0.5).pow(float(gamma))
        a0, a1 = math.atan(sigma_min), math.atan(sigma_max)
        sig_n = ((1.0 - w) * a1 + w * a0).tan()
        sig = torch.zeros(n + 1, dtype=torch.float32, device=device)
        sig[:-1] = sig_n
        sig[0] = torch.minimum(sig[0], torch.tensor(sigma_max, dtype=sig.dtype, device=device))
        eps = 1e-6
        for i in range(1, sig.shape[0] - 1):
            if sig[i] >= sig[i - 1]:
                dec = max(eps, eps * float(sig[i - 1].item()))
                sig[i] = max(0.0, float(sig[i - 1].item()) - dec)
        sig = torch.clamp(sig, min=0.0)
        sig[-1] = 0.0
        sig = sig.to(torch.float32).contiguous().clone().detach()
        sig.requires_grad_(False)
        return sig

def csu_flush_sigmas(*tensors: torch.Tensor) -> None:
    for t in tensors:
        if torch.is_tensor(t) and t.grad is not None: t.grad = None
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
