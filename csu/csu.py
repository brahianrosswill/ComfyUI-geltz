import math, gc, torch

def csu_scheduler(model_sampling, steps: int, gamma: float = 0.75, sgm: bool = False, floor: bool = False) -> torch.FloatTensor:
    with torch.no_grad():
        s = model_sampling
        steps = int(max(1, steps))
        t_start, t_end = float(s.timestep(s.sigma_max)), float(s.timestep(s.sigma_min))
        u = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float32)
        if sgm: u = u[:-1]
        w = ((1.0 - torch.cos(math.pi * u)) * 0.5).pow(float(gamma))
        t = t_start + (t_end - t_start) * w
        if floor: t = torch.floor(t)
        sig = s.sigma(t)
        sig = torch.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0).to(sig.dtype)
        sig = torch.cat([sig, torch.zeros(1, dtype=sig.dtype, device=sig.device)], dim=0) if sig.shape[0] != steps + 1 else sig.index_copy(0, torch.tensor([sig.shape[0]-1], device=sig.device), torch.zeros(1, dtype=sig.dtype, device=sig.device))
        eps = 1e-6
        for i in range(1, sig.shape[0] - 1):
            if sig[i] >= sig[i - 1]:
                dec = max(eps, eps * float(sig[i - 1].item()))
                sig[i] = max(0.0, float(sig[i - 1].item()) - dec)
        sig[0] = torch.minimum(sig[0], torch.tensor(float(s.sigma_max), dtype=sig.dtype, device=sig.device))
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
