import torch


def _rms(x):
    """Calculate root mean square with numerical stability."""
    return torch.sqrt((x.float() * x.float()).mean() + 1e-8)


class RDA:
    """
    Residual-Delta Acceleration optimizer wrapper.
    
    Accelerates iterative processes by reusing previous predictions when
    the residual delta is sufficiently small.
    """
    
    def __init__(self, f, steps, tau=0.02, gamma=1.2, max_stale=2, w=1.0):
        """
        Initialize RDA wrapper.
        
        Args:
            f: Function to wrap (typically a model forward pass)
            steps: Total number of steps
            tau: Threshold coefficient for residual comparison
            gamma: Decay exponent for threshold schedule
            max_stale: Maximum number of consecutive reused predictions
            w: Weight for delta extrapolation
        """
        self.f = f
        self.n = max(1, int(steps))
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.max_stale = int(max_stale)
        self.w = float(w)
        
        self.i = 0
        self.y1 = None
        self.y2 = None
        self.stale = 0
    
    @torch.no_grad()
    def __call__(self, x, sigma, **kw):
        """
        Execute one step with potential acceleration.
        
        Args:
            x: Input tensor
            sigma: Noise level
            **kw: Additional keyword arguments passed to wrapped function
            
        Returns:
            Prediction (either computed or extrapolated)
        """
        p = min(1.0, max(0.0, self.i / self.n))
        
        if self.y1 is None or self.y2 is None or self.stale >= self.max_stale:
            y = self.f(x, sigma, **kw)
            self.y2 = self.y1
            self.y1 = y
            self.stale = 0
            self.i += 1
            return y
        
        d = self.y1 - self.y2
        rel = _rms(d) / max(1e-8, _rms(self.y1))
        thr = self.tau * ((1.0 - p + 1e-3) ** self.gamma)
        
        if rel < thr:
            y = self.y1 + self.w * d
            s = torch.clamp(_rms(self.y1) / max(1e-8, _rms(y)), max=2.0).to(y)
            y = y * s
            
            self.y2 = self.y1
            self.y1 = y
            self.stale = min(self.stale + 1, self.max_stale)
            self.i += 1
            return y
        
        y = self.f(x, sigma, **kw)
        self.y2 = self.y1
        self.y1 = y
        self.stale = 0
        self.i += 1
        return y


def wrap_with_rda(model_fn, sigmas, tau=0.07, gamma=1.0, max_stale=2, w=0.8):
    """
    Convenience function to wrap a model function with RDA acceleration.
    
    Args:
        model_fn: Model function to wrap
        sigmas: Noise schedule (used to determine step count)
        tau: Threshold coefficient
        gamma: Decay exponent
        max_stale: Maximum consecutive reused predictions
        w: Extrapolation weight
        
    Returns:
        RDA wrapper instance
    """
    return RDA(
        model_fn,
        steps=max(1, (len(sigmas) - 1)*3),
        tau=tau,
        gamma=gamma,
        max_stale=max_stale,
        w=w
    )