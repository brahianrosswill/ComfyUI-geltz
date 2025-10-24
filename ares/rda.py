# Residual-Delta Acceleration

import torch

def _rms(x): return torch.sqrt((x.float() * x.float()).mean() + 1e-8)

class RDA:
    def __init__(self, f, steps, tau=0.02, gamma=1.2, max_stale=2, w=1.0):
        self.f=f; self.n=max(1,int(steps)); self.tau=float(tau); self.gamma=float(gamma)
        self.max_stale=int(max_stale); self.w=float(w); self.i=0; self.y1=None; self.y2=None; self.stale=0
    @torch.no_grad()
    def __call__(self, x, sigma, **kw):
        p=min(1.0,max(0.0,self.i/self.n))
        if self.y1 is None or self.y2 is None or self.stale>=self.max_stale:
            y=self.f(x,sigma,**kw); self.y2=self.y1; self.y1=y; self.stale=0; self.i+=1; return y
        d=self.y1-self.y2; rel=_rms(d)/max(1e-8,_rms(self.y1)); thr=self.tau*((1.0-p+1e-3)**self.gamma)
        if rel<thr:
            y=self.y1+self.w*d
            s=torch.clamp(_rms(self.y1)/max(1e-8,_rms(y)),max=2.0).to(y); y=y*s
            self.y2=self.y1; self.y1=y; self.stale=min(self.stale+1,self.max_stale); self.i+=1; return y
        y=self.f(x,sigma,**kw); self.y2=self.y1; self.y1=y; self.stale=0; self.i+=1; return y

def wrap_with_rda(model_fn, sigmas, tau=0.02, gamma=1.2, max_stale=2, w=1.0):
    return RDA(model_fn, steps=max(1, len(sigmas)-1), tau=tau, gamma=gamma, max_stale=max_stale, w=w)
