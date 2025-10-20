# Regional Split Sampler

import torch
import nodes
from comfy.samplers import KSampler

def _build_soft_mask(h, w, center, feather, device, dtype):
    x = torch.linspace(0, 1, w, device=device, dtype=dtype).view(1, 1, 1, w).expand(1, 1, h, w)
    if feather <= 0.0:
        m = (x < center).to(dtype)
    else:
        a = center - feather
        b = center + feather
        t = ((x - a) / max(b - a, 1e-6)).clamp(0.0, 1.0)
        t = t * t * (3.0 - 2.0 * t)
        m = 1.0 - t
    return m

def _set_mask(cond, mask):
    out = []
    for c, data in cond:
        nd = dict(data)
        nd["mask"] = mask
        out.append([c, nd])
    return out

def regional_split_sample(model, seed, steps, cfg, sampler_name, scheduler, positive_left, positive_right, negative, latent_image, denoise=1.0, center=0.5, feather=0.25):
    x = latent_image["samples"]
    b, _, h, w = x.shape
    device = x.device
    dtype = x.dtype
    m = _build_soft_mask(h, w, float(center), float(feather), device, dtype)
    m_cpu = m.squeeze(0).squeeze(0).to(torch.float32).cpu()
    pl = _set_mask(positive_left, m_cpu)
    pr = _set_mask(positive_right, (1.0 - m_cpu))
    pos = pl + pr
    out_lat = nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, pos, negative, latent_image, denoise)[0]
    return (out_lat,)

class RegionalSplitSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (KSampler.SAMPLERS,),
                "scheduler": (KSampler.SCHEDULERS,),
                "positive_left": ("CONDITIONING",),
                "positive_right": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.001}),
                "feather": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.5, "step": 0.001}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling"

    def sample(self, model, seed, steps, cfg, sampler_name, scheduler, positive_left, positive_right, negative, latent_image, denoise, center, feather):
        return regional_split_sample(model, seed, steps, cfg, sampler_name, scheduler, positive_left, positive_right, negative, latent_image, denoise, center, feather)

NODE_CLASS_MAPPINGS = {"RegionalSplitSampler": RegionalSplitSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"RegionalSplitSampler": "Regional Split Sampler"}



