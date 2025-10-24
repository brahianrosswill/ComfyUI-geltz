import os
import tempfile
from pathlib import Path
import torch
from torch import Tensor
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar

try:
    import sd_mecha
    from sd_mecha import merge_method, Parameter, Return
    SD_MECHA_AVAILABLE = True
except ImportError:
    SD_MECHA_AVAILABLE = False
    print("sd_mecha not found. please install: pip install sd-mecha")


def _mad(x: Tensor, eps: Tensor) -> Tensor:
    f = x.flatten()
    m = f.median()
    return (f - m).abs().median().clamp_min(eps)


def _trust_clamp(a: Tensor, y: Tensor, trust_k: float, eps: Tensor) -> Tensor:
    r = float(trust_k) * _mad(a, eps)
    return a + (y - a).clamp(-r, r)


def _finite_or_a(a: Tensor, y: Tensor) -> Tensor:
    return torch.where(torch.isfinite(y), y, a)


if SD_MECHA_AVAILABLE:
    @merge_method
    def orbit(
        a: Parameter(Tensor),
        b: Parameter(Tensor),
        alpha_par: Parameter(float) = 0.20,
        alpha_orth: Parameter(float) = 0.60,
        trust_k: Parameter(float) = 3.0,
        eps: Parameter(float) = 1e-8,
        coef_clip: Parameter(float) = 8.0,
    ) -> Return(Tensor):
        e = torch.as_tensor(float(eps), device=a.device, dtype=a.dtype)
        wp = torch.as_tensor(float(alpha_par), device=a.device, dtype=a.dtype)
        wo = torch.as_tensor(float(alpha_orth), device=a.device, dtype=a.dtype)
        af, bf = a.flatten(), b.flatten()
        den = (af @ af).clamp_min(e)
        coef = (bf @ af) / den
        if float(coef_clip) > 0.0:
            c = torch.as_tensor(float(coef_clip), device=a.device, dtype=a.dtype)
            coef = coef.clamp(-c, c)
        bp = coef * a
        bo = b - bp
        y = a + wp * (bp - a) + wo * bo
        y = _trust_clamp(a, y, trust_k, e)
        y = _finite_or_a(a, y)
        return y


def orbit_merge_state_dicts(sd_a, sd_b, alpha_par, alpha_orth, trust_k, eps, coef_clip, pbar=None):
    m = {}
    k = set(sd_a.keys()) & set(sd_b.keys())
    for key in k:
        ta, tb = sd_a[key], sd_b[key]
        if pbar:
            pbar.update(1)
        if isinstance(ta, Tensor) and isinstance(tb, Tensor) and ta.shape == tb.shape:
            # Ensure both tensors are on the same device
            if ta.device != tb.device:
                tb = tb.to(ta.device)
            
            e = torch.as_tensor(float(eps), device=ta.device, dtype=ta.dtype)
            wp = torch.as_tensor(float(alpha_par), device=ta.device, dtype=ta.dtype)
            wo = torch.as_tensor(float(alpha_orth), device=ta.device, dtype=ta.dtype)
            af, bf = ta.flatten(), tb.flatten()
            den = (af @ af).clamp_min(e)
            coef = (bf @ af) / den
            if float(coef_clip) > 0.0:
                c = torch.as_tensor(float(coef_clip), device=ta.device, dtype=ta.dtype)
                coef = coef.clamp(-c, c)
            bp = coef * ta
            bo = tb - bp
            y = ta + wp * (bp - ta) + wo * bo
            y = _trust_clamp(ta, y, trust_k, e)
            y = _finite_or_a(ta, y)
            m[key] = y
        else:
            m[key] = ta
    for key in sd_a.keys():
        if key not in m:
            m[key] = sd_a[key]
    return m


class ORBITModelMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_a": ("MODEL", {}),
                "model_b": ("MODEL", {}),
                "clip_a": ("CLIP", {}),
                "clip_b": ("CLIP", {}),
                "alpha_parallel": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "alpha_orthogonal": ("FLOAT", {"default": 0.50, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "trust_k": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "coef_clip": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 20.0, "step": 0.5}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "merge_models"
    CATEGORY = "advanced/model_merging"

    def merge_models(self, model_a, model_b, clip_a, clip_b, alpha_parallel, alpha_orthogonal, trust_k, coef_clip):
        print(f"\nORBIT Merge α∥={alpha_parallel:.2f} α⊥={alpha_orthogonal:.2f} trust_k={trust_k:.1f} coef_clip={coef_clip:.1f}")
        mmA, mmB = model_a.clone(), model_b
        mcA, mcB = clip_a.clone(), clip_b
        sdA, sdB = model_a.model.state_dict(), model_b.model.state_dict()
        keys = set(sdA.keys()) & set(sdB.keys())
        print(f"   Merging {len(keys)} MODEL tensors...")
        p = ProgressBar(len(keys))
        sdM = orbit_merge_state_dicts(sdA, sdB, alpha_parallel, alpha_orthogonal, trust_k, 1e-8, coef_clip, p)
        mmA.model.load_state_dict(sdM, strict=False)
        try:
            cA, cB = clip_a.cond_stage_model.state_dict(), clip_b.cond_stage_model.state_dict()
            print(f"   merging {len(set(cA.keys()) & set(cB.keys()))} CLIP tensors...")
            sdCM = orbit_merge_state_dicts(cA, cB, alpha_parallel, alpha_orthogonal, trust_k, 1e-8, coef_clip)
            mcA.cond_stage_model.load_state_dict(sdCM, strict=False)
        except Exception as e:
            print(f"CLIP merge skipped: {e}")
            mcA = clip_a.clone()
        print("ORBIT merge complete.\n")
        return mmA, mcA

NODE_CLASS_MAPPINGS = {"ORBITModelMerge": ORBITModelMerge}
NODE_DISPLAY_NAME_MAPPINGS = {"ORBITModelMerge": "ORBIT Merge"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
