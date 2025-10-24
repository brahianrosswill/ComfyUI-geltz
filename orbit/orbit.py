# ORBIT - Orthogonal Residual Blend In Tensors

import torch
import folder_paths
import comfy.model_management as mm
from comfy.utils import ProgressBar
import os
import tempfile
from pathlib import Path
from torch import Tensor

# Import sd_mecha and the orbit merge method
try:
    import sd_mecha
    from sd_mecha import merge_method, Parameter, Return
    SD_MECHA_AVAILABLE = True
except ImportError:
    SD_MECHA_AVAILABLE = False
    print("sd_mecha not found. Please install: pip install sd-mecha")

# ORBIT merge method implementation
def _mad(x: Tensor, eps: Tensor) -> Tensor:
    flat = x.flatten()
    med = flat.median()
    return (flat - med).abs().median().clamp_min(eps)

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
        # how much to move along A's direction vs inject orthogonal novelty
        alpha_par: Parameter(float) = 0.20,
        alpha_orth: Parameter(float) = 0.60,
        # robustness knobs
        trust_k: Parameter(float) = 3.0,
        eps: Parameter(float) = 1e-8,
        # optional: cap the parallel projection coefficient (stability)
        coef_clip: Parameter(float) = 8.0,
    ) -> Return(Tensor):
        """
        a: tensor from model A (kept as structure anchor)
        b: tensor from model B (feature donor)
        returns: merged tensor, dtype/device preserved
        """
        # scalar tensors in a's device/dtype
        eps_t = torch.as_tensor(float(eps), device=a.device, dtype=a.dtype)
        w_par = torch.as_tensor(float(alpha_par), device=a.device, dtype=a.dtype)
        w_orth = torch.as_tensor(float(alpha_orth), device=a.device, dtype=a.dtype)

        # compute parallel/orthogonal decomposition of B w.r.t. A
        af = a.flatten()
        bf = b.flatten()

        # coef = <B,A> / <A,A>
        denom = (af @ af).clamp_min(eps_t)
        coef = (bf @ af) / denom

        # optional stability: clip the projection coefficient
        if float(coef_clip) > 0.0:
            cmax = torch.as_tensor(float(coef_clip), device=a.device, dtype=a.dtype)
            coef = coef.clamp(-cmax, cmax)

        b_par = coef * a
        b_orth = b - b_par

        # blend: keep A, adjust slightly toward B along A (parallel),
        # and inject orthogonal novelty (structure-preserving change)
        y = a + w_par * (b_par - a) + w_orth * b_orth

        # robust trust clamp around A using MAD, and non-finite fallback
        y = _trust_clamp(a, y, trust_k, eps_t)
        y = _finite_or_a(a, y)
        return y


def orbit_merge_state_dicts(sd_a, sd_b, alpha_par, alpha_orth, trust_k, eps, coef_clip, pbar=None):
    """
    Apply ORBIT merge directly to state dicts
    """
    sd_merged = {}
    keys = set(sd_a.keys()) & set(sd_b.keys())
    
    for key in keys:
        tensor_a = sd_a[key]
        tensor_b = sd_b[key]
        if pbar: pbar.update(1)
        
        if isinstance(tensor_a, Tensor) and isinstance(tensor_b, Tensor):
            if tensor_a.shape == tensor_b.shape:
                # Apply ORBIT merge
                eps_t = torch.as_tensor(float(eps), device=tensor_a.device, dtype=tensor_a.dtype)
                w_par = torch.as_tensor(float(alpha_par), device=tensor_a.device, dtype=tensor_a.dtype)
                w_orth = torch.as_tensor(float(alpha_orth), device=tensor_a.device, dtype=tensor_a.dtype)

                af = tensor_a.flatten()
                bf = tensor_b.flatten()

                denom = (af @ af).clamp_min(eps_t)
                coef = (bf @ af) / denom

                if float(coef_clip) > 0.0:
                    cmax = torch.as_tensor(float(coef_clip), device=tensor_a.device, dtype=tensor_a.dtype)
                    coef = coef.clamp(-cmax, cmax)

                b_par = coef * tensor_a
                b_orth = tensor_b - b_par

                y = tensor_a + w_par * (b_par - tensor_a) + w_orth * b_orth
                y = _trust_clamp(tensor_a, y, trust_k, eps_t)
                y = _finite_or_a(tensor_a, y)
                
                sd_merged[key] = y
            else:
                sd_merged[key] = tensor_a
        else:
            sd_merged[key] = tensor_a
    
    # Copy keys only in A
    for key in sd_a.keys():
        if key not in sd_merged:
            sd_merged[key] = sd_a[key]
    
    return sd_merged


class ORBITModelMerge:
    """
    ORBIT: Orthogonal Residual Blend In Tensors
    Injects orthogonal novelty from Model B into Model A's structure
    Outputs merged MODEL and CLIP for use with Save Checkpoint node
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_a": ("MODEL", {"tooltip": "Base model (structure anchor)"}),
                "model_b": ("MODEL", {"tooltip": "Donor model (feature source)"}),
                "clip_a": ("CLIP", {"tooltip": "CLIP from model A"}),
                "clip_b": ("CLIP", {"tooltip": "CLIP from model B"}),
                "alpha_parallel": ("FLOAT", {
                    "default": 0.25,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "alpha_orthogonal": ("FLOAT", {
                    "default": 0.50,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider",
                }),
                "trust_k": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.5,
                    "max": 10.0,
                    "step": 0.1,
                }),
                "coef_clip": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.5,
                }),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP")
    RETURN_NAMES = ("model", "clip")
    FUNCTION = "merge_models"
    CATEGORY = "advanced/model_merging"
    
    def merge_models(self, model_a, model_b, clip_a, clip_b, 
                     alpha_parallel, alpha_orthogonal, trust_k, coef_clip):
        """
        Merge two models using ORBIT algorithm
        """
        print(f"\n🪐 ORBIT Merge Starting...")
        print(f"   α∥={alpha_parallel:.2f} α⊥={alpha_orthogonal:.2f}")
        print(f"   trust_k={trust_k:.1f} coef_clip={coef_clip:.1f}")
        
        # Clone model A as base
        merged_model = model_a.clone()
        merged_clip = clip_a.clone()
        
        # Get state dicts
        sd_a = model_a.model.state_dict()
        sd_b = model_b.model.state_dict()
        
        # Get common keys
        keys = set(sd_a.keys()) & set(sd_b.keys())
        total_keys = len(keys)
        
        print(f"   Merging {total_keys} MODEL tensors...")
        
        # Progress tracking
        pbar=ProgressBar(total_keys)
        sd_merged=orbit_merge_state_dicts(sd_a, sd_b, alpha_parallel, alpha_orthogonal, trust_k, 1e-8, coef_clip, pbar)
        
        # Load merged state dict
        merged_model.model.load_state_dict(sd_merged, strict=False)
        
        # Merge CLIP similarly
        try:
            sd_clip_a = clip_a.cond_stage_model.state_dict()
            sd_clip_b = clip_b.cond_stage_model.state_dict()
            
            clip_keys = set(sd_clip_a.keys()) & set(sd_clip_b.keys())
            print(f"   Merging {len(clip_keys)} CLIP tensors...")
            
            sd_clip_merged = orbit_merge_state_dicts(
                sd_clip_a, sd_clip_b,
                alpha_parallel, alpha_orthogonal,
                trust_k, 1e-8, coef_clip
            )
            
            merged_clip.cond_stage_model.load_state_dict(sd_clip_merged, strict=False)
        except Exception as e:
            print(f"CLIP merge skipped: {e}")
            merged_clip = clip_a.clone()
        
        print(f"ORBIT merge complete!\n")
        
        return (merged_model, merged_clip)


# Web UI Extensions for Saturn animation
WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "ORBITModelMerge": ORBITModelMerge,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ORBITModelMerge": "ORBIT Model Merge 🪐",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']