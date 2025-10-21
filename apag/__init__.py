import torch
import torch.nn.functional as F

class _SDPAPerturb:
    def __init__(self, s):
        self.s = float(s)
        self._orig = None
    def __enter__(self):
        self._orig = F.scaled_dot_product_attention
        s = self.s
        def patched(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
            if s <= 0.0:
                return self._orig(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
            v_mean = v.mean(dim=-2, keepdim=True)
            v_pert = (1.0 - s) * v + s * v_mean
            return self._orig(q, k, v_pert, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
        F.scaled_dot_product_attention = patched
        return self
    def __exit__(self, exc_type, exc, tb):
        F.scaled_dot_product_attention = self._orig

def _adaptive_pag(unet_apply, params, s_fixed=0.5):
    x = params["input"]
    t = params["timestep"]
    c = params["c"]
    eps = unet_apply(x=x, t=t, **c)
    with _SDPAPerturb(s_fixed):
        eps_hat = unet_apply(x=x, t=t, **c)
    return eps + s_fixed * (eps - eps_hat)

class _APAGWrapper:
    def __init__(self, strength):
        self.strength = float(strength)
    def __call__(self, unet_apply, params):
        return _adaptive_pag(unet_apply, params, s_fixed=self.strength)

class APAGModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/patches"

    def patch(self, model, strength):
        m = model.clone()
        m.set_model_unet_function_wrapper(_APAGWrapper(strength))
        return (m,)

NODE_CLASS_MAPPINGS = {"APAGModelPatch": APAGModelPatch}
NODE_DISPLAY_NAME_MAPPINGS = {"APAGModelPatch": "Adaptive PAG"}
