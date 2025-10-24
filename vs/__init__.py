import torch

class VelocityScaling:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "strength": ("FLOAT", {"default": 1.005, "min": 0.9, "max": 1.2, "step": 0.001})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "model/patches"
    TITLE = "Velocity Scaling"

    def apply(self, model, strength: float):
        m = model.clone()
        def vs_out(x: torch.Tensor, extra_options):
            mt = None
            if isinstance(extra_options, dict):
                mt = extra_options.get("model_type") or extra_options.get("prediction_type")
                s = extra_options.get("sampling")
                if mt is None and isinstance(s, dict): mt = s.get("prediction_type")
            return x / strength if isinstance(mt, str) and mt.lower().startswith("v") else x
        if hasattr(m, "set_model_output_patch"): m.set_model_output_patch(vs_out)
        else:
            opts = dict(getattr(m, "model_options", {})); opts["model_output_patch"] = vs_out; m.model_options = opts
        return (m,)

NODE_CLASS_MAPPINGS = {"VelocityScaling": VelocityScaling}
NODE_DISPLAY_NAME_MAPPINGS = {"VelocityScaling": "Velocity Scaling"}
