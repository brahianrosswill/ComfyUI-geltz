import torch
import torch.nn.functional as F

class KuwaharaFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "radius": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1, "display": "slider"}),
                "strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_kuwahara"
    CATEGORY = "image/filters"

    def apply_kuwahara(self, image, radius, strength):
        img = image.permute(0, 3, 1, 2)
        filt = self.kuwahara(img, radius).permute(0, 2, 3, 1)
        return (image * (1 - strength) + filt * strength,)

    def kuwahara(self, img, r):
        b, c, h, w = img.shape
        k = r + 1
        kernel = torch.ones(1, 1, k, k, device=img.device) / (k * k)
        p = F.pad(img, (r, r, r, r), mode="reflect")
        q1 = p[:, :, :h + r, :w + r]
        q2 = p[:, :, :h + r, r:w + 2 * r]
        q3 = p[:, :, r:h + 2 * r, :w + r]
        q4 = p[:, :, r:h + 2 * r, r:w + 2 * r]
        cat = torch.cat([q1, q2, q3, q4], 0)
        mean = F.conv2d(cat, kernel.expand(c, 1, -1, -1), groups=c)
        var = F.conv2d(cat * cat, kernel.expand(c, 1, -1, -1), groups=c) - mean * mean
        mean = mean.view(4, b, c, h, w)
        var = var.sum(1, keepdim=True).view(4, b, 1, h, w)
        idx = var.argmin(0)
        mask = F.one_hot(idx.squeeze(1), 4).permute(3, 0, 1, 2).unsqueeze(2).to(mean.dtype)
        return (mean * mask).sum(0)

NODE_CLASS_MAPPINGS = {"KuwaharaFilter": KuwaharaFilter}
NODE_DISPLAY_NAME_MAPPINGS = {"KuwaharaFilter": "Kuwahara Filter"}
