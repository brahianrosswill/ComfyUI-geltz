import json
import re
import os
import folder_paths
from PIL import Image
import torch
import numpy as np

def oneline(s):
    return " ".join((s or "").split()).lower()

def parse_a1111_parameters(s):
    lines = s.strip().split("\n")
    d = {"prompt": oneline(lines[0]), "negative_prompt": "", "settings": {}}
    if len(lines) > 1 and lines[1].startswith("Negative prompt:"):
        d["negative_prompt"] = oneline(lines[1].split(":", 1)[1])
        st = "\n".join(lines[2:])
    else:
        st = "\n".join(lines[1:])
    for k, v in re.findall(r"([^,]+?):\s*([^,]+)(?:,|$)", st):
        d["settings"][k.strip().lower()] = oneline(v)
    return d

def parse_comfyui_metadata(s):
    try:
        wf = json.loads(s)
    except Exception:
        return {"prompt": "", "negative_prompt": "", "settings": {}}
    ks = mdl = lat = None
    for _, nd in wf.items():
        t = nd.get("class_type")
        if t == "KSampler":
            ks = nd
        elif t == "CheckpointLoaderSimple":
            mdl = nd
        elif t == "EmptyLatentImage":
            lat = nd
    if not ks:
        return {"prompt": "", "negative_prompt": "", "settings": {}}
    inp = ks.get("inputs", {})
    st = {
        "steps": inp.get("steps"),
        "cfg scale": inp.get("cfg"),
        "sampler": inp.get("sampler_name"),
        "scheduler": inp.get("scheduler"),
        "seed": inp.get("seed")
    }
    if mdl:
        st["model"] = mdl["inputs"].get("ckpt_name", "unknown")
    if lat:
        w, h = lat["inputs"].get("width"), lat["inputs"].get("height")
        if w and h:
            st["size"] = f"{w}x{h}"
    
    def txt(ref):
        if ref and isinstance(ref, list) and ref:
            nd = wf.get(str(ref[0]), {})
            return oneline(nd.get("inputs", {}).get("text", ""))
        return ""
    
    return {
        "prompt": txt(inp.get("positive")),
        "negative_prompt": txt(inp.get("negative")),
        "settings": {k.lower(): oneline(str(v)) for k, v in st.items() if v is not None}
    }

def parse_universal_metadata(meta):
    r = {"prompt": "", "negative_prompt": "", "settings": {}, "source": "unknown"}
    if "parameters" in meta:
        try:
            p = parse_a1111_parameters(meta["parameters"])
            p["source"] = "automatic1111"
            return p
        except Exception:
            pass
    if "prompt" in meta:
        try:
            p = parse_comfyui_metadata(meta["prompt"])
            p["source"] = "comfyui"
            return p
        except Exception:
            pass
    for k, v in meta.items():
        if isinstance(v, str) and "negative prompt:" in v:
            try:
                p = parse_a1111_parameters(v)
                p["source"] = f"generic ({k.lower()})"
                return p
            except Exception:
                pass
    if meta:
        r["settings"] = {
            k.lower(): oneline(str(v))
            for k, v in meta.items()
            if k.lower() not in ["software", "dpi"]
        }
    return r

def format_output_unified(d):
    lines = []
    lines.append(f"[source detected: {oneline(d.get('source', 'unknown'))}]")
    pos = d.get("prompt", "") or "(not found)"
    neg = d.get("negative_prompt", "") or "(not found)"
    lines.append(f"\npositive prompt:\n{pos}")
    lines.append(f"\nnegative prompt:\n{neg}")
    lines.append("\ngeneration settings:")
    if d.get("settings"):
        for k, v in d["settings"].items():
            if v is not None:
                lines.append(f"  {k}: {str(v)}")
    else:
        lines.append("  (not found)")
    return "\n".join(lines)

class LoadImageWithMetadata:
    """
    Loads an image and extracts its metadata (prompts and generation settings).
    Works with Automatic1111, ComfyUI, and other AI-generated images.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "metadata")
    FUNCTION = "load_image"
    CATEGORY = "image"

    def load_image(self, image):
        input_dir = folder_paths.get_input_directory()
        image_path = folder_paths.get_annotated_filepath(image)
        
        try:
            # Open image with PIL to extract metadata
            img = Image.open(image_path)
            
            # Extract metadata from PNG info
            raw_meta = img.info if hasattr(img, 'info') else {}
            
            # Parse metadata
            if raw_meta:
                parsed = parse_universal_metadata(raw_meta)
                metadata_text = format_output_unified(parsed)
            else:
                metadata_text = "no special metadata found in this image."
            
            # Convert to RGB if needed
            if img.mode == 'I':
                img = img.point(lambda i: i * (1 / 255))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to tensor format expected by ComfyUI
            image_np = np.array(img).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            
            # Create mask if image has alpha channel
            if 'A' in img.getbands():
                mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            
            return (image_tensor, mask, metadata_text)
            
        except Exception as e:
            # Return error as metadata
            error_msg = f"error loading image or extracting metadata: {str(e)}"
            # Create a dummy image
            dummy = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            dummy_mask = torch.zeros((64, 64), dtype=torch.float32)
            return (dummy, dummy_mask, error_msg)

class ImageMetadataExtractor:
    """
    Extracts metadata from an image path/file.
    For use with images that still have their original metadata intact.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("metadata",)
    FUNCTION = "extract_metadata"
    CATEGORY = "image/analysis"
    OUTPUT_NODE = True

    def extract_metadata(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        
        try:
            # Open image with PIL to extract metadata
            with Image.open(image_path) as img:
                raw_meta = img.info if hasattr(img, 'info') else {}
                
                if not raw_meta:
                    return ("no special metadata found in this image.",)
                
                # Parse metadata
                parsed = parse_universal_metadata(raw_meta)
                
                # Format output
                output_text = format_output_unified(parsed)
                
                return (output_text,)
            
        except Exception as e:
            return (f"error extracting metadata: {str(e)}",)

# Node registration
NODE_CLASS_MAPPINGS = {
    "LoadImageWithMetadata": LoadImageWithMetadata,
    "ImageMetadataExtractor": ImageMetadataExtractor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageWithMetadata": "Load Image With Metadata",
    "ImageMetadataExtractor": "Image Metadata Extractor"
}