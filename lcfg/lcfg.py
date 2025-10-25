import json
import struct
import folder_paths

class LoraConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_config",)
    FUNCTION = "extract_config"
    CATEGORY = "loaders"
    OUTPUT_NODE = True

    def _read_header(self, f):
        """Read safetensors header"""
        b = f.read(8)
        if len(b) != 8:
            raise ValueError("Invalid safetensors file")
        n = struct.unpack("<Q", b)[0]
        h = f.read(n)
        if len(h) != n:
            raise ValueError("Truncated metadata header")
        try:
            return json.loads(h.decode("utf-8"))
        except Exception:
            raise ValueError("Metadata header is not valid utf-8 json")

    def _extract_metadata(self, meta):
        """Extract metadata from header"""
        if isinstance(meta, dict):
            # Check for __metadata__ key
            m = meta.get("__metadata__")
            if isinstance(m, dict):
                return m
            # Check for metadata key
            if "metadata" in meta and isinstance(meta["metadata"], dict):
                return meta["metadata"]
            # Filter out tensor definitions (have dtype and shape)
            return {
                k: v for k, v in meta.items() 
                if not (isinstance(v, dict) and {"dtype", "shape"}.issubset(v.keys()))
            }
        return {}

    def extract_config(self, lora_name):
        """Extract configuration from LoRA file"""
        lora_path = folder_paths.get_full_path("loras", lora_name)
        
        if not lora_path:
            return (json.dumps({"error": "LoRA file not found"}, indent=2),)
        
        try:
            with open(lora_path, "rb") as f:
                meta = self._read_header(f)
            
            data = {"metadata": self._extract_metadata(meta)}
            json_output = json.dumps(data, indent=2, ensure_ascii=False)
            
            return (json_output,)
        
        except Exception as e:
            return (json.dumps({"error": str(e)}, indent=2),)


NODE_CLASS_MAPPINGS = {
    "LoraConfig": LoraConfig
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoraConfig": "Lora Config"
}