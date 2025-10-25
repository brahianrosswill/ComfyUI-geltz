import torch
import numpy as np
from PIL import Image
from sklearn.cluster import MiniBatchKMeans

class ColorPaletteExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_colors": ("INT", {
                    "default": 8,
                    "min": 2,
                    "max": 16,
                    "step": 1
                }),
                "square_layout": ("BOOLEAN", {
                    "default": False
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("palette_image", "hex_colors")
    FUNCTION = "extract_palette"
    CATEGORY = "image/analysis"
    OUTPUT_NODE = True

    def extract_palette(self, image, num_colors, square_layout):
        # Convert ComfyUI tensor to numpy array
        # ComfyUI format: (batch, height, width, channels) in range [0, 1]
        img_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        
        # Downsample for faster processing
        h, w = img_np.shape[:2]
        max_dim = 300
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img_pil = Image.fromarray(img_np)
            img_pil = img_pil.resize((new_w, new_h), Image.BILINEAR)
            img_np = np.array(img_pil)
        
        # Reshape image to list of pixels
        pixels = img_np.reshape(-1, 3)
        
        # Sample pixels for even faster processing
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]
        
        # Use MiniBatchKMeans for faster clustering
        kmeans = MiniBatchKMeans(n_clusters=num_colors, random_state=42, batch_size=1000, n_init=3)
        kmeans.fit(pixels)
        
        # Get the colors
        colors = kmeans.cluster_centers_.astype(int)
        
        # Sort colors by frequency
        labels = kmeans.labels_
        label_counts = np.bincount(labels)
        sorted_indices = np.argsort(-label_counts)
        colors = colors[sorted_indices]
        
        # Convert to hex
        hex_colors = ['#{:02x}{:02x}{:02x}'.format(r, g, b) for r, g, b in colors]
        
        # Create palette image
        square_size = 100
        
        if square_layout:
            # Calculate grid dimensions for square layout
            cols = int(np.ceil(np.sqrt(num_colors)))
            rows = int(np.ceil(num_colors / cols))
            palette_width = square_size * cols
            palette_height = square_size * rows
            
            palette_array = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
            
            for i, color in enumerate(colors):
                row = i // cols
                col = i % cols
                y_start = row * square_size
                y_end = y_start + square_size
                x_start = col * square_size
                x_end = x_start + square_size
                palette_array[y_start:y_end, x_start:x_end] = color
            
        else:
            # Horizontal strip layout
            palette_width = square_size * num_colors
            palette_height = square_size
            palette_array = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
            
            for i, color in enumerate(colors):
                x_start = i * square_size
                x_end = x_start + square_size
                palette_array[:, x_start:x_end] = color
        
        # Convert to ComfyUI tensor format
        palette_tensor = torch.from_numpy(palette_array).float() / 255.0
        palette_tensor = palette_tensor.unsqueeze(0)  # Add batch dimension
        
        # Join hex colors as string
        hex_string = ','.join(hex_colors)
        
        return (palette_tensor, hex_string)

# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "ColorPaletteExtractor": ColorPaletteExtractor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ColorPaletteExtractor": "Color Palette Extractor"
}