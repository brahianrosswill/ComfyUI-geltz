import torch
import numpy as np

class tmp:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "temperature": ("FLOAT", {
                    "default": 0.0,
                    "min": -1.0,
                    "max": 1.0,
                    "step": 0.01,
                    "display": "slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "adjust_temperature"
    CATEGORY = "image/color"
    
    def adjust_temperature(self, image, temperature):
        # Clone to avoid modifying original
        img = image.clone()
        
        # Convert to numpy for processing
        img_np = img.cpu().numpy()
        batch_size = img_np.shape[0]
        
        for b in range(batch_size):
            frame = img_np[b]
            
            # Convert RGB to LAB color space for perceptual processing
            lab = self.rgb_to_lab(frame)
            
            # Apply temperature adjustment in LAB space (reduced intensity)
            if temperature > 0:  # Warm
                # Shift toward warm (yellow-orange)
                lab[:, :, 1] += temperature * 20  # A channel (green-red) - reduced from 30
                lab[:, :, 2] += temperature * 25  # B channel (blue-yellow) - reduced from 40
                
                # Subtle luminance adjustment to prevent washout
                lab[:, :, 0] *= (1.0 + temperature * 0.03)  # reduced from 0.05
                
            else:  # Cool
                t = abs(temperature)
                # Shift toward cool (blue-cyan)
                lab[:, :, 1] -= t * 12  # A channel - reduced from 20
                lab[:, :, 2] -= t * 30  # B channel - reduced from 50
                
                # Slight luminance increase for cooler feel
                lab[:, :, 0] *= (1.0 + t * 0.05)  # reduced from 0.08
            
            # Convert back to RGB
            rgb = self.lab_to_rgb(lab)
            
            # Apply subtle saturation adjustment
            hsv = self.rgb_to_hsv(rgb)
            if temperature > 0:
                # Increase saturation slightly for warm
                hsv[:, :, 1] *= (1.0 + temperature * 0.1)  # reduced from 0.15
            else:
                # Decrease saturation slightly for cool
                hsv[:, :, 1] *= (1.0 + temperature * 0.06)  # reduced from 0.1
            
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 1)
            rgb = self.hsv_to_rgb(hsv)
            
            img_np[b] = np.clip(rgb, 0, 1)
        
        return (torch.from_numpy(img_np).to(image.device),)
    
    def rgb_to_lab(self, rgb):
        """Convert RGB to LAB color space"""
        # First convert to XYZ
        xyz = self.rgb_to_xyz(rgb)
        
        # Then XYZ to LAB
        xyz = xyz / np.array([0.95047, 1.0, 1.08883])
        
        mask = xyz > 0.008856
        xyz = np.where(mask, np.power(xyz, 1/3), (7.787 * xyz) + (16/116))
        
        lab = np.zeros_like(rgb)
        lab[:, :, 0] = (116 * xyz[:, :, 1]) - 16  # L
        lab[:, :, 1] = 500 * (xyz[:, :, 0] - xyz[:, :, 1])  # A
        lab[:, :, 2] = 200 * (xyz[:, :, 1] - xyz[:, :, 2])  # B
        
        return lab
    
    def lab_to_rgb(self, lab):
        """Convert LAB back to RGB"""
        # LAB to XYZ
        fy = (lab[:, :, 0] + 16) / 116
        fx = lab[:, :, 1] / 500 + fy
        fz = fy - lab[:, :, 2] / 200
        
        xyz = np.stack([fx, fy, fz], axis=2)
        
        mask = xyz > 0.2068966
        xyz = np.where(mask, np.power(xyz, 3), (xyz - 16/116) / 7.787)
        xyz = xyz * np.array([0.95047, 1.0, 1.08883])
        
        # XYZ to RGB
        return self.xyz_to_rgb(xyz)
    
    def rgb_to_xyz(self, rgb):
        """Convert RGB to XYZ color space"""
        mask = rgb > 0.04045
        rgb_linear = np.where(mask, np.power((rgb + 0.055) / 1.055, 2.4), rgb / 12.92)
        
        matrix = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041]
        ])
        
        xyz = np.dot(rgb_linear, matrix.T)
        return xyz
    
    def xyz_to_rgb(self, xyz):
        """Convert XYZ to RGB color space"""
        matrix = np.array([
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252]
        ])
        
        rgb = np.dot(xyz, matrix.T)
        
        # Clip before power operation to avoid negative values
        rgb = np.clip(rgb, 0, None)
        
        mask = rgb > 0.0031308
        rgb = np.where(mask, 1.055 * np.power(rgb, 1/2.4) - 0.055, 12.92 * rgb)
        
        return np.clip(rgb, 0, 1)
    
    def rgb_to_hsv(self, rgb):
        """Convert RGB to HSV"""
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc
        
        deltac = maxc - minc
        
        # Add small epsilon to avoid division by zero
        s = np.divide(deltac, maxc, out=np.zeros_like(deltac), where=maxc!=0)
        
        rc = np.divide(maxc - r, deltac, out=np.zeros_like(deltac), where=deltac!=0)
        gc = np.divide(maxc - g, deltac, out=np.zeros_like(deltac), where=deltac!=0)
        bc = np.divide(maxc - b, deltac, out=np.zeros_like(deltac), where=deltac!=0)
        
        h = np.zeros_like(v)
        h = np.where((r == maxc), bc - gc, h)
        h = np.where((g == maxc), 2.0 + rc - bc, h)
        h = np.where((b == maxc), 4.0 + gc - rc, h)
        h = (h / 6.0) % 1.0
        
        return np.stack([h, s, v], axis=2)
    
    def hsv_to_rgb(self, hsv):
        """Convert HSV to RGB"""
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        
        i = (h * 6.0).astype(int)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        
        rgb = np.zeros((*h.shape, 3))
        
        mask = (i == 0)
        rgb[mask] = np.stack([v[mask], t[mask], p[mask]], axis=1)
        mask = (i == 1)
        rgb[mask] = np.stack([q[mask], v[mask], p[mask]], axis=1)
        mask = (i == 2)
        rgb[mask] = np.stack([p[mask], v[mask], t[mask]], axis=1)
        mask = (i == 3)
        rgb[mask] = np.stack([p[mask], q[mask], v[mask]], axis=1)
        mask = (i == 4)
        rgb[mask] = np.stack([t[mask], p[mask], v[mask]], axis=1)
        mask = (i == 5)
        rgb[mask] = np.stack([v[mask], p[mask], q[mask]], axis=1)
        
        return rgb


NODE_CLASS_MAPPINGS = {
    "Temperature Adjust": tmp
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Temperature Adjust": "Temperature Adjust"
}