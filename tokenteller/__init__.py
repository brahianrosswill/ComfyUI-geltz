import math
import typing as T
import torch
import torch.nn.functional as F


def _looks_like_tokens(t: torch.Tensor) -> bool:
    return isinstance(t, torch.Tensor) and t.dim() == 3 and t.shape[1] >= 1 and t.shape[2] >= 64


def _gather_3d(obj):
    o = []
    if isinstance(obj, torch.Tensor):
        if _looks_like_tokens(obj):
            o.append(obj)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            o += _gather_3d(v)
    elif isinstance(obj, dict):
        for v in obj.values():
            o += _gather_3d(v)
    return o


def _gather_texts(cond):
    cands = []
    if isinstance(cond, dict):
        for k in ("prompt", "text", "token_strings", "tokens", "caption", "string"):
            if k in cond:
                cands.append(cond[k])
    return cands


def _to_words(x):
    if isinstance(x, str):
        return x.split()
    if isinstance(x, (list, tuple)) and all(isinstance(v, str) for v in x):
        return list(x)
    return None


def conditioning_to_tokens_and_words(conditioning: T.List, limit_streams: int = 4):
    streams, words = [], []
    for pair in conditioning:
        if not isinstance(pair, (list, tuple)) or len(pair) == 0:
            continue
        cond = pair[0]
        cands = _gather_3d(cond) or _gather_3d([cond])
        seen = set()
        for t in cands:
            if id(t) in seen:
                continue
            seen.add(id(t))
            b, tt, c = t.shape
            streams.append(t.reshape(b * tt, c))
            if len(streams) >= limit_streams:
                break
        if len(words) == 0:
            for c in _gather_texts(cond):
                w = _to_words(c)
                if w and len(w) > 0:
                    words = w
                    break
        if len(streams) >= limit_streams:
            break
    if not streams:
        return torch.zeros(0, 768, dtype=torch.float32), []
    tok = torch.cat(streams, dim=0)
    if len(words) == 0:
        n = tok.shape[0]
        words = [f"{i}" for i in range(n)]
    return tok, words


def token_values(tokens: torch.Tensor, mode: str = "norm"):
    if tokens.dim() != 2 or tokens.numel() == 0:
        return torch.zeros(0)
    if mode == "norm":
        v = torch.linalg.vector_norm(tokens, dim=1)
    elif mode == "var":
        v = tokens.var(dim=1)
    elif mode == "mean":
        v = tokens.mean(dim=1).abs()
    else:
        v = torch.linalg.vector_norm(tokens, dim=1)
    lo = torch.quantile(v, 0.01) if v.numel() >= 10 else v.min()
    hi = torch.quantile(v, 0.99) if v.numel() >= 10 else v.max()
    return ((v - lo) / (hi - lo + 1e-6)).clamp(0, 1)


def _catrom(p0, p1, p2, p3, t):
    t2 = t * t
    t3 = t2 * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def _raster_curve(h, w, pts, thick):
    dev = float(thick) * 0.6
    yx = pts.detach().clone().to(torch.float32)
    img = torch.zeros(1, 1, h, w)
    if yx.shape[0] == 0:
        return img
    n = yx.shape[0]
    for i in range(n):
        y, x = yx[i]
        y = float(y)
        x = float(x)
        y0 = max(0, int(round(y - 3 * dev)))
        y1 = min(h - 1, int(round(y + 3 * dev)))
        x0 = max(0, int(round(x - 3 * dev)))
        x1 = min(w - 1, int(round(x + 3 * dev)))
        if y0 > y1 or x0 > x1:
            continue
        yy = torch.arange(y0, y1 + 1).view(-1, 1).float()
        xx = torch.arange(x0, x1 + 1).view(1, -1).float()
        d2 = (yy - y) ** 2 + (xx - x) ** 2
        k = torch.exp(-d2 / (2 * dev * dev))
        img[:, :, y0 : y1 + 1, x0 : x1 + 1] = torch.maximum(img[:, :, y0 : y1 + 1, x0 : x1 + 1], k.unsqueeze(0).unsqueeze(0))
    m = img.amax(dim=(2, 3), keepdim=True)
    img = img / (m + 1e-6)
    return img


_font = {
    "A": [0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001],
    "B": [0b11110, 0b10001, 0b11110, 0b10001, 0b10001, 0b10001, 0b11110],
    "C": [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110],
    "D": [0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100],
    "E": [0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b11111],
    "F": [0b11111, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000, 0b10000],
    "G": [0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110],
    "H": [0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001, 0b10001],
    "I": [0b01110, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "J": [0b00001, 0b00001, 0b00001, 0b00001, 0b10001, 0b10001, 0b01110],
    "K": [0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001],
    "L": [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
    "M": [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001],
    "N": [0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001],
    "O": [0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    "P": [0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000],
    "Q": [0b01110, 0b10001, 0b10001, 0b10001, 0b10101, 0b10010, 0b01101],
    "R": [0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001],
    "S": [0b01111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    "T": [0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    "U": [0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110],
    "V": [0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b01010, 0b00100],
    "W": [0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001],
    "X": [0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b01010, 0b10001],
    "Y": [0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100],
    "Z": [0b11111, 0b00010, 0b00100, 0b01000, 0b10000, 0b10000, 0b11111],
    "0": [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
    "1": [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
    "2": [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
    "3": [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
    "4": [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
    "5": [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
    "6": [0b01110, 0b10000, 0b11110, 0b10001, 0b10001, 0b10001, 0b01110],
    "7": [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b10000],
    "8": [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
    "9": [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00001, 0b01110],
    "-": [0b00000, 0b00000, 0b00000, 0b01110, 0b00000, 0b00000, 0b00000],
    "?": [0b01110, 0b10001, 0b00010, 0b00100, 0b00100, 0b00000, 0b00100],
    "!": [0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00000, 0b00100],
    ".": [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00100, 0b00100],
    " ": [0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b00000],
    "*": [0b00100, 0b10101, 0b01110, 0b11111, 0b01110, 0b10101, 0b00100],
}


def _draw_char(img, y, x, scale, ch, val=1):
    g = _font.get(ch, _font["*"])
    h, w = img.shape[-2], img.shape[-1]
    for r, row in enumerate(g):
        for c in range(5):
            if (row >> (4 - c)) & 1:
                yy = y + r * scale
                xx = x + c * scale
                if yy >= h or xx >= w:
                    continue
                img[..., yy : min(yy + scale, h), xx : min(xx + scale, w)] = val


def _draw_text(img, y, x, scale, s, val=1, spacing=1):
    s = s.upper()
    cx = x
    for ch in s:
        _draw_char(img, y, cx, scale, ch, val)
        cx += scale * 6 + spacing


def _layout_wave_positions(n, w, h, amp, phase):
    xs = torch.linspace(int(w * 0.08), int(w * 0.92), n)
    mid = int(h * 0.5)
    ys = []
    for i in range(n):
        t = (i / max(1, n - 1)) * math.pi * 2 + phase
        ys.append(mid + amp * math.sin(t))
    return torch.stack([torch.tensor(ys, dtype=torch.float32, device=xs.device), xs], dim=1)


def _apply_spikes(pts, vals, spike_h):
    if len(vals) == 0 or pts.shape[0] == 0:
        return pts
    v = vals.detach().float()
    v = (v - v.min()) / (v.max() - v.min() + 1e-6)
    yx = pts.clone()
    yx[:, 0] -= v * spike_h
    return yx


def _smooth_path(yx, samples_per_seg: int = 20):
    n = yx.shape[0]
    if n < 2:
        return yx
    pts = []
    for i in range(n - 1):
        p0 = yx[max(i - 1, 0)]
        p1 = yx[i]
        p2 = yx[i + 1]
        p3 = yx[min(i + 2, n - 1)]
        for k in range(samples_per_seg):
            t = k / float(samples_per_seg)
            y = _catrom(p0[0], p1[0], p2[0], p3[0], t)
            x = _catrom(p0[1], p1[1], p2[1], p3[1], t)
            pts.append(torch.stack([y, x]))
    pts.append(yx[-1])
    return torch.stack(pts, dim=0)


def _viridis_like(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0, 1)
    r = (0.18 + 0.9 * x - 0.28 * x * x).clamp(0, 1)
    g = (0.01 + 1.2 * x - 0.9 * x * x + 0.2 * x**3).clamp(0, 1)
    b = (0.5 + 0.5 * (1 - (x - 0.3).abs())).clamp(0, 1)
    return torch.stack([r, g, b], dim=-1)


def _compose_rgb_color(line_map, text_map, h, w):
    ln = line_map.clamp(0, 1).squeeze(0).squeeze(0)
    color = _viridis_like(ln)
    rgb = torch.zeros(1, h, w, 3, dtype=torch.float32)
    rgb = rgb + color.unsqueeze(0) * ln.unsqueeze(0).unsqueeze(-1)
    if text_map is not None:
        rgb = torch.maximum(rgb, text_map)
    return rgb.clamp(0, 1)


class tokenteller:
    CATEGORY = "Visualization/Conditioning"
    FUNCTION = "render"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"conditioning": ("CONDITIONING",)},
            "optional": {
                "out_res": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "limit_streams": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "amp": ("FLOAT", {"default": 80.0, "min": 0.0, "max": 400.0, "step": 1.0}),
                "phase": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 6.283, "step": 0.01}),
                "thickness": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 12.0, "step": 0.5}),
                "spike_height": ("FLOAT", {"default": 140.0, "min": 0.0, "max": 600.0, "step": 5.0}),
                "text_scale": ("INT", {"default": 2, "min": 1, "max": 6, "step": 1}),
                "spacing": ("INT", {"default": 2, "min": 0, "max": 8, "step": 1}),
                "value_mode": (["norm", "var", "mean"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = False

    def render(
        self,
        conditioning,
        out_res: int = 768,
        limit_streams: int = 4,
        amp: float = 80.0,
        phase: float = 0.0,
        thickness: float = 3.0,
        spike_height: float = 140.0,
        text_scale: int = 2,
        spacing: int = 2,
        value_mode: str = "norm",
    ):
        tokens, words = conditioning_to_tokens_and_words(conditioning, limit_streams=limit_streams)
        n_words = len(words)
        vals = token_values(tokens, mode=value_mode)
        if n_words == 0:
            img = torch.zeros(1, out_res, out_res, 3, dtype=torch.float32)
            return (img,)
        if vals.numel() < n_words:
            pad = torch.zeros(n_words - vals.numel(), device=vals.device)
            vals = torch.cat([vals, pad], dim=0)
        else:
            vals = vals[: n_words]
        h = out_res
        w = out_res
        base = _layout_wave_positions(n_words, w, h, amp, phase)
        waved = _apply_spikes(base, vals, spike_height)
        path = _smooth_path(waved, max(4, int(20)))
        line = _raster_curve(h, w, path, thickness)
        labels = torch.zeros(1, h, w, dtype=torch.float32)
        for i in range(n_words):
            wy = int(max(0, min(h - 8 * text_scale, int(float(waved[i, 0]) - int(14 * text_scale)))))
            wx = int(max(0, min(w - 1, int(float(waved[i, 1])))))
            _draw_text(labels, wy, wx, text_scale, words[i], val=1, spacing=spacing)
        listmap = torch.zeros(1, h, w, dtype=torch.float32)
        y_cursor = int(h * 0.06)
        x_cursor = int(w * 0.04)
        line_h = 8 * text_scale + spacing + 1
        for i in range(n_words):
            v = float(vals[i].item()) if i < vals.numel() else 0.0
            s = f"{words[i]} {v:.2f}"
            _draw_text(listmap, y_cursor, x_cursor, text_scale, s, val=1, spacing=spacing)
            y_cursor += line_h
            if y_cursor + 8 * text_scale >= h:
                break
        labels = torch.maximum(labels, listmap).clamp(0, 1)
        text_rgb = torch.stack([labels, labels, labels], dim=-1)
        rgb = _compose_rgb_color(line, text_rgb, h, w)
        return (rgb,)


tt = tokenteller
NODE_CLASS_MAPPINGS = {"tokenteller": tokenteller}
NODE_DISPLAY_NAME_MAPPINGS = {"tokenteller": "tokenteller"}
