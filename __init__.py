import sys
import pkgutil
import importlib

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

def _merge(mod):
    d = getattr(mod, "NODE_CLASS_MAPPINGS", None)
    if isinstance(d, dict):
        NODE_CLASS_MAPPINGS.update(d)
    d = getattr(mod, "NODE_DISPLAY_NAME_MAPPINGS", None)
    if isinstance(d, dict):
        NODE_DISPLAY_NAME_MAPPINGS.update(d)

importlib.invalidate_caches()

names = [name for _, name, _ in pkgutil.walk_packages(__path__, prefix=__name__ + ".")]
for name in sorted(set(names)):
    try:
        mod = importlib.import_module(name)
        _merge(mod)
    except Exception:
        continue

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print()

# --- tiny gradient helpers ---
def _hex_to_rgb(h):
    h = h.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)

def _gradient_emit(s, start_hex, end_hex, *, bg=False):
    r1, g1, b1 = _hex_to_rgb(start_hex)
    r2, g2, b2 = _hex_to_rgb(end_hex)
    n = max(1, len(s) - 1)
    out = []
    for i, ch in enumerate(s):
        t = i / n
        r = int(round(r1 + (r2 - r1) * t))
        g = int(round(g1 + (g2 - g1) * t))
        b = int(round(b1 + (b2 - b1) * t))
        out.append(f"\x1b[{48 if bg else 38};2;{r};{g};{b}m{ch}")
    out.append("\x1b[0m")
    return "".join(out)

def _log_loaded():
    msg = f"[geltz] loaded {len(NODE_CLASS_MAPPINGS)} nodes ⧊"
    # Lavender → soft indigo
    print(_gradient_emit(msg, "#BCBCF2", "#7A77FF"), flush=True)

_log_loaded()
print()
