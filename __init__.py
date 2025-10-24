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
COLOR = "\033[38;2;216;191;216m"
RESET_COLOR = "\033[0m"

def _log_loaded():
    # Prepend the color code and append the reset code to the string
    print(f"{COLOR}[geltz] loaded {len(NODE_CLASS_MAPPINGS)} nodes{RESET_COLOR}", flush=True)

_log_loaded()
print()