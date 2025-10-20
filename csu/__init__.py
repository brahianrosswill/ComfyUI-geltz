# Cosine-Uniform Scheduler

import comfy.samplers as cs
from .csu import csu_scheduler

def _csu_handler(model_sampling, steps):
    return csu_scheduler(model_sampling, int(steps))

def register_csu():
    name = "csu"
    if hasattr(cs, "SCHEDULER_HANDLERS"):
        if name not in cs.SCHEDULER_HANDLERS:
            cs.SCHEDULER_HANDLERS[name] = cs.SchedulerHandler(_csu_handler, use_ms=True)
        if hasattr(cs, "SCHEDULER_NAMES") and name not in cs.SCHEDULER_NAMES:
            cs.SCHEDULER_NAMES.append(name)
        if hasattr(cs, "KSampler") and hasattr(cs.KSampler, "SCHEDULERS") and name not in cs.KSampler.SCHEDULERS:
            cs.KSampler.SCHEDULERS.append(name)
    else:
        if hasattr(cs, "calculate_sigmas_scheduler"):
            _orig = cs.calculate_sigmas_scheduler
            def _wrapped(model_sampling, scheduler_name, steps):
                if scheduler_name == name:
                    return _csu_handler(model_sampling, steps)
                return _orig(model_sampling, scheduler_name, steps)
            cs.calculate_sigmas_scheduler = _wrapped
        elif hasattr(cs, "calculate_sigmas"):
            _orig = cs.calculate_sigmas
            def _wrapped(model_sampling, scheduler_name, steps):
                if scheduler_name == name:
                    return _csu_handler(model_sampling, steps)
                return _orig(model_sampling, scheduler_name, steps)
            cs.calculate_sigmas = _wrapped
        if hasattr(cs, "SCHEDULER_NAMES") and name not in cs.SCHEDULER_NAMES:
            cs.SCHEDULER_NAMES.append(name)
        if hasattr(cs, "KSampler") and hasattr(cs.KSampler, "SCHEDULERS") and name not in cs.KSampler.SCHEDULERS:
            cs.KSampler.SCHEDULERS.append(name)

register_csu()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
