

### Adaptive Refined Exponential Solver (ares)

Deterministic variation of the `res_multistep` sampler. Batches σ, auto-converts model outputs between ε/x₀/v, computes Δt, and applies a momentum aware Heun step to advance the latent and estimate x₀.

Clamps σ to [sigma_min, sigma_max], falling back to Euler when unavailable or with <2 sigmas, iterating `_ARES_STEP` across the schedule. Registers as `ares` within KSampler's sampler selection.

---

### Attention Shuffle Guidance (asg)

Improves consistency of generated images. Hooks PyTorch SDPA to mix window-shuffled attention, runs a guided pass, then nudges the base output using a rescaled and RMS-clamped delta.

Exposes strength and rescale and installs via `set_model_unet_function_wrapper`.

---

### Cosine-Uniform Scheduler (csu)

Inspired from `sgm_uniform`. Computes a cosine-eased sigma schedule: it maps uniform u∈[0,1] through w=((1−cos(πu))/2)^γ to timesteps, converts to sigmas, enforces strict decrease, caps the first at σ_max, and ends with 0.

---

### Dithered Isotropic Latent (dil)

Improves empty latents. Adds two latent initializers, `DIL_EmptyLatent` and `DIL2_EmptyLatent`, that start from noise and iteratively ascend a differentiable score based on edges, high-frequency energy, kurtosis, and orientation coherence, with normalization and dithering to return a `LATENT`.

The spectral variant adds per-channel seeds and frequency-domain shaping via `beta` and `spectral_mix`, and the file includes the gradient, blur, FFT, dtype/device, and node-registration utilities.

---

### Quantile Match Scaling (qms)

Precise rescaling of CFG to prevent oversaturation. Does not affect original structure. Hooks pre-CFG and rescales the guidance g = cond − uncond by matching low, mid, and high frequency quantiles to the conditional.

Adapts cutoffs and quantiles each step, fits per-band linear maps with EMA clamps and a CFG-dependent rescale, applies them in FFT space, and returns cond_new = uncond + g_scaled.

---

### Sigma-Weighted Shuffle (sws)

(add description here)

---

### tokenteller

Useful to detect "prompt bleed". Parses conditioning to gather up to `limit_streams` token embeddings, derives a per-token value by norm/var/mean normalized to [0,1], and assigns word labels from prompt-like fields or indices.

Renders a 2D wave path displaced by those values into spikes, rasterizes a colored viridis-like curve and bitmap text plus a left list of word value pairs, and outputs a single `IMAGE` tensor.

---

### vectorpusher

Improve adherence to prompts. Adds a conditioning node `vectorpusher` that tokenizes the prompt and, for each CLIP token, nudges its embedding toward a soft top-k neighbor blend using an entropy and attention-scaled trust-region step with a KL bound and angle cap.

Inspired from [Vector Sculptor by Extraltodeus.](https://github.com/Extraltodeus/Vector_Sculptor_ComfyUI)
