### Adaptive PAG (apag)

Edit of [Perturbed Attention Guidance](https://arxiv.org/pdf/2403.17377). Perturbs only the attention values (V) by blending them with their mean: `V_perturbed = (1 - s) × V + s × mean(V)`.

Does two unet passes: one normal and one where the attention values are smoothed by mixing each V with its mean; it then adds `strength ×(clean−perturbed)` as an explicit residual guidance term, by temporarily patching torch’s `scaled_dot_product_attention` and exposing a single global strength.

Original PAG perturbs attention during the forward pass to steer outputs, but typically targets the attention maps. This variant keeps it minimal by perturbing V and turning the delta into guidance.

---

### Adaptive Refined Exponential Solver (ares)

Deterministic variation of the `res_multistep` sampler. Batches σ, auto-converts model outputs between ε/x₀/v, computes Δt, and applies a momentum aware Heun step to advance the latent and estimate x₀.

Clamps σ to [sigma_min, sigma_max], falling back to Euler when unavailable or with <2 sigmas, iterating `_ARES_STEP` across the schedule. Registers as `ares` within KSampler's sampler selection.

---

### Cosine-Uniform Scheduler (csu)

Inspired from `sgm_uniform`. Computes a cosine-eased sigma schedule: it maps uniform u∈[0,1] through w=((1−cos(πu))/2)^γ to timesteps, converts to sigmas, enforces strict decrease, caps the first at σ_max, and ends with 0.

Flushes sigmas after generating to prevent NaNs. Registers as `csu` within KSampler's scheduler selection.

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

Improves image consistency. Hooks attention, derives a progress variable u from σ or step, scales Q by a temperature τ(u), and uses local Gaussian Sinkhorn transport to stochastically shuffle K and V while keeping the attention distribution within a KL cap.

Mixing is gated by attention entropy and u, uses EMA-smoothed transport, binary-searches α for K, schedules α for V, infers H×W from sequence length, installs via `set_model_attn2_patch`, and registers the node as SWS. 

---

### tokenteller

Useful to detect "prompt bleed". Parses conditioning to gather up to `limit_streams` token embeddings, derives a per-token value by norm/var/mean normalized to [0,1], and assigns word labels from prompt-like fields or indices.

Renders a 2D wave path displaced by those values into spikes, rasterizes a colored viridis-like curve and bitmap text plus a left list of word value pairs, and outputs a single `IMAGE` tensor.

---

### vectorpusher

Improve adherence to prompts. Adds a conditioning node `vectorpusher` that tokenizes the prompt and, for each CLIP token, nudges its embedding toward a soft top-k neighbor blend using an entropy and attention-scaled trust-region step with a KL bound and angle cap.

Inspired from [Vector Sculptor by Extraltodeus.](https://github.com/Extraltodeus/Vector_Sculptor_ComfyUI)
