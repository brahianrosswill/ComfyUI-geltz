### Adaptive Refined Exponential Solver (ares)

Deterministic sampler for controlled denoising with momentum-aware steps. Batches sigma values, auto-converts between epsilon/x₀/v predictions, computes delta-t intervals, and applies Heun integration to advance the latent while estimating the clean image. Clamps sigma to valid ranges and falls back to Euler method when needed.

---

### Attention Shuffle Guidance (asg)

Improves visual consistency in generated images. Hooks self-attention to blend window-shuffled patterns during a guided pass, then nudges the base output using a rescaled delta clamped by RMS normalization.

---

### Cosine-Uniform Scheduler (csu)

Cosine-eased sigma schedule for smoother denoising transitions. Maps uniform samples through w=((1−cos(πu))/2)^γ to timesteps, converts to sigmas, enforces monotonic decrease, and caps endpoints at σ_max and zero.

---

### Dithered Isotropic Latent (dil)

Generates structured initial latents instead of pure noise. Optimizes noise through gradient ascent on a differentiable score combining edge detection, high-frequency energy, kurtosis, and orientation coherence. The spectral variant adds per-channel seeding and frequency-domain shaping via beta and spectral_mix parameters.

---

### Hybrid Cosine-Arctan Scheduler (hca)

Cosine-eased sigma schedule using the arctan(sigma) space for a non-linear noise reduction curve. Interpolates arctan(sigma\_max) and arctan(sigma\_min) using the w=((1−cos(pi u))/2)\^gamma weight, then converts back to sigmas, enforces monotonic decrease, and caps endpoints at sigma\_max and zero.

---

### Quantile Match Scaling (qms)

Prevents CFG oversaturation while preserving image structure. Rescales guidance (cond − uncond) by matching low, mid, and high frequency quantiles to the conditional distribution. Adapts per-band linear transformations with EMA smoothing and applies them in FFT space.

---

### Sigma-Weighted Shuffle (sws)

Perturbs attention with controlled local shuffling of keys/values. Derives normalized progress from log-sigma, scales queries/keys with adaptive temperature, and estimates attention entropy to set blend strength. Builds block-wise cyclic permutations that shrink during denoising, selecting blend weights via KL-bounded binary search. Handles dimension mismatches with orthonormal projections.

---

### tokenteller

Visualizes token influence to detect prompt bleed. Extracts token embeddings from conditioning, computes per-token values via norm/variance/mean normalized to [0,1], and renders a 2D wave path with spikes proportional to each token's influence. Outputs colored curve with viridis-like gradient and labeled word-value pairs.

---

### Token-Weighted Shuffle (tws)

Perturbs attention by remixing nearby tokens in entropy-selected heads. Mixes k/v inside a shrinking window via banded top-k, preserves scale with RMS matching, and sets strengths with KL-bounded search. Intensity and phase set token fraction, noise, and budgets, while optional query mirroring plus cached permutations and orthogonal projections keep it fast and stable.

Inspired by [Token Perturbation Guidance.](https://github.com/TaatiTeam/Token-Perturbation-Guidance)

---

### vectorpusher

Strengthens prompt adherence by refining token embeddings. Nudges each CLIP token embedding toward a soft top-k neighbor blend using entropy and attention-scaled trust-region optimization with KL bounds and angle constraints.

Inspired by [Vector Sculptor.](https://github.com/Extraltodeus/Vector_Sculptor_ComfyUI).
