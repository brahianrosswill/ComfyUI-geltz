# ComfyUI-geltz

Advanced nodes for guidance manipulation, latent/image operations, and flexible sampling control.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/geltz/ComfyUI-geltz.git
```

Restart ComfyUI after installation.

## Nodes

### Samplers

**Adaptive Refined Euler Solver (ares)**  
Deterministic sampler with momentum-aware steps and Heun integration. Auto-converts between prediction types (epsilon/x₀/v), batches sigma values, and clamps to valid ranges with Euler fallback.

**Adaptive Refined Euler Solver RDA (ares_rda)**  
Enhanced variant using Residual-Delta Acceleration. Reuses the last two UNet predictions to boost speed when model changes are minimal.

### Schedulers

**Cosine-Uniform Scheduler (csu)**  
Cosine-eased sigma schedule for smooth denoising. Maps uniform samples via `w=((1−cos(πu))/2)^γ` to timesteps, enforcing monotonic decrease with capped endpoints.

**Hybrid Cosine-Arctan Scheduler (hca)**  
Non-linear sigma schedule operating in arctan space. Interpolates between `arctan(σ_max)` and `arctan(σ_min)` using cosine weighting for distinct noise reduction curves.

### Guidance

**Attention Shuffle Guidance (asg)**  
Improves visual consistency by blending window-shuffled self-attention patterns during guided passes. Nudges output using RMS-normalized, rescaled deltas.

**Quantile Match Scaling (qms)**  
Prevents CFG oversaturation while preserving structure. Rescales guidance by matching frequency-band quantiles to the conditional distribution using EMA-smoothed FFT transformations.

**Sigma-Weighted Shuffle (sws)**  
Perturbs attention via controlled local shuffling of keys/values. Uses log-sigma progress, adaptive temperature scaling, and entropy-based blend strength with KL-bounded binary search.

**Token-Weighted Shuffle (tws)**  
Remixes nearby tokens in entropy-selected attention heads using banded top-k within a shrinking window. Includes RMS matching, KL-bounded strength tuning, and optional query mirroring.  
*Inspired by [Token Perturbation Guidance](https://github.com/TaatiTeam/Token-Perturbation-Guidance)*

**Velocity Scaling (vs)**  
Adapted from Epsilon Scaling for v-prediction models. Reduces over-brightening tendency in generated images.  
*Based on [Elucidating the Exposure Bias in Diffusion Models](https://arxiv.org/abs/2308.15321)*

### Latents

**Dithered Isotropic Latent (dil)**  
Generates structured initial latents via gradient ascent on a differentiable score (edge detection, frequency energy, kurtosis, orientation coherence). Spectral variant adds per-channel seeding and frequency-domain shaping.

### Filters

**Kuwahara Filter (kwh)**  
Fast edge-preserving filter selecting mean color from the minimum-variance quadrant.

**L₀ Gradient Minimization (lzero)**  
Global edge-aware smoothing that sparsifies image gradients to flatten regions while preserving sharp boundaries. Uses alternating hard-shrinkage on gradients with FFT-based Poisson.

**Local Laplacian Filter (llap)**  
Halo-free, multi-scale detail/tone manipulation via Laplacian pyramids. Compresses large contrasts while boosting fine details; implemented with separable Gaussian blurs and pyramid ops.

### Tokens

**tokenteller**  
Visualizes token influence to detect prompt bleed. Renders a 2D wave path with spikes proportional to each token's normalized influence (norm/variance/mean), outputting a colored curve with word-value labels.

**vectorpusher**  
Strengthens prompt adherence by nudging CLIP embeddings toward soft top-k neighbor blends using entropy-scaled trust-region optimization with KL bounds and angle constraints.  
*Inspired by [Vector Sculptor](https://github.com/Extraltodeus/Vector_Sculptor_ComfyUI)*

### Utilities

**ORBIT Merge (orbit)**  
Direction-aware model merger decomposing source–base delta into parallel/orthogonal components. Scales components independently with per-tensor trust blending. Supports UNet/CLIP/LoRA state dicts and mixed precision.
*Uses the [sd-mecha](https://github.com/ljleb/sd-mecha) api*

## License

Check the repository for license information.
