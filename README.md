# ComfyUI-geltz

Advanced nodes for guidance manipulation, latent/image operations, and flexible sampling control.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/geltz/ComfyUI-geltz.git
```

Restart ComfyUI after installation.

## Nodes

### Filters

**Kuwahara Filter (kwh)**  
Fast edge-preserving filter selecting mean color from the minimum-variance quadrant.

**Local Laplacian Filter (llap)**  
Halo-free detail/tone manipulation via Laplacian pyramids with separable Gaussian blurs.

**L₀ Gradient Minimization (lzero)**  
Global edge-aware smoothing that flattens regions while preserving sharp boundaries.

**Temperature Adjust (tmp)**  
LAB-space white-balance adjustment with HSV saturation compensation, range -1.0…+1.0.

### Guidance

**Attention Shuffle Guidance (asg)**  
Improves visual consistency by blending window-shuffled self-attention patterns during guided passes.

**Quantile Match Scaling (qms)**  
Prevents CFG oversaturation by matching frequency-band quantiles to conditional distribution.

**Sigma-Weighted Shuffle (sws)**  
Perturbs attention via controlled local shuffling with adaptive temperature and entropy-based strength.

**Token-Weighted Shuffle (tws)**  
Remixes nearby tokens in entropy-selected attention heads using banded top-k within a shrinking window.  
*Inspired by [Token Perturbation Guidance](https://arxiv.org/abs/2506.10036)*

**Velocity Scaling (vs)**  
Reduces over-brightening in v-prediction models via epsilon scaling adaptation.  
*Based on [Elucidating the Exposure Bias in Diffusion Models](https://arxiv.org/abs/2308.15321)*

### Image

**Color Palette Extractor (cpe)**  
Extracts N dominant colors via MiniBatchKMeans and outputs palette image plus CSV of hex codes.

**Image Metadata Extractor (ime.info)**  
Reads PNG/TIFF info and outputs normalized prompt/settings summary as a single string.

**Load Image With Metadata (ime.load)**  
Loads image from `/input` with embedded prompts/settings extraction, returns image, mask, and metadata text.

### Latents

**Dithered Isotropic Latent (dil)**  
Generates structured initial latents via gradient ascent on differentiable edge/frequency/orientation scores.

### Loaders

**Lora Config (loracfg)**  
Parses `.safetensors` header and extracts human-readable metadata as JSON.  
*Output format compatible with [Kohya's sd-scripts](https://github.com/kohya-ss/sd-scripts)*

### Samplers

**Adaptive Refined Euler Solver (ares)**  
Deterministic sampler with momentum-aware steps, Heun integration, and auto-conversion between prediction types.

**Adaptive Refined Euler Solver RDA (ares_rda)**  
Enhanced variant using Residual-Delta Acceleration to boost speed when model changes are minimal.

### Schedulers

**Cosine-Uniform Scheduler (csu)**  
Cosine-eased sigma schedule for smooth denoising with monotonic decrease and capped endpoints.

**Hybrid Cosine-Arctan Scheduler (hca)**  
Non-linear sigma schedule interpolating in arctan space using cosine weighting.

### Tokens

**tokenteller**  
Visualizes token influence to detect prompt bleed via 2D wave path with normalized spikes.

**vectorpusher**  
Strengthens prompt adherence by nudging CLIP embeddings toward soft top-k neighbor blends.  
*Inspired by [Vector Sculptor](https://github.com/Extraltodeus/Vector_Sculptor_ComfyUI)*

### Utilities

**ORBIT Merge (orbit)**  
Direction-aware model merger decomposing deltas into parallel/orthogonal components with independent scaling.  
*Uses the [sd-mecha](https://github.com/ljleb/sd-mecha) API*
