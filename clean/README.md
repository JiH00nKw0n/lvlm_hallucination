# VLM-SAE: Multimodal Sparse Autoencoder Alignment

Reproducible suite for the paper **"Cross-Modal Feature Heterogeneity in
Multimodal SAEs"**. Single YAML per experiment, no hardcoded paths,
end-to-end runnable from one Docker image.

---

## Quick start

```bash
# Docker
docker build -f clean/Dockerfile -t vlm-sae .
docker run --rm --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e CONFIG=clean/configs/cc3m/overrides/clip_l14.yaml \
  -v $PWD/cache:/app/cache -v $PWD/outputs:/app/outputs \
  vlm-sae

# Local
pip install -e clean/
python -m clean.run clean/configs/<...>.yaml [--stage all|extract|train|perm|eval|plot]
```

| Figure / Table | Config | Wall-clock (A100) |
|---|---|---|
| Fig 1 (α-sweep) | `configs/synthetic/alpha_sweep.yaml` | ~30 min × 5 seeds |
| Fig 2 (λ-sweep) | `configs/synthetic/lambda_sweep.yaml` | ~70 min × 5 seeds |
| Fig multi-density | `configs/multi_density.yaml` | ~6 h (8 models) |
| Table CC3M (per model) | `configs/cc3m/overrides/<key>.yaml` | ~13 h × 3 seeds |

---

## 1. Synthetic α-sweep (Fig 1)

### Procedure (step-by-step)

1. **Load config** — `clean/configs/synthetic/alpha_sweep.yaml` (kind=`synthetic_sweep`).
2. **Sweep loop** — for each `(α, seed) ∈ {0, 0.2, 0.4, 0.6, 0.8, 1.0} × {1, …, 5}`:
   1. **Generate paired data.** `SyntheticPairedBuilder` (in `clean/src/data/synthetic.py`) samples $\boldsymbol\Phi, \boldsymbol\Psi$ with $\cos(\phi_i, \psi_i) = \alpha$ for $i \in [n_S]$ and $|\phi_i^\top \phi_j| \le \varepsilon_{\max}$ for $i \neq j$. Generates `num_train` train and `num_eval` eval pairs $(x, y)$.
   2. **Train per method** — for each method in `cfg.methods` (`shared`, `separated`):
      - **Build SAE** (`clean/src/training/trainer.py:_build_model`): `TopKSAE(L=8192, k=16)` for shared, `TwoSidedTopKSAE(L=8192, k=16)` for separated.
      - **Optimize** for 10 epochs with AdamW (lr=5e-4, wd=1e-5, batch=256, max_grad_norm=1.0, cosine warmup 5%); decoder column unit-norm projected after every step.
      - **Save weights** to `params/run_a<α>_s<seed>_m<method>.npz` with `w_enc_{img,txt}, b_enc_{img,txt}, w_dec_{img,txt}, b_dec_{img,txt}, phi_S, psi_S, alpha_target, seed`.
   3. **Compute metrics** — `evaluate_method` (in `src/metrics/evaluate.py`) on the eval split:
      - `CR` = collapse rate (fraction of shared atoms whose best-image and best-text columns coincide with cos > 0.95)
      - `RE` = ½(‖x − x̂‖² + ‖y − ŷ‖²)
      - `GRE` = MSE when feeding GT atom $\phi_i / \psi_i$ through the learned SAE pipeline
3. **Aggregate + plot** — `clean/src/plotting/alpha_sweep.py` reads `params/*.npz`, computes per-(α, method) mean ± std across 5 seeds, draws 3-panel figure (CR / RE / GRE), saves to `<output_root>/fig1.{pdf,svg,png}`.

### Generative model

For each sample, draw $\boldsymbol{z} \in \mathbb{R}^{n_S + n_I + n_T}$ where
the first $n_S$ coordinates are *shared* concepts (jointly active in image and
text) and the remaining $n_I + n_T$ are modality-specific:

$$
\boldsymbol{z}_i \sim
\begin{cases}
0 & \text{w.p. } s \\
\mathrm{Exp}(1) & \text{w.p. } 1-s
\end{cases}
\qquad
\boldsymbol{x} = \boldsymbol\Phi \boldsymbol{z} + \sigma_{\mathrm{obs}}\boldsymbol\eta_x,
\quad
\boldsymbol{y} = \boldsymbol\Psi \boldsymbol{z} + \sigma_{\mathrm{obs}}\boldsymbol\eta_y
$$

Image dictionary $\boldsymbol\Phi \in \mathbb{R}^{d \times (n_S + n_I)}$,
text dictionary $\boldsymbol\Psi \in \mathbb{R}^{d \times (n_S + n_T)}$.
For each shared concept $i \in [n_S]$, the column pair $(\phi_i, \psi_i)$
is sampled with controlled cosine similarity:

$$
\cos(\phi_i, \psi_i) = \alpha,
\quad
\max_{i \neq j} |\phi_i^\top \phi_j| \le \varepsilon_{\max},
\quad
\max_{i \neq j} |\psi_i^\top \psi_j| \le \varepsilon_{\max}
$$

The α-sweep walks $\alpha$ from 0 (orthogonal) to 1 (identical directions).

### Hyperparameters

| Parameter | Value |
|---|---|
| representation_dim $d$ | 256 |
| #shared atoms $n_S$ | 1024 |
| #image-only atoms $n_I$ | 512 |
| #text-only atoms $n_T$ | 512 |
| sparsity $s$ | 0.99 |
| coefficient distribution | $\mathrm{Exp}(1)$ |
| obs noise $\sigma_{\mathrm{obs}}$ | 0.05 |
| max interference $\varepsilon_{\max}$ | 0.10 |
| #train samples | 50,000 |
| #eval samples | 10,000 |
| α grid (= $\cos(\phi_i, \psi_i)$) | {0.0, 0.2, 0.4, 0.6, 0.8, 1.0} |
| seeds | 5 (1, 2, 3, 4, 5) |
| SAE width $L$ | 8,192 |
| TopK $k$ | 16 |

### Methods

| YAML name | Architecture | Parameters | Loss |
|---|---|---|---|
| `shared` | `TopKSAE` (single dictionary, $L$=8,192) | one $W_{\mathrm{enc}}, W_{\mathrm{dec}} \in \mathbb{R}^{L \times d}$ | $\frac{1}{2}(\\|\hat x - x\\|^2 + \\|\hat y - y\\|^2)$ |
| `separated` | `TwoSidedTopKSAE` (per-modality, $L/2$=4,096 each) | two disjoint dicts | per-modality recon, summed |

### Training

AdamW lr=5e-4, weight_decay=1e-5, batch_size=256, 10 epochs,
max_grad_norm=1.0, cosine warmup over 5% of total steps, decoder column
unit-norm projected after every optimizer step.

### Metrics

For shared atom $i \in [n_S]$ let $\mathrm{best}^{(I)}(i) = \arg\max_{j} |\phi_i^\top W_{\mathrm{dec},j}^{(I)}|$ and analogously $\mathrm{best}^{(T)}(i)$.

| Metric | Definition |
|---|---|
| **CR** (Collapse Rate) | $\frac{1}{n_S} \sum_{i=1}^{n_S} \mathbb{1}[\mathrm{best}^{(I)}(i) = \mathrm{best}^{(T)}(i) \,\wedge\, \cos > 0.95]$ |
| **RE** | $\frac{1}{2}\mathbb{E}\\|x - \hat x\\|^2 + \frac{1}{2}\mathbb{E}\\|y - \hat y\\|^2$ |
| **GRE** (untied) | feed GT atom $\phi_i$ (resp. $\psi_i$) through the *learned* SAE pipeline ($W_{\mathrm{enc}} \to \mathrm{ReLU} \to \text{TopK-1} \to W_{\mathrm{dec}}$); report MSE |

### Output layout

```
outputs/theorem2_alpha_sweep/
├── runs/
│   └── run_<timestamp>/
│       ├── params/
│       │   ├── run_a<α>_s<seed>_m<method>.npz   # W_enc, W_dec, b_enc, b_dec, Φ, Ψ, …
│       │   └── …
│       └── summary.json
└── fig1.{pdf,svg,png}
```

**Plot script.** `clean/src/plotting/alpha_sweep.py` — 3 panels (CR, RE, GRE)
with $\cos(\phi_i, \psi_i)$ on x-axis. figsize = (5.5, 1.015), 0.7× vertical.

---

## 2. Synthetic λ-sweep (Fig 2)

### Procedure (step-by-step)

1. **Load config** — `clean/configs/synthetic/lambda_sweep.yaml` (kind=`synthetic_sweep`).
2. **Sweep loop** — for each `(method, seed)` over 12 method entries × 5 seeds at fixed α=0.5:
   1. **Generate paired data** as in §1, with `max_interference=0.30` (harder regime).
   2. **Train SAE per method** with the appropriate auxiliary loss:
      - `shared` / `separated`: pure recon (no aux loss)
      - `iso_align`: recon $- \beta \cdot \cos(z_I, z_T)$ on dense latents (`iso_alignment_penalty` in `src/training/losses.py`)
      - `group_sparse`: recon $+ \lambda \cdot \sum_j \sqrt{z_{I,j}^2 + z_{T,j}^2}$ (`group_sparse_loss`)
   3. **Save weights** to `params/run_<method>_<aux_weight>_s<seed>.npz`.
3. **Build canonical Hungarian perm (Post-hoc Alignment)** — `compute_canonical_perm` (in `src/metrics/canonical_perm.py` and clean's `src/alignment/synthetic_perm.py`) computes a single alive-restricted Hungarian on standardized Pearson correlation. **The same perm is used by both ESim and FSim metrics** so panels are mutually consistent.
4. **Compute 4-panel metrics** on eval split:
   - `RE` = avg recon error
   - `GRE` = GT-input recon through learned SAE pipeline
   - `ESim` = mean $\cos(z_I, z_T_{\text{permuted}})$ over paired eval samples
   - `FSim` = mean $\cos(z_I, z_T_{\text{permuted}})$ over GT-atom probes
5. **Aggregate + plot** — `clean/src/plotting/lambda_sweep.py` reads `params/*.npz`, computes mean ± std across 5 seeds, draws 4-panel figure with Modality-Specific (separated) as a dotted line at `zorder=6` so it stays visible over the GS curve. Saves to `<output_root>/fig2.{pdf,svg,png}`.

### Setup

Same generative model as §1, with $\varepsilon_{\max} = 0.30$ (harder
interference) and α fixed to 0.5 (mid-difficulty heterogeneity).

### Methods (12 total)

| YAML name | Loss |
|---|---|
| `shared` | recon |
| `separated` | per-modality recon |
| `iso_align` × 5 | recon $- \beta\cos(z_I, z_T)$ for $\beta \in \{10^{-6}, 10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}\}$ |
| `group_sparse` × 5 | recon $+ \lambda \sum_{j} \sqrt{z_{I,j}^2 + z_{T,j}^2}$ for $\lambda \in \{0.0125, 0.025, 0.05, 0.10, 0.20\}$ |

### Post-hoc Alignment (Ours)

**Not a separate training run.** Computed offline from the `separated` ckpt:

```
1. Encode training pairs (x, y) → (z_I, z_T) via `separated.image_sae`, `separated.text_sae`.
2. Pearson correlation matrix C = corr(z_I, z_T) ∈ R^(L/2 × L/2).
3. Hungarian: σ* = argmax_σ Σ_i C[i, σ(i)]   (scipy.optimize.linear_sum_assignment(-C)).
4. Save perm.npz; at eval time z_T_aligned[:, i] = z_T[:, perm[i]].
```

In the λ-sweep figure, Post-hoc Alignment is plotted as a single horizontal
line — $\lambda$ has no meaning for it.

### Metrics (4-panel)

| Metric | Definition |
|---|---|
| **RE** | $\frac{1}{2}(\mathbb{E}\\|x - \hat x\\|^2 + \mathbb{E}\\|y - \hat y\\|^2)$ |
| **GRE** | GT-atom recon through learned SAE pipeline (untied) |
| **ESim** | $\mathbb{E}_{(x_n, y_n)} \cos(z_I^{(n)}, z_T^{(n)})$ on paired eval |
| **FSim** | feed GT shared atom $\phi_i$ vs $\psi_i$ as separate inputs; compute $\cos(z_I, z_T)$ over $i \in [n_S]$ |

### Plot conventions

- RE log scale; GRE log scale + $y_{\min}=0.2$; ESim/FSim linear.
- Modality-Specific (`separated`) plotted as dotted line with `zorder=6` so it stays visible above the GS curve at large λ.
- Color palette (Vibrant Tones): Shared = Willow Green `#90be6d`,
  Modality-Specific = Atomic Tangerine `#f9844a`,
  Iso-Energy = Seaweed `#43aa8b`, Group-Sparse = Blue Slate `#577590`,
  Post-hoc (Ours) = Strawberry Red `#f94144`.

### Output layout

```
outputs/theorem2_lambda_sweep/
├── runs/run_<ts>/params/run_<...>.npz
├── .esim_posthoc_cache.json    # cached Hungarian latents → ESim/FSim
└── fig2.{pdf,svg,png}
```

---

## 3. Multi-VLM density (paper §3.3, Fig multi_density)

### Procedure (step-by-step)

1. **Load config** — `clean/configs/multi_density.yaml` (kind=`multi_density`). Lists 8 model entries and a templated `cache_dir: cache/{key}_coco`.
2. **Per-model loop** — for each `model_cfg` in `cfg.models`:
   1. **Resolve cache path** — substitute `{key}` → `model_cfg.key`, e.g. `cache/clip_b32_coco`.
   2. **Extract paired COCO embeddings** (`clean/src/data/extract.py:extract_cache`):
      - Stream `yerevann/coco-karpathy` (split=`train`) → for each row, expand 5 captions to 5 paired samples.
      - Encode via `load_encoder(model_cfg)` (transformers or openclip backend, `clean/src/encoders/`).
      - Save as `image_embeddings.npy` + `text_embeddings.npy` + `keys.json` + `splits.json` + `meta.json`. Idempotent: skips if all three stacked files exist.
   3. **Train modality-specific SAE** — `train_method` with `MethodConfig(name="separated")`:
      - `TwoSidedTopKSAE(hidden_size=model_cfg.hidden_size, latent_size=8192, k=8)`.
      - 30 epochs, AdamW lr=5e-4 wd=1e-5, batch=1024, cosine warmup 5%, decoder unit-norm projected every step.
      - Save `<output_root>/<key>/final/{config.json, model.safetensors}`.
   4. **Build alive-restricted Hungarian perm** — `build_perm` (in `clean/src/alignment/hungarian.py`):
      - Stream up to 50,000 paired training embeddings through the frozen `TwoSidedTopKSAE`.
      - Compute Pearson correlation $C \in \mathbb{R}^{L \times L}$ between $\tilde z_I$ and $\tilde z_T$.
      - Mask dead rows / columns with `BIG_NEG = −1e9` so dead slots cannot steal alive matches.
      - Run `linear_sum_assignment(−C_masked)` → text→image slot perm; save `<output_root>/<key>/perm.npz` (with `perm`, `C`, `alive_image`, `alive_text`).
   5. **Density samples** — for each alive matched pair `(i, perm[i])`:
      - `decoder_cos_i = cos(image_sae.W_dec[i], text_sae.W_dec[perm[i]])`
      - `correlation_i = C[i, perm[i]]`
      - Cache the per-bin density into `cosines[model_cfg.key]`.
3. **Joint plot** — `clean/src/plotting/multi_density.py:plot_density` overlays all 8 models' decoder-cosine densities binned by correlation. Saves `<output_root>/multi_density.{pdf,svg,png}`.

### Models (4 Base + 4 Large)

| Family | Base (key, backend, id) | Large (key, backend, id) |
|---|---|---|
| OpenAI CLIP | `clip_b32` · transformers · `openai/clip-vit-base-patch32` | `clip_l14` · transformers · `openai/clip-vit-large-patch14` |
| MetaCLIP | `metaclip_b32` · transformers · `facebook/metaclip-b32-400m` | `metaclip_l14` · transformers · `facebook/metaclip-l14-400m` |
| OpenCLIP / DataComp | `datacomp_b32` · openclip · `ViT-B-32 / datacomp_xl_s13b_b90k` | `datacomp_l14` · openclip · `ViT-L-14 / datacomp_xl_s13b_b90k` |
| SigLIP2 | `siglip2_base` · transformers · `google/siglip2-base-patch16-224` | `siglip2_large` · transformers · `google/siglip2-large-patch16-256` |

### Cache

Dataset = COCO Karpathy (yerevann/coco-karpathy), split=train. Each model
produces `cache/<key>_coco/{image_embeddings.npy, text_embeddings.npy, keys.json, splits.json, meta.json}`. 5 captions per image expand to 5 paired samples per image.

### SAE training (per model)

| Parameter | Value |
|---|---|
| Architecture | `TwoSidedTopKSAE` |
| Hidden size | encoder dim (512 / 768 / 1024) |
| Latent size $L$ | 8,192 |
| TopK $k$ | 8 |
| Optimizer | AdamW (lr=5e-4, wd=1e-5, β=(0.9, 0.999)) |
| Batch | 1024 |
| Epochs | 30 |
| Warmup | cosine, 5% of total steps |
| Grad clip | max_grad_norm = 1.0 |
| Decoder norm | unit-row projected every step |
| Seed | 0 |

### Density measurement

Per model, after training:

```
1. Compute Hungarian text→image perm on training cache (50k samples):
   z_I = image_sae.encode(images), z_T = text_sae.encode(texts)
   C[i,j] = corr(z_I[:,i], z_T[:,j]),   perm = argmax_σ tr(C ∘ σ).

2. Drop dead latents:  alive_I = (z_I.std(0) > 0),  alive_T similarly.

3. For each alive matched pair (i, perm[i]):
     decoder_cos_i = cos(image_sae.W_dec[i], text_sae.W_dec[perm[i]])
     correlation_i = C[i, perm[i]]

4. Plot decoder_cos density per correlation bin; overlay all 8 models.
```

### Headline result

For correlation bin $C \ge 0.6$, all 8 VLMs concentrate decoder cosine near
$\sim 0.5$ — concept-correlated latents do **not** share the same direction
across modalities. Cross-modal feature heterogeneity is universal.

### Output layout

```
outputs/multi_density/
├── <model_key>/
│   ├── final/              # SAE checkpoint
│   └── perm.npz            # Hungarian perm + correlation matrix
└── multi_density.{pdf,svg,png}
```

---

## 4. CC3M downstream (paper Table)

### Procedure (step-by-step)

For each `(model, seed)` combo (typically 6 models × 3 seeds = 18 runs):

1. **Load config** — `clean/configs/cc3m/overrides/<model_key>.yaml` (kind=`cc3m_downstream`). It composes via `!ref ../_shared.yaml` to inherit `training`, `methods`, `eval`.
2. **Extract paired CC3M embeddings** (`clean/src/data/extract.py:extract_cache`):
   - Stream `pixparse/cc3m-wds` split=`train` (webdataset) → decode JPG, take caption text.
   - Encode via `load_encoder(model_cfg)`; for SigLIP, use `padding="max_length"` + `max_length=64` (variable padding produces wrong features).
   - Write `cache/<model_key>_cc3m/{image_embeddings.npy, text_embeddings.npy, keys.json, splits.json, meta.json}`. Idempotent.
3. **Train each method** — `train_method` per entry in `cfg.methods` (5 entries: `shared`, `separated`, `iso_align`, `group_sparse`, `ours`):
   - `_build_model` chooses `TopKSAE` for shared / iso_align / group_sparse; `TwoSidedTopKSAE` for separated and `ours` (`ours` re-uses `separated` ckpt; no separate training run).
   - 10 epochs, AdamW lr=5e-4 wd=1e-5, batch=1024, cosine warmup 5%, max_grad_norm=1.0, decoder unit-norm projected every step.
   - Loss dispatch in `_step_loss`:
     - `shared`: avg per-modality recon
     - `separated` / `ours`: per-modality recon (TwoSidedTopKSAE forward)
     - `iso_align`: recon + `aux_weight × iso_alignment_penalty(z_I, z_T)`
     - `group_sparse`: recon + `aux_weight × group_sparse_loss(z_I, z_T)`
   - Save `<output_root>/<method>/final/{config.json, model.safetensors}`.
4. **Build single Hungarian perm for `ours`** (`clean/src/alignment/hungarian.py:build_perm`):
   - Load `separated` ckpt as a `TwoSidedTopKSAE`.
   - Stream up to 50,000 CC3M training pairs through it; compute alive-restricted Hungarian on standardized Pearson correlation; mask dead rows/cols with `BIG_NEG`.
   - Save `<output_root>/ours/perm.npz`. **Used by every downstream evaluation; never recomputed per eval dataset.**
5. **Run downstream evals.** Each evaluator loads `perm.npz` for `ours`; raw decoder for other methods:
   1. **COCO retrieval** (`clean/src/eval/retrieval.py`) — Karpathy `test` (5,000 images × 5 captions). Score = $\cos(\text{decode}(z_I), \text{decode}(z_T))$ for separated/ours; $\cos(\hat x, \hat y)$ for shared. Report I→T R@1/5/10 and T→I R@1/5/10. Output `<output_root>/<variant>/coco_retrieval/summary.json`.
   2. **ImageNet zero-shot** (`clean/src/eval/zeroshot.py`):
      1. For each of 1,000 classes: load 80 cached template embeddings, L2-normalize each, mean across templates, L2-normalize prototype.
      2. Pass prototypes through text-side SAE (apply `perm` for `ours`).
      3. Pass 50,000 val image embeddings through image-side SAE.
      4. **Always-on filter**: drop latents with image-side fire rate > `max_fire_rate` (default 0.5).
      5. L2-normalize both sides on the masked subspace, cosine + argmax → top-1 accuracy.
      6. Output `<output_root>/<variant>/imagenet/zeroshot.json` with `accuracy`, `kept_latents`, `max_fire_rate`.
   3. **Cross-modal steering** (`clean/src/eval/steering.py`) — for each of 80 COCO concepts:
      1. Identify the text-side latent slot for the concept name (via `perm` for `ours`).
      2. For each of 100 base images, inject $\hat x_\alpha = x + \alpha \cdot W_{\text{dec}}^{(I)}[\text{slot}]$ for $\alpha \in \{0, 0.1, 0.25, 0.5, 1.0, 2.0\}$.
      3. Compute retrieval rank of the concept text against image feature pool, plus `preserve_mean` (cosine to original image).
      4. Output `<output_root>/<variant>/cross_modal_steering/summary.json` with per-α `{r1, r5, r10, r50, r100, map, mrr, ndcg10, preserve_mean}`.
   4. **MS (Pach 2025 §3.2 Eq 9)** (`clean/src/eval/ms.py`) — per-modality monosemanticity:
      1. Encode CC3M val pairs through the SAE → sparse latents.
      2. Per latent slot $j$, take Top-$k$ samples by activation on training-encoder side.
      3. Compute mean cosine in the **external probe space** (MetaCLIP B/32, `cache/metaclip_b32_cc3m_val`) between Top-$k$ samples vs. random pairs.
      4. $\mathrm{MS}_j = \mathbb{E}_{\text{Top-}k}[\cos] - \mathbb{E}_{\text{random}}[\cos]$.
      5. Output `<output_root>/<variant>/ms_cc3m_val/ms_summary.json` with `ms_image_mean`, `ms_text_mean`.
6. **Aggregate** — `<output_root>/table.md` (one per seed). Cross-seed mean ± std produced offline by `scripts/real_alpha/build_real_table.py`.

### Pipeline (per model, per seed)

| # | Step | Output marker | Idempotent skip |
|---|---|---|---|
| 1 | Extract paired CC3M embeddings (streaming `pixparse/cc3m-wds`) | `<cache>/image_embeddings.npy` | yes |
| 2 | Train each method's SAE | `<root>/<method>/final/config.json` | yes |
| 3 | Build single Hungarian perm on CC3M train (50k samples) | `<root>/ours/perm.npz` | yes |
| 4 | COCO retrieval / ImageNet zero-shot / steering / MS | `<root>/<eval>/summary.json` | yes |
| 5 | Aggregate `<root>/table.md` | `<root>/table.md` | yes |

### Methods (5)

| YAML name | Architecture | Loss |
|---|---|---|
| `shared` | `TopKSAE`, $L$=8,192 | recon (avg over modalities) |
| `separated` | `TwoSidedTopKSAE` ($L/2$=4,096 per side) | per-modality recon, summed |
| `iso_align` | `TopKSAE` | recon $-\beta\cos(z_I, z_T)$, $\beta=10^{-4}$ |
| `group_sparse` | `TopKSAE` | recon $+\lambda\sum_j\sqrt{z_{I,j}^2 + z_{T,j}^2}$, $\lambda=0.05$ |
| `ours` | `TwoSidedTopKSAE` (= `separated` ckpt at eval) | per-modality recon at training, perm built post-hoc |

### Training hyperparameters (`configs/cc3m/_shared.yaml`)

| Parameter | Value |
|---|---|
| TopK $k$ | 32 |
| Latent size $L$ | 8,192 |
| Optimizer | AdamW (lr=5e-4, wd=1e-5, β=(0.9, 0.999)) |
| Batch | 1024 |
| Epochs | 10 |
| Warmup | cosine, 5% of total steps |
| Grad clip | max_grad_norm = 1.0 |
| Decoder norm | unit-row projected every step |
| Seeds | 0, 1, 2 |

### Hungarian perm (the only post-hoc step)

One perm built from the CC3M training cache, reused for **every** downstream
evaluation. No per-eval-dataset perm building.

```python
# clean/src/alignment/hungarian.py
build_perm(model=separated_ckpt, cache_dir=cfg.cache.cache_dir,
           split="train", max_samples=50_000)
# saves outputs/cc3m_<key>/ours/perm.npz; loaded by every evaluator below.
```

### Evaluations

#### 4a. COCO Karpathy retrieval (test split)

| Setting | Value |
|---|---|
| Cache | `cache/<model>_coco` (extracted via `model.encode_image / encode_text`) |
| Test split | Karpathy `test` (5,000 images × 5 captions) |
| Metric | I→T R@1 / R@5 / R@10 and T→I R@1 / R@5 / R@10 |
| Method retrieval ranking | $\mathrm{score}(i, t) = \cos(W_{\mathrm{dec}}^{(I)} z_I^{(i)}, \, W_{\mathrm{dec}}^{(T)} z_T^{(t)})$ for `separated` / `ours`; for `shared` use $\cos(\hat x^{(i)}, \hat y^{(t)})$ |

#### 4b. ImageNet-1k zero-shot

| Setting | Value |
|---|---|
| Cache | `cache/<model>_imagenet` (1,000 class names → text embeddings; val split images → image embeddings) |
| Prompt template | `"a photo of a {class_name}"` |
| Metric | top-1 / top-5 accuracy on val (50,000 images) |

#### 4c. Cross-modal steering

| Setting | Value |
|---|---|
| Cache | `cache/<model>_coco` |
| Concepts | 80 COCO concept terms (`scripts/cocoid_to_concept.json`) |
| α grid | {0.0, 0.1, 0.25, 0.5, 1.0, 2.0} |
| Base images per concept | 100 |
| Steering vector | per concept $c$, the image-side decoder column $W_{\mathrm{dec}}^{(I)}[\mathrm{perm}(c)]$ |
| Metric | $\cos(\hat x_\alpha, t_c)$ where $\hat x_\alpha = x + \alpha \cdot W_{\mathrm{dec}}^{(I)}[\mathrm{perm}(c)]$ |

#### 4d. Monosemanticity Score (MS, Pach 2025 §3.2 Eq 9)

| Setting | Value |
|---|---|
| Cache (training-encoder side) | `cache/<model>_cc3m_val` |
| External similarity probe | MetaCLIP B/32 — `cache/metaclip_b32_cc3m_val` |
| Dataset / split | CC3M validation (~12k pairs) |
| Per-modality formula | $\mathrm{MS}_j = \mathbb{E}_{(x, x') \in \mathrm{Top-}k_j}[\cos(\mathrm{ext}(x), \mathrm{ext}(x'))] - \mathbb{E}_{(x, x')}[\cos(\mathrm{ext}(x), \mathrm{ext}(x'))]$ |

External encoder is decoupled from the training encoder so the score isn't
tautological.

### Output layout

```
outputs/cc3m_<model>/
├── shared/final/            # ckpt
├── separated/final/         # ckpt — also the ckpt used by `ours`
├── iso_align/final/
├── group_sparse/final/
├── ours/perm.npz            # single Hungarian perm
├── retrieval/summary.json
├── zeroshot/summary.json
├── cross_modal_steering/<variant>/summary.json
├── ms_cc3m_val/<variant>/ms_summary.json
├── config.json
└── table.md
```

### Aggregation

`table.md` formats per-method rows × per-metric columns:

```
| Method            | I→T R@1 | T→I R@1 | ImageNet zs | Steering @α=1 | MS image | MS text |
|-------------------|---------|---------|-------------|---------------|----------|---------|
| Shared            | …       | …       | …           | …             | …        | …       |
| Modality-Specific | …       | …       | …           | …             | …        | …       |
| Iso-Energy        | …       | …       | …           | …             | …        | …       |
| Group-Sparse      | …       | …       | …           | …             | …        | …       |
| **Ours**          | …       | …       | …           | …             | …        | …       |
```

---

## Adding a new VLM

1. Create `clean/configs/models/<key>.yaml`:

   ```yaml
   key: my_vlm
   backend: transformers       # or openclip
   hf_id: org/my-vlm           # transformers backend
   arch: ViT-B-32              # openclip backend
   pretrained: my_tag          # openclip backend
   hidden_size: 512
   text_max_length: 77
   is_siglip: false            # true for SigLIP family
   image_size: 224
   ```

2. Create `clean/configs/cc3m/overrides/<key>.yaml`:

   ```yaml
   kind: cc3m_downstream
   model: !ref ../../models/<key>.yaml
   cache: { cache_dir: cache/<key>_cc3m, dataset: cc3m, split: train }
   training: !ref ../_shared.yaml#training
   methods:  !ref ../_shared.yaml#methods
   eval:     !ref ../_shared.yaml#eval
   output:   { root: clean/outputs/cc3m_<key>, save_decoders: true }
   ```

3. Run: `python -m clean.run clean/configs/cc3m/overrides/<key>.yaml`

---

## Layout

```
clean/
├── README.md, pyproject.toml, Dockerfile, docker/entrypoint.sh
├── configs/
│   ├── models/              — 8 VLM definitions
│   ├── cc3m/                — _shared.yaml + 6 per-model overrides
│   ├── synthetic/           — alpha_sweep.yaml, lambda_sweep.yaml
│   └── multi_density.yaml
├── src/
│   ├── encoders/            — transformers / openclip behind one Encoder protocol
│   ├── data/                — cache_io.py + extract.py + paired_dataset.py
│   ├── models/              — TopKSAE + TwoSidedTopKSAE
│   ├── training/            — losses + trainer (5 method dispatch)
│   ├── alignment/           — hungarian.py (single training-perm)
│   ├── pipelines/           — synthetic_sweep | multi_density | cc3m_downstream
│   ├── eval/                — retrieval, steering, MS, etc.
│   ├── plotting/            — figure scripts
│   └── utils/config.py      — YAML loader (anchors + !ref)
├── tests/                   — pytest scaffold
├── run.py                   — single entrypoint, kind-dispatched
└── outputs/, cache/         — runtime artifacts
```
