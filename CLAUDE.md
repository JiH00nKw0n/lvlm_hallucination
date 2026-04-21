# CLAUDE.md — Project Context for Multimodal SAE Alignment

> Read this first when resuming work on the synthetic Theorem-2 / real-VLM $\alpha$-diagnostic project. Captures the **what** (theory), the **how** (experimental setup and server workflow), the **where we are** (current results), and the **what next** (real-data plan). Written for future Claude sessions, not for a paper draft.

> Note: this repo also contains an older codebase for VLM hallucination decoding methods (VCD / AvisC / VISTA / VTI / Deco / OPERA / Octopus / FarSight / Middle Layers) under `src/decoding/`. That is a **different project** and is not the focus here. Look in `src/decoding/CLAUDE.md` (if present) or the method source files if you need that context.

---

## 1. Project Overview

This project asks a specific structural question about Sparse Autoencoders (SAEs) trained on paired image–text embeddings: **when the ground-truth "shared" atoms of the two modalities are only partially aligned (cosine $\alpha < 1$), does a single-decoder SAE structurally collapse those atoms into a pooled column, and is there a training-time construction that avoids the collapse while preserving slot-level sharing across modalities?**

Two legs:

1. **Synthetic leg (current).** Prove Theorems 1 and 2 in a controlled generative setup, then measure how existing SAE baselines and our method behave on synthetic paired embeddings. Ground truth — $\alpha$, shared atoms, private atoms — is known, so we can verify recovery directly.
2. **Real-data leg.** Use the synthetic findings to design *indirect* diagnostics for $\alpha$ that can be applied to real VLM encoders (CLIP, SigLIP, etc.), where $\alpha$ is not observable. The outcome is an estimate $\hat{\alpha}$ for real CLIP-like spaces. **Status**: COCO Diagnostic A+B done for CLIP ViT-B/32 (§5); Multi-VLM Diagnostic B done for 5 Base models (§5.11); ImageNet-1K controlled experiment in progress (§5.12).

"Done" for the synthetic leg: Figures 1–2, the bridge-matrix diagnostic, and the three-panel CR / GRR / RE story all agree that **Ours uniquely combines slot-level sharing with modality-specific atoms**. "Done" for the real-data leg: a calibrated $\hat{\alpha}$ for at least one real VLM with an honest sensitivity analysis.

---

## 2. Theoretical Context

### 2.1 Generative Model

Paired image–text embeddings are modeled as sparse linear combinations of latent features. Let $n = n_S + n_I + n_T$ and

$$
\boldsymbol{z} = \begin{pmatrix}\boldsymbol{z}_S & \boldsymbol{z}_I & \boldsymbol{z}_T\end{pmatrix}^{\!\top} \in \mathbb{R}^n_+ ,
$$

with $\boldsymbol{z}_S$ shared across modalities and $\boldsymbol{z}_I, \boldsymbol{z}_T$ modality-private. Dictionaries are

$$
\boldsymbol{\Phi} = \begin{pmatrix}\boldsymbol{\Phi}_S & \boldsymbol{\Phi}_I & \boldsymbol{0}\end{pmatrix}, \qquad
\boldsymbol{\Psi} = \begin{pmatrix}\boldsymbol{\Psi}_S & \boldsymbol{0} & \boldsymbol{\Psi}_T\end{pmatrix},
$$

with all columns unit-norm. Observed embeddings:

$$
\boldsymbol{x} = \boldsymbol{\Phi}_S \boldsymbol{z}_S + \boldsymbol{\Phi}_I \boldsymbol{z}_I + \boldsymbol{\epsilon}_I, \qquad
\boldsymbol{y} = \boldsymbol{\Psi}_S \boldsymbol{z}_S + \boldsymbol{\Psi}_T \boldsymbol{z}_T + \boldsymbol{\epsilon}_T.
$$

The **partial alignment parameter** is the controlled quantity:

$$
\alpha_i \;:=\; \cos\!\big([\boldsymbol{\Phi}_S]_{:,i},\; [\boldsymbol{\Psi}_S]_{:,i}\big) \;\in\; [0, 1].
$$

In practice we set $\alpha_i = \alpha$ for all $i$.

### 2.2 Theorem 1 — Separate-Decoder Recovery

If an SAE is trained with **two disjoint decoders** $\boldsymbol{V}, \boldsymbol{W} \in \mathbb{R}^{d \times L}$ (i.e. Modality-Masking SAE), the per-modality reconstruction loss is minimised when each decoder column recovers the corresponding GT atom:

$$
\boldsymbol{V}^{\star}[:,k] = [\boldsymbol{\Phi}_S]_{:,k}, \qquad \boldsymbol{W}^{\star}[:,\pi(k)] = [\boldsymbol{\Psi}_S]_{:,k},
$$

for some slot permutation $\pi$. Crucially, **the angle between matched pairs survives training**: $\cos(\boldsymbol{V}^{\star}[:,k], \boldsymbol{W}^{\star}[:,\pi(k)]) = \alpha$. This is the structural foothold Diagnostic B (§5) exploits.

### 2.3 Theorem 2 — Single-Decoder Bisector Merge

If the same SAE is trained with a **single shared decoder** $\boldsymbol{D} \in \mathbb{R}^{d \times L}$ applied to both modalities, the optimal column for a shared concept $i$ under $\alpha \in (0, 1)$ is the **bisector** of the two modality-specific atoms:

$$
\boldsymbol{D}^{\star}[:,k] \;\propto\; [\boldsymbol{\Phi}_S]_{:,i} + [\boldsymbol{\Psi}_S]_{:,i}.
$$

The two GT atoms are pooled into a single average direction — the single-decoder SAE is forced to destroy the angle information. At $\alpha = 1$ the merge is lossless; at $\alpha \in (0, 1)$ it is structurally sub-optimal; at $\alpha = 0$ there is no shared component to merge.

### 2.4 Implication for Practice

- **Conventional SAE (single decoder)** pays the bisector-merge cost for every shared concept when $\alpha < 1$.
- **Modality-Masking SAE (two disjoint halves)** avoids the merge (Theorem 1) but has no slot-index correspondence between its two halves.
- **Ours** uses two separate decoders *plus* an auxiliary loss that forces the first $m_S$ slots to represent matching shared concepts at the same slot index. It should simultaneously: (i) recover per-modality atoms, (ii) produce a slot-index-aligned layout.

The synthetic experiments test these three regimes against each other.

---

## 3. Experimental Setup (Synthetic)

### 3.1 Data Generation (as of `followup12` onwards)

Exact hyperparameters. Verified against `src/datasets/synthetic_theory_feature.py`.

| Component | Symbol | Value |
|---|---|---|
| Embedding dim | $d$ | 256 |
| Shared features | $n_S$ | 1024 |
| Image-private features | $n_I$ | 512 |
| Text-private features | $n_T$ | 512 |
| Max off-pair interference | $\varepsilon_{\max}$ | 0.10 |
| Latent sparsity | $s$ | 0.99 (Bernoulli per-coord) |
| Expected active coords per modality | — | $(1 - s)(n_S + n_I) \approx 15.4$ |
| Coefficient distribution | — | $\mathrm{Exp}(1)$, truncated $\ge 0$ |
| Shared-coeff mode | — | `independent` (same mask, independent magnitudes) |
| Observation noise std | $\sigma_\mathrm{obs}$ | 0.05 |

Two pitfalls that have already bitten us:

- **$k = 16$ is the SAE encoder top-$k$, not the data sparsity.** Data is Bernoulli $p = 0.01$, giving ~15.4 active coords per modality on average.
- **Shared coefficients are sampled independently across modalities.** Only the mask is shared — $[\boldsymbol{z}_S^{\mathrm{img}}]_i$ and $[\boldsymbol{z}_S^{\mathrm{txt}}]_i$ are two independent $\mathrm{Exp}(1)$ samples. This halves the observed paired cosine: $\mathbb{E}[\cos(\boldsymbol{x}, \boldsymbol{y})] \approx 0.33\,\alpha$, not $0.67\,\alpha$.

### 3.2 Methods Compared — and *which methods actually share a decoder*

**Verified against code**, not from memory. A crucial axis for interpreting every metric:

| Method | Single SAE? | `same_model_flag` in saved params | Key loss |
|---|---|---|---|
| Conventional SAE (a.k.a. 1R, `single_recon`) | Yes | 1 | reconstruction only |
| Iso-Energy (IA, `iso_align`) | Yes | 1 | recon $+ \beta_\mathrm{IA} \cdot (-\cos(\text{masked}\,\boldsymbol{z}_I, \text{masked}\,\boldsymbol{z}_T))$ |
| Group-Sparse (GS, `group_sparse`) | Yes | 1 | recon $+ \lambda_\mathrm{GS} \sum_j \sqrt{(z_I^j)^2 + (z_T^j)^2}$ |
| Modality-Masking SAE (a.k.a. 2R, `two_recon`) | No (two disjoint decoders) | 0 | reconstruction only, per-modality |
| **Ours** | No (two decoders + aux loss) | 0 | recon $+ \lambda_\mathrm{aux} \cdot$ paired-slot alignment over first $m_S$ slots |

Conventional / Iso-Energy / Group-Sparse literally share decoder parameters ($\boldsymbol{V} = \boldsymbol{W}$). Every alignment metric interpreted for these methods must respect that: decoder-cosine diagonal is trivially 1; only the co-activation pattern can vary.

Training entry points: `_train_single_paired_aux` (single-model path: IA / GS / Ours-single-model variant) and the paired-SAE loop (two-model path: Modality-Masking, Ours) in `synthetic_theorem2_method.py`.

### 3.3 Metrics

Every metric used, with its definition, code location, and a one-line purpose.

| Metric | Definition | Code | What it tells us |
|---|---|---|---|
| **RE** (Reconstruction Error) | $\mathbb{E}[\|\boldsymbol{x} - \tilde{\boldsymbol{x}}\|_2^2 + \|\boldsymbol{y} - \tilde{\boldsymbol{y}}\|_2^2]$ | `avg_eval_loss` | standard recon quality |
| **GRR** (Ground-truth Recovery Rate at $\tau$) | fraction of GT shared atoms for which some decoder column lies within $\tau$-cone | `img_mgt_shared_tau0.95` | does the dictionary contain the GT atoms at all |
| **CR** (Collapse Rate, formerly MR) | fraction of GT shared atoms whose best image column and best text column are nearly parallel ($> 0.95$) | `merged_fraction` / offline `compute_mr_geom` | does the decoder *merge* the two modalities' directions |
| **ELSim** (Embedding-pair Latent Similarity) | $\mathbb{E}_{(\boldsymbol{x}_n,\boldsymbol{y}_n)\sim\gamma}[\cos(\boldsymbol{z}_I^{(n)}, \boldsymbol{z}_T^{(n)})]$ | `_eval_pair_latent_cosine` / `pair_cos_mean` | "for a paired eval sample, do the two sparse codes look alike" |
| **FLSim** (Feature-pair Latent Similarity) | mean over GT atoms $g$ of $\cos(\boldsymbol{z}_I^{(g)}, \boldsymbol{z}_T^{(g)})$ where inputs are $[\boldsymbol{\Phi}_S]_{:,g}$, $[\boldsymbol{\Psi}_S]_{:,g}$ | `_probe_gt_pair_activation` / `probe_vec_cos` | "for a pure GT feature probe, do the two sparse codes agree" |
| **Cross-slot correlation (top-$m_S$)** | mean of diagonal entries of per-slot Pearson corr matrix over first $m_S$ slots | `_cross_corr_mean_parts` / `cross_cos_top_mS_mean` | heavily favours Ours by construction |
| **Bridge matrix** | $\boldsymbol{B} = \mathbb{E}[\boldsymbol{z}_I \boldsymbol{z}_T^{\!\top}] \odot \tilde{\boldsymbol{V}}\tilde{\boldsymbol{W}}^{\!\top}$ | offline in `report_bridge_matrix.md` | structural diagnostic of where bridge energy lives |
| **Dead neuron fraction** | fraction of slots with zero activation on *every* eval sample | offline | capacity / collapse warning |

Notes:

- **ELSim and FLSim share the "latent cosine" shape**; the only axis between them is the *input distribution* (real paired eval data vs pure GT feature probes). They were renamed from PLC / Probe Cosine to make this explicit.
- **CR answers a different question than bridge trace.** CR is about decoder geometry under argmax; bridge trace is about what top-$k$ inference actually does. §4.3 spells out the reconciliation.

### 3.4 File Layout

- `src/datasets/synthetic_theory_feature.py` — the data generator. `SyntheticTheoryFeatureBuilder` produces $(\boldsymbol{\Phi}_S, \boldsymbol{\Psi}_S, \boldsymbol{\Phi}_I, \boldsymbol{\Psi}_T)$ and the paired sample stream. **Read this file, not memory, when asked "how is the data generated".**
- `synthetic_theorem2_method.py` — main experiment driver. Trains all five methods, evaluates metrics, saves decoder / encoder weights when `--save-decoders` is passed.
- `scripts/plot_fig1_three_panels.py` — Figure 1 (masking vs no masking across $\alpha$).
- `scripts/plot_fig2_lambda_sweep.py` — Figure 2 ($\lambda$ sweep at $\alpha = 0.5$).
- `scripts/plot_alignment_metric_explainer.py` — ELSim / FLSim explainer.
- `scripts/plot_metric_explainer.py` — CR / GRR / RE explainer.
- `scripts/plot_sae_architecture_explainer.py` — single SAE vs modality-masking schematic.
- `scripts/plot_diagnostic_B_intuition.py` — Diagnostic B intuition (three panels, Korean text).
- `outputs/theorem2_followup_{N}/runs/*/params/alpha*_seed*_*.npz` — saved parameters. Schema: `(w_enc_img, b_enc_img, w_enc_txt, b_enc_txt, w_dec_img, b_dec_img, w_dec_txt, b_dec_txt, phi_S, psi_S, phi_I, psi_T, alpha_target, seed, latent_size_img, latent_size_txt, same_model_flag)`. Use these for **all** offline metric recomputation — do not retrain.
- `outputs/theorem2_followup_{N}/runs/*/result.json` — per-run training-time metrics. `sweep_results[0]["aggregate"]` is the metric dict.

### 3.5 Run Directory Index

When in doubt, look here before creating a new directory.

| Dir | Purpose | $\alpha$ | $\sigma$ | Methods | Notes |
|---|---|---|---|---|---|
| `theorem2_followup_8` | Legacy basic setup | 0.5, 0.7, 0.9 | 0.1 | all | superseded; do not use |
| `theorem2_followup_12` | Main synthetic run — low noise | 0.2–0.8 | 0.05 | all six | source of Figure 1 |
| `theorem2_followup_15` | $\lambda$ sweep at fixed $\alpha$ | 0.5 | 0.05 | 1R/2R baselines + IA/GS/Ours at $\times\{1/16, 1/4, 1, 4, 16\}$ | source of Figure 2, bridge report |
| `theorem2_followup_16` | Wide $\alpha$ sweep | 0.1–1.0 | 0.05 | 1R/2R only | Figure 1 wide version |

Starting a new sweep: create `theorem2_followup_{N+1}` and write one line in this table explaining its purpose. Never overwrite an existing directory.

**Real-data runs** (CLIP ViT-B/32 on COCO Karpathy, Diagnostic A + B):

| Dir | $L$ (one / two per-side) | one final | two final | $\Delta_\mathrm{loss}$ | Notes |
|---|---|---:|---:|---:|---|
| `real_alpha_followup_1` | 8192 / 4096 | 0.2820 | 0.2811 | +0.0009 | primary run; Diagnostic B boxplot + report |
| `real_alpha_followup_2` | 4096 / 2048 | 0.2847 | 0.2921 | −0.0074 | two worse (capacity-limited) |
| `real_alpha_followup_3` | 16384 / 8192 | 0.2704 | 0.2650 | +0.0054 | largest gap; confirms L-capacity effect |

Common settings: `openai/clip-vit-base-patch32`, $k=8$, 30 epochs, batch 1024, AdamW cosine+warmup 5%, lr 5e-4, wd 1e-5, grad clip 1.0, `normalize_decoder=True`, loss $= (\mathcal{L}_I + \mathcal{L}_T)/2$, seed 0, eval on Karpathy test (25k pairs).

Embedding cache: `cache/clip_b32_coco/` (1.6 GB total: 123k images + 617k texts, post-projection ℓ2-unnormalized, normalized at dataset load time via CLIP-native `_get_vector_norm`).

---

## 4. Current Results

### 4.1 Figure 1 — Masking vs No Masking Across $\alpha$

Runs: `followup12` (narrow $\alpha$, six methods) and `followup16` (wide $\alpha$ 0.1–1.0, only 1R/2R). Script: `scripts/plot_fig1_three_panels.py`. Panels: (a) RE (log scale), (b) GRR at $\tau = 0.95$, (c) CR.

- **CR is the cleanest signal.** Conventional SAE's CR grows monotonically in $\alpha$, hitting $\approx 1$ at $\alpha = 1$. Modality-Masking SAE stays at $\approx 0$ for every $\alpha$. Direct empirical shadow of Theorem 2.
- **GRR crosses** at high $\alpha$: single SAE overtakes masking near $\alpha = 1$ (shared column becomes genuinely optimal when atoms coincide). At $\alpha \lesssim 0.7$ masking wins.
- **RE gap is small but monotone** — masking reconstructs slightly better across the whole range, matching Diagnostic A's pre-registered prediction.

### 4.2 Figure 2 — $\lambda$ Sweep at $\alpha = 0.5$

Run: `followup15`. Script: `scripts/plot_fig2_lambda_sweep.py`. IA / GS / Ours swept at $\times\{1/16, 1/4, 1, 4, 16\}$ of their paper-default. Conventional SAE as a dashed reference.

- **Ours matches Modality-Masking on RE and CR at every $\lambda$** while keeping GRR slightly above it. Concrete "best-of-both" claim.
- **Iso-Energy is essentially inert.** Five orders of magnitude of $\beta_\mathrm{IA}$ barely move any metric.
- **Group-Sparse has a sharp cliff between $\lambda \times 1$ and $\lambda \times 4$** — RE jumps ~5×, CR drops to zero. This is the collapse regime just above the paper default.

### 4.3 Bridge Matrix Diagnostic

Offline: `outputs/theorem2_followup_15/report_bridge_matrix.md`. Protocol: regenerate paired eval from saved dictionaries, forward through saved SAE encoders, assemble $\boldsymbol{B} = \boldsymbol{C} \odot \boldsymbol{A}$.

- **Diagonal-concentration ratio** $\rho_\mathrm{diag} = \mathrm{tr}(\boldsymbol{B}) / \sum_{ij}\boldsymbol{B}_{ij}$: Ours $\approx 0.50$, single-SAE baselines $\approx 0.35$, Modality-Masking $\approx 0.001$. Only Ours concentrates bridge energy on matched slot indices.
- **Does not contradict CR.** CR measures whether the decoder destroys per-modality directions (lower is better, Ours = 0). Bridge diagonal measures whether matched slots co-fire AND are aligned (higher is better, Ours = 0.5). Ours wins both because its two decoders preserve the per-modality angle $\alpha \approx 0.5$ *and* force matching slot indices — a combination no single-SAE baseline can structurally achieve.
- **Iso-Energy at $\lambda \times 16$** crosses Ours on raw bridge trace by inflating activation magnitudes rather than rearranging slot structure — sum, trace, Frobenius all scale together, so $\rho_\mathrm{diag}$ stays below Ours. Known pitfall of cosine-based penalties.

### 4.4 Supplementary Tables (at $\alpha = 0.5$, across $\lambda$)

Headlines only; full tables in the bridge report.

- **ELSim**: Ours $\approx 0.19$ across all $\lambda$, every baseline $\le 0.13$, Modality-Masking $\approx 0.003$.
- **FLSim**: Ours $\approx 0.86$–$0.90$, Conventional $\approx 0.43$, Modality-Masking $\approx 0$.
- **Dead neuron fraction**: Ours and Modality-Masking $\sim 3$–$7\%$; Conventional $\sim 11\%$; Group-Sparse $> 50\%$ in the collapse regime.

---

## 5. Real-Data Diagnostic — Plan, Execution, First Results

Original plan: `outputs/theorem2_followup_15/plan_real_alpha_diagnostic.md`.
Notion page: `https://www.notion.so/Real-Data-343abf49604280ab8771e7f26fb38f04`.
Detailed analysis report (local): `outputs/real_alpha_followup_1/diagnostic_B_report.md`.

### 5.1 Research Question

Estimate $\hat{\alpha}$ for a real paired VLM (CLIP / SigLIP / BLIP) without access to ground-truth atoms. Primary hypothesis $H$: real CLIP sits at $\hat{\alpha} \in (\alpha_\mathrm{min}, 1)$ with $\alpha_\mathrm{min}$ bounded away from 0.

### 5.2 Two Diagnostics (definitions unchanged)

**Diagnostic A — Reconstruction-loss sign test.** $\Delta_\mathrm{loss} := \mathcal{L}_\mathrm{rec}^\mathrm{one} - \mathcal{L}_\mathrm{rec}^\mathrm{two}$. Positive means two-SAE (modality masking) reconstructs better.

**Diagnostic B — Matched decoder cosine.** Train two-SAE; compute Pearson correlation matrix $C \in [-1,1]^{L/2 \times L/2}$ between paired dense latents on the train set; Hungarian-match on $-C$ (signed, maximize positive correlation); for each matched pair $(i, \pi(i))$ compute $\rho_k = \cos(\boldsymbol{V}_I[:,i],\, \boldsymbol{V}_T[:,\pi(i)])$.

### 5.3 Execution Status

**Done (2026-04-15/16):**

- CLIP ViT-B/32 embedding cache extracted (COCO Karpathy, 123k images + 617k texts, `cache/clip_b32_coco/`, 1.6 GB).
- Three $L$-sweep runs completed: `real_alpha_followup_{1,2,3}` ($L \in \{4096, 8192, 16384\}$).
- Diagnostic A and B computed for all three.
- Results uploaded to Notion `Real-` page.

### 5.4 First Results — Diagnostic A ($\Delta_\mathrm{loss}$ per $L$)

| $L$ | one SAE | two SAE | $\Delta_\mathrm{loss}$ |
|---:|---:|---:|---:|
| 4096 | 0.2847 | 0.2921 | −0.0074 |
| 8192 | 0.2820 | 0.2811 | +0.0009 |
| 16384 | 0.2704 | 0.2650 | +0.0054 |

$\Delta$ flips sign between $L=4096$ (capacity-limited, two-SAE loses due to per-side halving) and $L \ge 8192$ (Theorem 2 bisector cost visible). Gap widens with $L$: +0.0009 → +0.0054.

### 5.5 First Results — Diagnostic B (decoder cosine by $C$ bin)

**Key observations (all from `real_alpha_followup_1`, $L=8192$, Hungarian-alive 558 matched pairs):**

- **85% of latents are dead** (fire rate = 0 on train set). Alive: 558 image / 569 text out of 4096 each.
- Hungarian restricted to alive-alive submatrix → 558 matched pairs.
- **Pearson $r(C, \rho) = 0.685$** ($p \approx 10^{-78}$): decoder cosine increases monotonically with co-activation correlation.
- Binned by $C$ (0.2 width):

| $C$ bin | $n$ | $\rho$ median | $\rho$ mean |
|---|---:|---:|---:|
| [0.0, 0.2) | 289 | 0.126 | 0.157 |
| [0.2, 0.4) | 132 | 0.345 | 0.334 |
| [0.4, 0.6) | 70 | 0.431 | 0.427 |
| [0.6, 0.8) | 45 | 0.528 | 0.506 |
| [0.8, 1.0) | 22 | 0.488 | 0.498 |

- Strongly co-firing pairs ($C \ge 0.6$) show median decoder cosine ~0.5.
- Without calibration, this number cannot be directly equated to $\hat\alpha$; it is the raw observable.

### 5.5b Single vs Two SAE Comparison + Merge (Collapse) Evidence

**Per-sample top-k overlap (direct merge measurement):**

For each paired sample $(x, y)$, count $|\mathrm{topk}(z_I(x)) \cap \mathrm{topk}(z_T(y))|$ — same latent fires for both modalities in single SAE.

| $L$ | samples with overlap > 0 | mean overlap / $k$ | random chance | ratio |
|---:|---:|---:|---:|---:|
| 4096 | 6.44% | 0.067 | 0.016 | 4.3× |
| 8192 | 4.36% | 0.045 | 0.008 | 5.7× |
| 16384 | 3.46% | 0.036 | 0.004 | 9.3× |

Merge happens at all $L$ (well above chance) but is rare in absolute terms (93–97% have zero overlap).

**High-cos pairs by architecture ($\cos \ge \tau$, Hungarian-alive):**

| $\cos \ge$ | Two (8192) | Single (8192) | self/cross | Two (4096) | Single (4096) | self/cross |
|---:|---:|---:|---|---:|---:|---|
| 0.5 | 95 | 64 | 16/48 | 86 | 68 | 24/44 |
| 0.7 | 4 | **17** | **16**/1 | 2 | **25** | **24**/1 |
| 0.8 | 0 | 16 | 16/0 | 0 | 24 | 24/0 |

$\cos \ge 0.7$ in single SAE is almost entirely **self-matched (merged) latents**. Two SAE has almost none in this range — structurally unable to merge.

**Collapse rate (single SAE, $L = 4096$):**

- Hungarian-alive matched pairs with $C \ge 0.2$: **236**
- Self-matched (cos = 1, collapsed) within those: **16**
- **Collapse rate = 16/236 = 6.8%**

The 16 collapsed pairs in single SAE were cross-matched to two SAE pairs via sample-overlap (Jaccard on top-50 activating samples). Of 13 successfully matched, their **two SAE cos: mean 0.546, median 0.589**.

Interpretation: concepts that single SAE merges into a bisector ($\cos = 1$) correspond to two SAE pairs with $\cos \approx 0.55$ — i.e., these are shared concepts with partial alignment $\alpha \approx 0.55$.

**Comparison to synthetic**: at synthetic $\alpha = 0.5$ the CR (Collapse Rate) in the single-SAE baseline is $\approx 10\%$ (followup12 data). The real CLIP collapse rate of 6.8% at $L = 4096$ is in the same ballpark, consistent with the shared concepts having $\alpha$ in the 0.4–0.6 range.

**Modality ratio (single SAE $L = 8192$)**:

Per-latent firing ratio $r_k = \mathrm{fire}_I(k) / (\mathrm{fire}_I(k) + \mathrm{fire}_T(k))$:

| Category | $n$ | % of alive |
|---|---:|---:|
| Image-dominant ($r > 0.9$) | 618 | 45.7% |
| Text-dominant ($r < 0.1$) | 632 | 46.7% |
| Shared ($0.2 \le r \le 0.8$) | 57 | 4.2% |
| Strong shared ($0.4 \le r \le 0.6$) | 18 | 1.3% |

The 18 strong-shared latents show bisector-like decoder placement: median $|\cos(W_k, \bar x_I) - \cos(W_k, \bar x_T)| = 0.084$ (0 = perfect bisector). Their concept-level alignment: $\cos(\bar x_I, \bar x_T) \approx 0.37$.

### 5.6 Bisector Verification (Theorem 2 in Real Data)

For each collapsed single SAE pair, we matched it to the corresponding two SAE pair (via Jaccard on top-50 activating samples — see `collapsed_pairs_comparison.json`), then computed decoder cosines. Of 14 matched pairs with `two_pair != null`:

- **cos(single, bisector(img+txt))**: 4 YES (>0.95), 7 near (0.8–0.95), 3 NO (<0.8, low Jaccard)
- **11/14 confirm Theorem 2**: single SAE decoder lies near the bisector of two SAE's image and text decoder columns
- **Asymmetric merge**: cos(s, img) > cos(s, txt) in 10/14 cases — single decoder leans toward image direction
- The 3 NO cases have Jaccard < 0.13 (bad concept matching) or anomalous structure

### 5.7 Iso-Energy Analysis

Verified the Iso-Energy Assumption ($E_k^{img} := \mathbb{E}[(z_k^{img})^2] = E_k^{txt} := \mathbb{E}[(z_k^{txt})^2]$) using cached CLIP embeddings + trained SAEs at $L = 4096$.

**Single SAE (all 1127 alive latents):**

| Category | Count | % |
|---|---:|---:|
| Image-dominant ($\mu > 0.8$) | 558 | 49.5% |
| Text-dominant ($\mu < 0.2$) | 526 | 46.7% |
| Bimodal ($0.2 \le \mu \le 0.8$) | 43 | 3.8% |

Where $\mu_k = E_k^{img} / (E_k^{img} + E_k^{txt})$. Iso-energy **does NOT hold** globally — most latents are strongly unimodal.

**Collapsed 24 latents**: $\mu$ mean = 0.41, median = 0.40 — concentrated near the iso-energy diagonal. These are the bimodal shared concepts.

**Two SAE (Hungarian-matched C ≥ 0.2, 212 pairs):**

| Stat | Value |
|---|---|
| Bimodal (0.2 ≤ μ ≤ 0.8) | 182/212 (86%) |
| log₁₀(E_img/E_txt) median | 0.005 |
| log₁₀ ratio std | 0.43 |

Two SAE's matched pairs sit tightly on the iso-energy diagonal — shared concepts have balanced energy across modalities. Single SAE concentrates this into a few collapsed latents.

Scripts: `scripts/real_alpha/iso_energy_check.py` (single), `scripts/real_alpha/iso_energy_two.py` (two). Outputs: `outputs/real_alpha_followup_2/iso_energy.npz`, `iso_energy_two.npz`, `fig_iso_energy_comparison.png/pdf`.

### 5.8 Per-Atom Reconstruction Error (Theorem 2 Cost)

Uses Two SAE decoder columns as GT atom proxies. For each collapsed concept, feeds the atom through Single SAE and Two SAE, measures $\|\text{atom} - \text{SAE}(\text{atom})\|^2$.

**Collapsed 14 concepts (same atom, single vs two):**

| | Two SAE | Single SAE | Δ (single−two) | % increase |
|---|---:|---:|---:|---:|
| Image atoms | 0.428 | 0.630 | +0.203 | +47% |
| Text atoms | 0.335 | 0.616 | +0.281 | +84% |

- Δ_img positive in 12/14, Δ_txt positive in 13/14
- Text Δ larger because bisector leans toward image (consistent with asymmetric merge in §5.6)

**Note**: this measurement uses the actual SAE forward pass (TopK, untied weights, bias) — not exactly the paper's $\|phi_i - V\sigma(V^\top \phi_i)\|^2$ (ReLU, tied weights, no bias) — but the spirit is the same: per-atom reconstruction cost.

Script: `scripts/real_alpha/per_atom_recon_error.py`. Output: `outputs/real_alpha_followup_2/per_atom_recon.json`.

### 5.9 Monosemanticity (MS) Experiment

Tested whether collapsed latents have lower monosemanticity scores. Used CLIP embeddings (same-space baseline).

**Aggregate result** (misleading):

| Group | n | MS_img | MS_txt |
|---|---:|---:|---:|
| Single collapsed | 16 | 0.598 | 0.604 |
| Single non-collapsed | 220 | 0.661 | 0.662 |
| Two SAE | 212 | 0.649 | 0.654 |

**Paired comparison** (same concept, 11 matched pairs): collapsed single ≈ corresponding two SAE (Δ ≈ 0, not significant). The aggregate difference is a **selection effect** — concepts that get collapsed happen to have lower MS, not because of collapse itself.

MS measures clustering of top-activating samples' embeddings. Collapse affects decoder direction (bisector vs true), not which samples a latent responds to. MS is not the right metric for collapse cost — use per-atom reconstruction error (§5.8) instead.

Script: `scripts/real_alpha/monosemanticity_experiment.py` (`--emb-type clip` for fast, `--emb-type dino` for independent DINOv2-base/E5-base). Output: `outputs/real_alpha_followup_2/monosemanticity_clip.json`.

### 5.10 New Code for Real Experiments

| File | Role |
|---|---|
| `scripts/real_alpha/_bootstrap.py` | Import bootstrap — bypasses broken `src.*.__init__` on server |
| `scripts/real_alpha/extract_clip_coco_cache.py` | Idempotent CLIP embedding extraction → `cache/clip_b32_coco/` |
| `scripts/real_alpha/train_real_sae.py` | Entry point: `--variant {one_sae, two_sae}`, HF Trainer |
| `scripts/real_alpha/run_diagnostic_B.py` | Hungarian + decoder cosine + cofiring threshold sweep |
| `scripts/real_alpha/plot_diagnostic_A_pub.py` | Publication-quality 2-panel (log-scale loss + delta) |
| `scripts/real_alpha/plot_diagnostic_B_violin.py` | Publication-quality boxplot (C bins vs decoder cosine) |
| `scripts/real_alpha/render_collapse_v2.py` | Collapsed pair comparison HTML (shared/single-only/two-only grids) |
| `scripts/real_alpha/cache_single_sae_C.py` | Single SAE correlation matrix cache |
| `scripts/real_alpha/inspect_top_activations.py` | Top activating sample HTML (`--single` for single SAE) |
| `scripts/real_alpha/iso_energy_check.py` | Iso-energy $E[z_k^2]$ check for single SAE |
| `scripts/real_alpha/iso_energy_two.py` | Iso-energy check for two SAE (Hungarian-matched) |
| `scripts/real_alpha/monosemanticity_experiment.py` | MS experiment (`--emb-type clip\|dino`) |
| `scripts/real_alpha/per_atom_recon_error.py` | Per-atom recon error: single vs two SAE |
| `src/datasets/cached_clip_pairs.py` | `CachedClipPairsDataset` — loads cached embeddings, ℓ2-normalizes |
| `src/models/configuration_sae.py` | `TwoSidedTopKSAEConfig` added |
| `src/models/modeling_sae.py` | `TwoSidedTopKSAE(PreTrainedModel)` added |
| `src/runners/trainer.py` | `OneSidedSAETrainer`, `TwoSidedSAETrainer` added |
| `scripts/real_alpha/extract_imagenet_cache.py` | ImageNet-1K embedding extraction (HF streaming) → `cache/clip_b32_imagenet/` |
| `src/datasets/cached_imagenet_pairs.py` | `CachedImageNetPairsDataset` — class-based pairing, random template, `max_per_class` balanced subsample |
| `scripts/real_alpha/plot_multi_model_boxplot.py` | Multi-VLM Diagnostic B boxplot (5 Base models) |
| `scripts/real_alpha/run_multi_model_pipeline.sh` | Multi-model pipeline: extract → train two-SAE → Diagnostic B |
| `experiment.sh` | Reproducible 10-model experiment (5 Base + 5 Large), Docker-compatible |
| `Dockerfile.experiment` | Docker image for `experiment.sh` |

### 5.11 Multi-VLM Diagnostic B (5 Base Models)

Ran Diagnostic B on 5 VLMs (all ViT-B/32 or Base scale) using COCO Karpathy embeddings. Each model: extract → train two-SAE ($L=8192$) → Hungarian-alive matching → decoder cosine by $C$ bin.

Models: CLIP ViT-B/32, MetaCLIP B/32, DataComp B/32, MobileCLIP2-B, SigLIP2 Base.

All 5 models show consistent $C \uparrow \Rightarrow \rho \uparrow$ trend. Even for strongly co-firing pairs ($C \ge 0.6$), decoder cosine concentrates around $\sim 0.5$ across all models — indirect evidence that shared features in real VLMs are only partially aligned.

Embedding caches on server: `cache/{clip_b32,metaclip_b32,datacomp_b32,mobileclip2b,siglip2_base}_coco/`.
Boxplot: `outputs/multi_model_boxplot.pdf`. Script: `scripts/real_alpha/plot_multi_model_boxplot.py`.

### 5.12 ImageNet-1K Controlled Experiment (in progress)

Uses ImageNet-1K as a more structured test case than COCO: class-label pairing (not free-form captions), 80 OpenAI prompt templates per class.

**Embedding cache completed** (2026-04-19):

| Component | Count | Key scheme |
|---|---:|---|
| Train images | 1,281,167 | `0` ~ `1281166` |
| Val images | 50,000 | `2000000` ~ `2049999` (offset) |
| Text templates | 80,000 (1000 classes × 80) | `"{class_idx}_{template_idx}"` |

Cache: `cache/clip_b32_imagenet/` (~3.1 GB). Model: `openai/clip-vit-base-patch32`, dim=512.
Class names and templates from `open_clip.zero_shot_metadata`.

**Dataset class**: `CachedImageNetPairsDataset` picks a random template per `__getitem__` call. `max_per_class` parameter enables balanced subsampling (e.g., 1000/class for 1M total).

**Next steps**: train two-SAE on ImageNet, run Diagnostic B, compare with COCO results.

### 5.7 Scope and Limits

- **Linear atom assumption** — shared with prior SAE-for-VLM work but not proven.
- **SAE optimisation reaches local optima**; mitigate with seed sweeps.
- **Dead latents 85%**: likely a hyperparameter issue (top-k=8, no AuxK). Alive-restricted analysis compensates but doesn't fix the underlying waste.
- **L-capacity effect**: at small $L$ (4096), two-SAE is worse due to per-side halving. The $\Delta > 0$ signal appears only at $L \ge 8192$. Real conclusions require $L$ large enough.
- **Hungarian matching is a proxy** for the GT permutation in Theorem 1. Matched pairs are the best approximation, not ground truth.
- **Single seed, single encoder (CLIP ViT-B/32), no calibration.** Cross-check with multiple seeds, VLMs, and a synthetic lookup table needed before quantitative $\hat\alpha$ claims.

---

## 6. Server Workflow (must survive across sessions)

### 6.1 Machine

- SSH alias: `elice-40g` (also `elice-10g` available). Config in `~/.ssh/config`. Hostname is an Elice tunnel; the alias handles port/identity for you.
- Working directory on server: `/mnt/working/lvlm_hallucination`. This is a **separate checkout** from the local repo — edits need to be copied over explicitly.
- GPU: 10 GB class (despite the alias name `elice-40g`). Batch 1024 fits for $L \le 16384$ without gradient accumulation. `nvidia-smi` requires `--no-permissions` or returns "Insufficient Permissions"; `torch.cuda` works fine.
- Python env: `.venv/` in `/mnt/working/lvlm_hallucination`, activated by `source .venv/bin/activate`. **transformers 5.5.3** — `CLIPModel.get_image_features` returns `BaseModelOutputWithPooling`, not a Tensor. Use `model.vision_model(...)` + `model.visual_projection(...)` explicitly.
- Embedding caches on server (`cache/`): `clip_b32_coco/` (1.7G), `metaclip_b32_coco/` (1.7G), `datacomp_b32_coco/` (1.7G), `mobileclip2b_coco/` (1.7G), `siglip2_base_coco/` (2.4G), `clip_b32_imagenet/` (3.1G). Total ~12.3G. rsync individual dirs when doing offline analysis locally.
- Server disk: 128G total, ~86G used (as of 2026-04-20). Major consumers: HF model cache (~8.5G in `~/.cache/huggingface/hub/`), HF dataset cache (~19G in `~/.cache/huggingface/datasets/` — COCO Karpathy raw images), `.venv/` (7.7G). Run `pip cache purge` periodically.

### 6.2 Canonical Experiment Loop (proven, use this)

All synthetic sweeps follow the same pattern. Do not deviate unless you have a reason and document it in the run-directory index.

1. **Write a shell script locally.** Template: `run_followup{N}_{description}.sh`. Must `cd /mnt/working/lvlm_hallucination`, `source .venv/bin/activate`, `mkdir -p .log outputs/theorem2_followup_{N}`, log to `.log/followup{N}_sweep.log`, and pass `--save-decoders --device cuda --output-root outputs/theorem2_followup_{N}` to `synthetic_theorem2_method.py`. Look at `run_followup12_lownoise.sh` or `run_followup15_4x.sh` as templates.
2. **Copy to server**: `scp run_followup{N}_{description}.sh elice-40g:/mnt/working/lvlm_hallucination/`
3. **Start in background** (non-blocking — important, single runs take 10–30 min):
   ```
   ssh elice-40g "cd /mnt/working/lvlm_hallucination && chmod +x run_followup{N}_{description}.sh && nohup bash run_followup{N}_{description}.sh > /dev/null 2>&1 & echo started pid=\$!"
   ```
4. **Verify it started**: `ssh elice-40g "tail -3 /mnt/working/lvlm_hallucination/.log/followup{N}_sweep.log; pgrep -af synthetic_theorem2"` — should show the progress bar and a running process.
5. **Schedule a wake-up** (`ScheduleWakeup` tool) for the expected completion time plus a small buffer. Do not poll.
6. **On completion, rsync results back**:
   ```
   rsync -av elice-40g:/mnt/working/lvlm_hallucination/outputs/theorem2_followup_{N}/ outputs/theorem2_followup_{N}/
   ```
   This pulls both `runs/*/result.json` and `runs/*/params/*.npz`.
7. **Plot locally** using the script(s) in `scripts/` — do not re-render on the server. Figures live under `outputs/theorem2_followup_{N}/`.

### 6.3 Progress Estimation

Per-method-per-$\alpha$ training takes roughly 3 minutes on the A100 at the current hyperparameters ($L = 8192$, 10 epochs, batch 256, 50k train). Use this to pick the `ScheduleWakeup` delay. Sanity check the log file's `sweep:  X/10 [...]` progress bar if unsure.

### 6.4 Sharp Edges

- **Two separate repos.** The local repo under `~/Desktop/Projects/lvlm_hallucination` and the server working copy at `/mnt/working/lvlm_hallucination` are distinct. Changes on one side do not propagate. When you edit `synthetic_theorem2_method.py` or a sweep script locally, either `scp` the file(s) across or `rsync`.
- **Never `scp` the whole repo.** It contains large `params/` dumps. Use targeted `scp` for scripts and `rsync` for `outputs/theorem2_followup_{N}/` only.
- **Decoder/encoder save is opt-in.** Without `--save-decoders`, the `.npz` files are not written, and every offline metric recomputation is impossible. Always include it for real experiments.
- **The running process does not die if the ssh session ends** — that is why we use `nohup ... > /dev/null 2>&1 &`. But it also means you can accidentally leave zombie sweeps running. `ssh elice-40g "pgrep -af synthetic_theorem2"` to check.

---

## 7. Open Threads / TODO

### 7.1 Experiments Not Yet Run

- ~~Real CLIP paired-encoder sweep~~ — **Done** (followup 1/2/3). See §5.
- **Synthetic calibration for Diagnostic A + B.** Apply the same Hungarian-alive protocol to synthetic $\alpha \in \{0.2, 0.5, 0.7, 0.9, 1.0\}$ and build a $(\alpha, \rho_\mathrm{median})$ lookup table. This is the most load-bearing validation step — without it, real decoder cosine cannot be mapped to a calibrated $\hat\alpha$.
- **Mean-centering control.** Subtract per-modality training-set mean from embeddings before ℓ2-normalizing, retrain, redo Diagnostic B. Tests whether the observed $\rho$ is driven by real shared structure or by the CLIP modality gap (Liang et al. 2022).
- **Dead-latent mitigation.** Add AuxK loss ($\text{weight} = 2^{-5}$) or reduce $L$ to raise alive fraction from ~14% to $\ge 50\%$. Check whether newly alive atoms land in the $\rho > 0$ regime.
- ~~Multi-VLM cross-check~~ — **Done** for 5 Base models (CLIP/MetaCLIP/DataComp/MobileCLIP2/SigLIP2) on COCO. See §5.11. Large models (L-14) available via `experiment.sh`.
- **ImageNet-1K controlled experiment.** Embeddings cached (§5.12). Train two-SAE, run Diagnostic B, compare with COCO. More structured than COCO (class labels, template texts).
- **Seed sweep for variance bars.** All real runs and followup15/16 numbers are single-seed. Three seeds before putting error bars on a figure.

### 7.2 Writing / Figure Gaps

- **Figure 2 companion table** — bridge trace / sum / $\rho_\mathrm{diag}$ across $\lambda$ are in the report but not yet in a figure-2-style panel.
- **CR curve at wide $\alpha$ for Ours / IA / GS** — only 1R/2R were run in `followup16`. Rerun with all methods across 0.1–1.0 would strengthen Figure 1.
- **Notion upload status**: `Exp-syn` page has the synthetic-setup write-up; `Real-` page has Diagnostic A + B results (tables, no figures — figures require manual drag-drop). Bridge matrix report (`report_bridge_matrix.md`) is local-only until it is considered stable.

### 7.3 Known Issues / Gotchas

- **Matplotlib mathtext** does not support `\boldsymbol`, `\displaystyle`, `\dfrac`, `\tfrac`, `\mathrm` inside some contexts, `\bigl`/`\bigr`, or `\!`/`\,`/`\;`. Use `\mathbf`, `\frac`, plain subscripts (`_{S}` not `_{\mathrm S}`), parentheses for grouping. Scripts have been patched multiple times to avoid these.
- **Notion upload tool** (`.claude/scripts/upload_md_to_notion.py`) requires `\boldsymbol{...}` and `\mathbb{R}` — not `\v` / `\sR` style custom macros. Macros will fail silently or render as raw text.
- **`same_model_flag` caveat for bridge matrix.** Single-SAE baselines have `w_dec_img == w_dec_txt` byte-for-byte; $\boldsymbol{A}$ has a trivially all-1 diagonal. Any "baseline fails at slot-level alignment" claim must acknowledge this — the bridge-diagonal story is about co-activation, not decoder cosine.
- **Followup16 only runs 1R and 2R.** A script trying to read "Ours at $\alpha = 1.0$" from followup16's `result.json` will get a missing-key error. Use followup12 for narrower $\alpha$, or run a new sweep.
- **CJK font for Korean text.** `scripts/plot_diagnostic_B_intuition.py` has Korean captions. Matplotlib needs an AppleGothic / NanumGothic font; there is a fallback list in the script. If all fonts are missing, Korean glyphs render as boxes.
- **`dead_neuron_fraction` is offline-computed** in the bridge report protocol, not stored in `result.json`. For new runs, rerun the offline block rather than expecting it in the JSON.
- **`_bootstrap.py` is required** for all `scripts/real_alpha/*.py` that import from `src.*`. It stubs the package `__init__.py` chains to avoid broken imports (`src.integrations.flex_attention` fails on transformers 5.5.3). Import it via `sys.path.insert(0, ...) ; import _bootstrap`.
- **Real-data dead latent fraction is ~85%** at $k=8$, $L \in \{4096, 8192, 16384\}$, 30 epochs, no AuxK. All Diagnostic B analysis must use alive-restricted Hungarian (firing rate $> 0.001$). Cached firing rates are saved to `diagnostic_B_firing_rates.npz` in the run dir for fast re-plots.
- **L-capacity effect on $\Delta_\mathrm{loss}$**: at $L=4096$ two-SAE is worse than one-SAE ($\Delta < 0$) because each sub-SAE only has 2048 latents. The Theorem 2 signal ($\Delta > 0$) only appears at $L \ge 8192$. Always check $L$ is large enough before interpreting the sign.
