# Hypothesis — single-decoder failure modes on partially aligned shared atoms

> Audience: project-internal (terminology known). Explains **why** a single shared decoder fails on partially aligned shared pairs, and **why** a prior alignment loss amplifies that failure, through two distinct mechanisms: **interference** and **dimensionality reduction**.

## 0. Terminology

- **Shared atom pair** $(\boldsymbol{\phi}_i, \boldsymbol{\psi}_i)$ — the image and text columns of $\mathbf{\Phi}_{\mathrm S}, \mathbf{\Psi}_{\mathrm S}$ that correspond to the same shared latent coordinate $i \in [n_{\mathrm S}]$.
- **Partially aligned** — $\cos(\boldsymbol{\phi}_i, \boldsymbol{\psi}_i) = \alpha$ with $\alpha \in (0, 1)$ (neither orthogonal nor identical). Our default data regime.
- **Single decoder** — a single matrix $\mathbf{V}$ applied to both $\mathbf{x}$ (image) and $\mathbf{y}$ (text). 1R / GS / TA / IA in the paper's shorthand.
- **Modality masking** — partitioning the decoder so that $\mathbf{x}$ and $\mathbf{y}$ are reconstructed by disjoint column sets. 2R / ours.
- **Matching** — Algorithm 1's correlation-based Hungarian permutation that establishes a cross-modal slot correspondence after Stage 1.

## 1. The hypothesis in one picture

![hypothesis_figure](https://raw.githubusercontent.com/JiH00nKw0n/lvlm_hallucination/master/outputs/theorem2_followup_8/figure_hypothesis.png)

**(a)** our synthetic bimodal generative model — the shared latent $\mathbf{z}$ is mixed into both modalities through $\mathbf{\Phi}, \mathbf{\Psi}$, and a single-decoder SAE reconstructs each modality with the same parameter matrix.

**(b)** geometry of a single shared pair under three conditions:
- **(i)** Ideal — two dictionary columns separately land on $\boldsymbol{\phi}_i$ and $\boldsymbol{\psi}_i$.
- **(ii)** Naive single decoder — **interference** forces one shared column to the bisector.
- **(iii)** + prior alignment loss — **dimensionality reduction** forces additional slots to collapse along the same bisector.

## 2. Cause of failure (ii): interference across modalities

When both modalities share one decoder column, every time the shared latent $[\mathbf{z}_{\mathrm S}]_i$ fires, **both** $\boldsymbol{\phi}_i$ and $\boldsymbol{\psi}_i$ contribute to the reconstruction target. A single column $\mathbf{v}$ cannot be close to both endpoints simultaneously when $\alpha < 1$, so the best it can do — minimizing combined reconstruction error — is to sit on the **bisector** $(\boldsymbol{\phi}_i + \boldsymbol{\psi}_i)/\lVert\cdot\rVert$. The residual error is distributed symmetrically: each endpoint is missed by an angle of $\alpha/2$.

This is the geometric consequence of the two atoms competing for the same capacity in the shared decoder — "interference". The recovered direction looks *more aligned than it should be*: the measured $\alpha_{\mathrm{proxy}}$ (cosine of best-matching columns across modalities) inflates toward $1$, even though neither endpoint is actually recovered.

**Prediction**: under a single decoder with only reconstruction loss,
- strict endpoint recall `mgt_shared_tau0.99` is structurally $\approx 0$,
- `cross_cos_gt` inflates above the ground-truth $\alpha$,
- `merged_fraction` (new metric: fraction of shared pairs where the image-best column equals the text-best column) is nontrivially positive,
- the best-matching column points near the bisector (`bisector_alignment` high).

## 3. Cause of failure (iii): dimensionality reduction by alignment loss

Prior alignment losses (e.g., $\mathrm{Corr}(\tilde z_{\mathrm I, i}, \tilde z_{\mathrm T, i}) \to 1$ in GS, TA, IA) add an explicit term saying "the activation of slot $i$ for an image input and the activation of slot $i$ for a text input must be equal". Under a single decoder, that is a direct order to **use the same column** $\mathbf{v}_i$ for both $\boldsymbol{\phi}_i$ and $\boldsymbol{\psi}_i$. The only direction that satisfies this for all partially aligned pairs is, again, the bisector — but now several distinct latent slots end up pinned to that same direction, producing a **dimensionality reduction**: the effective rank of the learned dictionary collapses. Multiple columns are wasted representing what is geometrically a single merged direction.

This is strictly worse than case (ii): interference already forced one column onto the bisector; the alignment loss now duplicates that column across slots. Nothing is added; capacity is removed.

**Prediction**: adding $\lambda \mathcal{L}_{\mathrm{align}}$ with no masking should
- leave strict recall at 0 (merging unchanged),
- increase `merged_fraction` further as $\lambda$ grows,
- increase `bisector_alignment` further (columns more sharply pinned),
- waste slots (fewer distinct directions in the dictionary; lower rank of the learned basis).

## 4. Our fix: modality masking + matching

**Masking**: if $\boldsymbol{\phi}_i$ is only ever reconstructed by $\mathbf{V}[:, :L/2]$ and $\boldsymbol{\psi}_i$ only by $\mathbf{V}[:, L/2:]$, the two halves have **disjoint parameter sets**. Interference across modalities cannot occur — each half only sees its own endpoint as a reconstruction target, so each half can place a column exactly on its endpoint. The bisector attractor disappears at the architectural level.

**Matching**: after Stage 1, the two halves have learned endpoint-locked dictionaries but with arbitrary slot orderings. Algorithm 1's correlation-based Hungarian pass finds the permutation that pairs image slot $k_{\mathrm I}$ with text slot $k_{\mathrm T}$ whenever they co-fire on shared latents. Applying an alignment loss **after** this permutation pulls the correct pair toward cross-modal identity — the image slot pointing at $\boldsymbol{\phi}_i$ and the text slot pointing at $\boldsymbol{\psi}_i$ — leaving the endpoint-locked positions undisturbed and only adding cross-modal slot coherence.

In the failure modes above, "alignment loss" was harmful because it operated on a single shared column and forced collapse. In our method, the same form of alignment loss is applied **to different columns** (one in each half), so it can only influence their **correspondence**, not their direction — turning the same loss term from harmful to helpful.

## 5. Testable 2×2 design

| | $\lambda_{\mathrm{align}} = 0$ | $\lambda_{\mathrm{align}} > 0$ |
|---|---|---|
| **masking = off** | naive single decoder → **interference merge** (ii) | **merge amplified** → dimensionality reduction (iii) |
| **masking = on** | two half-decoders, no coherence yet | **ours**: merge impossible + slot coherence via matching |

All four cells share architecture, loss form, hyperparameters; only the masking switch and the scalar $\lambda$ vary. Under the basic setup ($m{=}256$, $n_{\mathrm S}{=}1024$, $L{=}8192$, Exp$(1)$ + $\mathcal{N}(0,0.1^2)$, $k{=}16$), the experiment isolates:

- *"Does merging happen?"* → determined by the masking switch.
- *"Is the alignment loss helpful or harmful?"* → determined by whether matching was done first, i.e., by the combination with masking.

### Predicted signatures

| metric | off, λ=0 | off, λ↑ | on, λ=0 | on, λ↑ |
|---|---|---|---|---|
| `merged_fraction` (new) | moderate | **→ 1** | 0 | 0 |
| `bisector_alignment` (new) | moderate | **→ 1** | low | low |
| `mgt_shared_tau0.99` | 0 | 0 | ~0.2 | ~0.2 |
| `cross_cos_gt` (= α proxy) | > α | **→ 1** | ≈ α | ≈ α |
| `probe_top1_agree` (XMA raw) | α-dependent | α-dependent | α-independent | **high, α-independent** |

### Two headline figures

1. **`merged_fraction` vs $\lambda$** — masking off climbs toward 1, masking on stays flat at 0. Direct evidence that alignment-loss-driven dimensionality reduction is real and that masking blocks interference at the architecture level.
2. **`mgt_shared_tau0.99` vs $\lambda$** — masking off stays at 0 regardless of $\lambda$, masking on rises with $\lambda$. Alignment loss is *harmful* without masking and *helpful* with masking — sign flipped by the masking switch alone.

## 6. Why this is the key experiment

Earlier follow-ups bundled architecture family (1R / 2R / GS / TA / IA / ours) together, so the masking effect and the alignment-loss effect were confounded. The 2×2 design above isolates them with a single architectural switch and a single scalar, giving a minimal falsification test for the two-mechanism story: **interference** (single decoder, no explicit pull) and **dimensionality reduction** (single decoder, explicit pull).
