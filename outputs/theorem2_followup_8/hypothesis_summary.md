# Hypothesis — single-decoder failure modes on partially aligned shared atoms

> Audience: project-internal. Two distinct failure modes of a single shared SAE on a partially aligned pair, each traced back to a specific result in §Theoretical Analysis: **interference** (via the bimodal Proposition) and **dimensionality reduction** (via an added alignment-loss constraint on top of Theorem 2's partition). Our method disables both by applying **modality masking** before the matching step of Algorithm 1.

## 0. Notation (matches paper §Problem Setup)

| symbol | meaning |
|---|---|
| $d$ | embedding dimension (in our basic setup $d = 256$) |
| $n = n_{\mathrm S} + n_{\mathrm I} + n_{\mathrm T}$ | per-modality latent atom count ($n_{\mathrm S}$ shared, $n_{\mathrm I}$ image-only, $n_{\mathrm T}$ text-only). Basic setup: $n_{\mathrm S}{=}1024$, $n_{\mathrm I}{=}n_{\mathrm T}{=}512$, $n{=}2048$ |
| $\mathbf{z} \in \mathbb{R}_+^n$ | monosemantic latent vector. $\mathbf{z} = [\mathbf{z}_{\mathrm S}\;\mathbf{z}_{\mathrm I}\;\mathbf{z}_{\mathrm T}]^\top$ |
| $s$ | coordinate-wise sparsity: $\Pr([\mathbf{z}]_i = 0) = s$ for every $i$. Basic setup: $s = 0.99$ |
| $\mathbf{\Phi} = [\mathbf{\Phi}_{\mathrm S}\;\mathbf{\Phi}_{\mathrm I}\;\mathbf{0}_{d\times n_{\mathrm T}}] \in \mathbb{R}^{d\times n}$ | image generative mapping, $\mathbf{x} = \mathbf{\Phi}\mathbf{z}$ |
| $\mathbf{\Psi} = [\mathbf{\Psi}_{\mathrm S}\;\mathbf{0}_{d\times n_{\mathrm I}}\;\mathbf{\Psi}_{\mathrm T}] \in \mathbb{R}^{d\times n}$ | text generative mapping, $\mathbf{y} = \mathbf{\Psi}\mathbf{z}$ |
| $\boldsymbol{\phi}_i := [\mathbf{\Phi}]_{[:,i]}$, $\boldsymbol{\psi}_i := [\mathbf{\Psi}]_{[:,i]}$ | individual atom columns (unit norm) |
| $\alpha := [\mathbf{\Phi}]_{[:,i]}^\top [\mathbf{\Psi}]_{[:,i]} = \cos(\boldsymbol{\phi}_i, \boldsymbol{\psi}_i)$ | shared-pair alignment parameter; **partially aligned** = $\alpha \in (0,1)$. Experiment sweep: $\alpha \in \{0.5, 0.7, 0.9\}$ |
| $\mathbf{V}, \mathbf{W} \in \mathbb{R}^{d \times m}$ | SAE decoders for image and text modality (paper §Problem Setup). In our code this width is called $L$; basic setup: $m = 8192$ |
| $\tilde{\mathbf{z}}_{\mathrm I} = \sigma(\mathbf{V}^\top \mathbf{x})$, $\tilde{\mathbf{x}} = \mathbf{V}\tilde{\mathbf{z}}_{\mathrm I}$ | image SAE forward |
| $\tilde{\mathbf{z}}_{\mathrm T} = \sigma(\mathbf{W}^\top \mathbf{y})$, $\tilde{\mathbf{y}} = \mathbf{W}\tilde{\mathbf{z}}_{\mathrm T}$ | text SAE forward |
| $\mathcal{L}_{\mathrm{rec}}(\mathbf{V};\mathbf{\Phi}) := \mathbb{E}_{\mathbf{z}}\,\lVert\mathbf{\Phi}\mathbf{z} - \mathbf{V}\sigma(\mathbf{V}^\top\mathbf{\Phi}\mathbf{z})\rVert_2^2$ | per-atom reconstruction loss (paper eq. `loss:reconstruction`) |
| $\mu_i' := \mathbb{E}[[\mathbf{z}]_i^2 \mid [\mathbf{z}]_i \ne 0] > 0$ | per-atom activation second moment |
| $\mathbf{\Theta} := [\mathbf{\Phi}\;\mathbf{\Psi}] \in \mathbb{R}^{d\times 2n}$ | pooled atom matrix from the bimodal Proposition |
| $\mathbf{b}_i := (\boldsymbol{\phi}_i + \boldsymbol{\psi}_i) / \lVert \boldsymbol{\phi}_i + \boldsymbol{\psi}_i \rVert$ | bisector of the shared pair; top eigenvector of $\boldsymbol{\phi}_i \boldsymbol{\phi}_i^\top + \boldsymbol{\psi}_i \boldsymbol{\psi}_i^\top$ |

"**Single decoder**" means $\mathbf{V}$ is applied to both $\mathbf{x}$ and $\mathbf{y}$ — concretely $\mathbf{W} := \mathbf{V}$. "**Modality masking**" partitions $\mathbf{V}$ into two disjoint column blocks, one for each modality.

## 1. The hypothesis in one picture

![hypothesis_figure](https://raw.githubusercontent.com/JiH00nKw0n/lvlm_hallucination/master/outputs/theorem2_followup_8/figure_hypothesis_v2.png)

**(a)** generative model from paper §Problem Setup, eq. `eq:true:mapping`.
**(b)** geometry of a shared pair $(\boldsymbol{\phi}_i, \boldsymbol{\psi}_i)$ with $\alpha \in (0,1)$:
- **(i)** ideal — two SAE columns on each endpoint,
- **(ii)** single-decoder failure by **interference**,
- **(iii)** single-decoder failure by **dimensionality reduction** from a prior alignment loss.

## 2. Failure (ii): interference  —  direct corollary of the bimodal Proposition

### 2.1 Theorem statement we are using

The paper's **Proposition** (under $s \to 1$):
$$\mathcal{L}_{\mathrm{rec}}(\mathbf{V}; \mathbf{\Phi}) + \mathcal{L}_{\mathrm{rec}}(\mathbf{V}; \mathbf{\Psi}) \;=\; \mathcal{L}_{\mathrm{rec}}(\mathbf{V}; \mathbf{\Theta}) + o(1-s), \qquad \mathbf{\Theta} = [\mathbf{\Phi}\;\mathbf{\Psi}] \in \mathbb{R}^{d \times 2n}.$$

### 2.2 What it predicts for a single decoder

Training a single $\mathbf{V}$ on the bimodal reconstruction objective is **exactly equivalent** (to leading order in $1-s$) to training it on the pooled atom set $\mathbf{\Theta}$ of size $2n$. From $\mathbf{V}$'s point of view, the effective dictionary it must reconstruct includes every $\boldsymbol{\phi}_i$ **and** every $\boldsymbol{\psi}_i$ as separate targets. The two are never "the same object" — they are two distinct unit vectors in $\mathbb{R}^d$ — but they now share the same loss term and must be served by the same decoder.

Two outcomes are theoretically possible for this pooled problem, depending on how $m$ compares with the effective atom count $2n$:

1. **Overcomplete regime** ($m \ge 2n$, which includes our basic setup where $m = 8192 > 2n = 4096$). Theorem 1 (paper's first theorem) says a perfect zero-loss minimizer **exists**:
   $$\hat{\mathbf{V}} = [\mathbf{\Theta}\;\mathbf{0}_{d\times(m-2n)}]\mathbf{P}.$$
   The global optimum places one column of $\mathbf{V}$ on every $\boldsymbol{\phi}_i$ and a separate column on every $\boldsymbol{\psi}_i$. Theorem 1 guarantees zero loss in principle — but not that training converges there.

2. **Undercomplete regime** ($m < 2n$). Theorem 2 gives the exact optimum: a partition $(\mathcal{A}_1^\star, \dots, \mathcal{A}_m^\star)$ of the atom index set such that column $\mathbf{v}_j^\star$ is the top eigenvector of
   $$\sum_{i\in\mathcal{A}_j^\star} \mu_i'\, \boldsymbol{\theta}_i \boldsymbol{\theta}_i^\top,$$
   and the partition itself maximizes $\sum_j \lambda_{\max}(\cdot)$. When a partially aligned pair $\{\boldsymbol{\phi}_i, \boldsymbol{\psi}_i\}$ lands in the same group, their outer-product sum has eigenvalue $\mu_i'(1 + \alpha)$, strictly greater than splitting them into singletons ($\mu_i'$ each). For $\alpha$ not too small, the partition optimum **prefers merging**, and the top eigenvector of the merged group is precisely the bisector $\mathbf{b}_i$ — off each endpoint by angle $\approx \alpha/2$.

### 2.3 Which regime we empirically land in

Although our basic setup has $m > 2n$ (Theorem 1's perfect solution exists), empirically the single-decoder methods (1R/GS/TA/IA) **do not** converge to that perfect solution. What we observe is (a) recovered $\alpha$-proxy inflated above the ground-truth $\alpha$, (b) strict endpoint recall `mgt_shared_tau0.99` $\approx 0$, and (c) a single merged column that best-matches both $\boldsymbol{\phi}_i$ and $\boldsymbol{\psi}_i$. This matches the Theorem 2 bisector-merge behavior — under top-$k$ sparsity of the SAE encoder (our $k_{\mathrm{enc}} = 16$, not paper's $k$), only a small subset of columns fires per sample, creating an **effective per-sample bottleneck** that pushes optimization toward partition-style merges even when the overall capacity is large enough for Theorem 1's perfect solution. Interference is therefore an **optimization-level** phenomenon that mirrors the structural prediction of Theorem 2.

### 2.4 Why call it interference

Both atoms sit inside one loss term (Proposition's pooling) and compete for the same column in a single pass through the decoder; their pulls partially cancel, so the best compromise is the bisector. Nothing in the objective tells the decoder that $\boldsymbol{\phi}_i$ and $\boldsymbol{\psi}_i$ are "the same thing" — they literally are not — but the bimodal pooling collapses them into a single group that a single column cannot simultaneously serve.

## 3. Failure (iii): dimensionality reduction  —  prior alignment loss as an added constraint

### 3.1 The auxiliary loss

Prior methods add (with our notation for a single decoder $\mathbf{W} := \mathbf{V}$):
$$\mathcal{L}_{\mathrm{align}}(\mathbf{V}; m_{\mathrm S}) \;=\; \frac{1}{n}\!\left(\sum_{i\in [m_{\mathrm S}]}\bigl(\mathrm{Corr}(\tilde{\mathbf{z}}_{\mathrm I},\tilde{\mathbf{z}}_{\mathrm T})_{ii}-1\bigr)^2 \;+\; \sum_{i\notin[m_{\mathrm S}]}\mathrm{Corr}(\tilde{\mathbf{z}}_{\mathrm I},\tilde{\mathbf{z}}_{\mathrm T})_{ii}^2\right).$$

Under a single decoder this simplifies because $\tilde{\mathbf{z}}_{\mathrm I} = \sigma(\mathbf{V}^\top \mathbf{x})$ and $\tilde{\mathbf{z}}_{\mathrm T} = \sigma(\mathbf{V}^\top \mathbf{y})$ use the **same** column $[\mathbf{V}]_{[:,i]}$ for slot $i$. Forcing $\tilde z_{\mathrm I,i} = \tilde z_{\mathrm T,i}$ on shared samples (where $[\mathbf{z}_{\mathrm S}]_i \ne 0$ so both $\boldsymbol{\phi}_i$ and $\boldsymbol{\psi}_i$ enter the loss) demands
$$[\mathbf{V}]_{[:,i]}^\top \boldsymbol{\phi}_i \;=\; [\mathbf{V}]_{[:,i]}^\top \boldsymbol{\psi}_i,$$
i.e., $[\mathbf{V}]_{[:,i]} \perp (\boldsymbol{\phi}_i - \boldsymbol{\psi}_i)$. In the 2-D plane spanned by $\{\boldsymbol{\phi}_i, \boldsymbol{\psi}_i\}$, the only directions satisfying this are proportional to the bisector $\mathbf{b}_i$.

### 3.2 What this does to Theorem 2

The auxiliary loss **restricts the feasible set** for shared-index columns: no longer can $[\mathbf{V}]_{[:,i]}$ point at $\boldsymbol{\phi}_i$ and $[\mathbf{V}]_{[:,i']}$ point at $\boldsymbol{\psi}_i$ for some other slot $i'$; both shared-index slots must lie along $\mathbf{b}_i$. Theorem 2's partition optimum now runs over a smaller space, and every partition assigning two shared-index slots to the pair $\{\boldsymbol{\phi}_i, \boldsymbol{\psi}_i\}$ collapses those slots onto the same direction. The effective rank of the shared block of $\mathbf{V}$ drops: $m_{\mathrm S}$ slots are *pinned* to at most $n_{\mathrm S}$ distinct bisector directions, and any overflow is wasted.

### 3.3 Why call it dimensionality reduction

Without the auxiliary loss, Theorem 2 still prefers some merging but can in principle split some pairs into distinct slots, preserving full column rank on the shared block. With the auxiliary loss, the symmetry constraint removes that option: **every** shared-index column in the image-mode and text-mode pair must point along the same bisector, literally reducing the dictionary's rank on the shared block. Slots that Theorem 2 would have kept distinct are now forced duplicates. The merge of §2 is not merely preserved but structurally **amplified**.

## 4. Our method: masking + matching — Theorem 1 restored per modality

### 4.1 Masking disables the Proposition's pooling

In our method, we split $\mathbf{V} \in \mathbb{R}^{d\times m}$ into two disjoint column blocks $\mathbf{V}_{\mathrm I} = \mathbf{V}[:, :m/2]$ and $\mathbf{V}_{\mathrm T} = \mathbf{V}[:, m/2:]$, and reconstruct
$$\tilde{\mathbf{x}} = \mathbf{V}_{\mathrm I}\,\sigma(\mathbf{V}_{\mathrm I}^\top \mathbf{x}), \qquad \tilde{\mathbf{y}} = \mathbf{V}_{\mathrm T}\,\sigma(\mathbf{V}_{\mathrm T}^\top \mathbf{y}).$$
The training loss now **decouples**:
$$\mathcal{L}_{\mathrm{rec}}(\mathbf{V}_{\mathrm I}; \mathbf{\Phi}) \;+\; \mathcal{L}_{\mathrm{rec}}(\mathbf{V}_{\mathrm T}; \mathbf{\Psi}),$$
i.e., the Proposition no longer applies because $\mathbf{V}_{\mathrm I}$ and $\mathbf{V}_{\mathrm T}$ are **distinct parameter sets**. Each term is now a standard single-modality reconstruction problem on $n$ atoms (not $2n$), and Theorem 1 with $m/2 \ge n$ (our basic setup: $m/2 = 4096 > n = 2048$) guarantees a perfect zero-loss solution **per modality**: $\mathbf{V}_{\mathrm I}^\star$ has one column on every $\boldsymbol{\phi}_i$ and $\mathbf{V}_{\mathrm T}^\star$ has one column on every $\boldsymbol{\psi}_i$. Bisector merging is architecturally impossible because the pair $\{\boldsymbol{\phi}_i, \boldsymbol{\psi}_i\}$ never enters the same loss term.

### 4.2 Matching applies the alignment loss to the correct pair

After Stage 1 of Algorithm 1, $\mathbf{V}_{\mathrm I}$ and $\mathbf{V}_{\mathrm T}$ have each learned endpoint-locked dictionaries, but their slot orderings are independent. Stage 1.5 of Algorithm 1 computes $\mathbf{C} = \mathrm{Corr}(\tilde{\mathbf{z}}_{\mathrm I}, \tilde{\mathbf{z}}_{\mathrm T})$ and greedily finds the Hungarian permutations $\mathbf{P}_{\mathrm I}, \mathbf{P}_{\mathrm T}$ that align slot $k$ of $\mathbf{V}_{\mathrm I}\mathbf{P}_{\mathrm I}$ with slot $k$ of $\mathbf{V}_{\mathrm T}\mathbf{P}_{\mathrm T}$ whenever the two co-fire — identifying these as the "same shared-index $i$" after training.

Now Stage 2 applies the auxiliary loss $\lambda\,\mathcal{L}_{\mathrm{aux}}(\mathbf{V}_{\mathrm I},\mathbf{V}_{\mathrm T}; m_{\mathrm S})$. The symmetry constraint $\tilde z_{\mathrm I, k} = \tilde z_{\mathrm T, k}$ now reads
$$[\mathbf{V}_{\mathrm I}]_{[:,k]}^\top \boldsymbol{\phi}_i \;=\; [\mathbf{V}_{\mathrm T}]_{[:,k]}^\top \boldsymbol{\psi}_i,$$
and **crucially**, $[\mathbf{V}_{\mathrm I}]_{[:,k]}$ and $[\mathbf{V}_{\mathrm T}]_{[:,k]}$ are two **independent vectors** (they live in disjoint parameter blocks), so the constraint can be satisfied with $[\mathbf{V}_{\mathrm I}]_{[:,k]} = \boldsymbol{\phi}_i$ and $[\mathbf{V}_{\mathrm T}]_{[:,k]} = \boldsymbol{\psi}_i$ — directly matching the Theorem 1 optimum per modality. No bisector collapse is required.

### 4.3 The sign flip of the alignment loss

The same auxiliary loss formula $\mathcal{L}_{\mathrm{aux}}$ is:
- **harmful** in §3 because it acts on a single shared column $[\mathbf{V}]_{[:,i]}$, forcing bisector collapse;
- **helpful** here because it acts on a pair of independent columns $([\mathbf{V}_{\mathrm I}]_{[:,k]}, [\mathbf{V}_{\mathrm T}]_{[:,k]})$, forcing only their mutual cross-modal correspondence.

The difference is whether the two latents being equated share a column or live in disjoint halves — which is exactly what masking controls.

## 5. Testable 2×2 design

| | $\lambda_{\mathrm{align}} = 0$ | $\lambda_{\mathrm{align}} > 0$ |
|---|---|---|
| **masking = off** | single decoder → **interference** (§2) | **dim. reduction** amplified (§3) |
| **masking = on** | two half-decoders, endpoint-locked but no correspondence yet | **ours**: endpoint-locked + cross-modal slot pairing (§4) |

All four cells share architecture family (one decoder, split or not), loss form (same $\mathcal{L}_{\mathrm{aux}}$), and all hyperparameters. The experiment isolates:

- **Does merging happen?** ⇒ controlled by the masking switch (whether the Proposition's pooling applies).
- **Is $\mathcal{L}_{\mathrm{aux}}$ helpful or harmful?** ⇒ controlled by whether the two latents it equates share a column or not, i.e., by masking combined with $\lambda$.

### Predicted signatures

| metric | off, $\lambda{=}0$ | off, $\lambda{\uparrow}$ | on, $\lambda{=}0$ | on, $\lambda{\uparrow}$ |
|---|---|---|---|---|
| `merged_fraction` (new) | moderate | $\to 1$ | 0 | 0 |
| `bisector_alignment` (new) | moderate | $\to 1$ | low | low |
| `mgt_shared_tau0.99` | 0 | 0 | $\sim 0.2$ | $\sim 0.2$ |
| `cross_cos_gt` (= $\alpha$-proxy) | $> \alpha$ | $\to 1$ | $\approx \alpha$ | $\approx \alpha$ |
| `probe_top1_agree` (XMA raw) | $\alpha$-dependent | $\alpha$-dependent | $\alpha$-independent (after match) | high, $\alpha$-independent |

### Two headline figures

1. **`merged_fraction` vs $\lambda$**. Masking off climbs to 1 (§3's dimensionality reduction); masking on stays flat at 0 (§4.1's decoupling). Directly separates the Proposition-driven merge from the Theorem 1 decoupling.
2. **`mgt_shared_tau0.99` vs $\lambda$**. Masking off stays at 0 regardless of $\lambda$ (bisector is never an endpoint); masking on rises with $\lambda$ (correctly paired columns reach cross-modal identity without sacrificing endpoints). The auxiliary loss sign-flips from harmful to helpful by the masking switch alone.

## 6. Why this is the key experiment

Earlier follow-ups compared bundled method families (1R / 2R / GS / TA / IA / ours), so the architectural effect (one shared $\mathbf{V}$ vs. split $\mathbf{V}_{\mathrm I},\mathbf{V}_{\mathrm T}$) and the alignment-loss effect were confounded. The 2×2 design flips one architectural switch (masking) and sweeps one scalar ($\lambda$), giving a minimal falsification test for:

1. **Proposition-driven interference** (single decoder on pooled atoms merges via loss concatenation),
2. **$\mathcal{L}_{\mathrm{aux}}$-driven dimensionality reduction** (shared-index columns pinned to bisectors).

Our method avoids both by **disabling the Proposition's pooling through masking** and then **applying $\mathcal{L}_{\mathrm{aux}}$ on the correctly-matched pair of independent columns**, so the same loss term that collapses dimensionality under §3 instead enforces only cross-modal correspondence.
