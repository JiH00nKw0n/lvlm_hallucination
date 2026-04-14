## Evaluation Metrics (in setup notation)

Notation recap: $\vx,\vy\in\sR^d$ are image/text embeddings, $\vV,\vW\in\sR^{d\times m}$ the image/text SAE dictionaries with columns $\vv_j,\vw_j$, $\sigma$ the (top-$k$) ReLU activation, and
$\tilde\vx=\vV\sigma(\vV^\top\vx)$, $\tilde\vy=\vW\sigma(\vW^\top\vy)$.
Shared ground-truth atoms are $\vphi_i^{\mathrm S}:=[\vPhi_{\mathrm S}]_{[:,i]}$ and $\vpsi_i^{\mathrm S}:=[\vPsi_{\mathrm S}]_{[:,i]}$ for $i\in[n_{\mathrm S}]$.

### 1. Reconstruction error

Population form (matches $\loss_{\mathrm{rec}}$ in the paper):
```latex
\loss_{\mathrm{rec}}(\vV,\vW)
\;:=\;
\E_{\vz}\!\left[\,\norm{\vx-\tilde\vx}_2^2 + \norm{\vy-\tilde\vy}_2^2\,\right].
```

Empirical (eval-set) form actually reported in `outputs/theorem*`. Following the paper's convention of expressing losses as expectations rather than introducing a sample-count symbol, write
```latex
\widehat{\loss}_{\mathrm{rec}}
\;:=\;
\E_{(\vx,\vy)\sim\sD_{\mathrm{eval}}}\!\left[\,
\norm{\vx-\tilde\vx}_2^2 \;+\; \norm{\vy-\tilde\vy}_2^2
\,\right],
```
where $\sD_{\mathrm{eval}}$ is the held-out set of paired embeddings and $\tilde\vx,\tilde\vy$ their SAE reconstructions. (This is the population loss in~\eqref{loss:reconstruction:full} evaluated under the empirical eval distribution.)

### 2. Ground-truth Recovery Rate $\mathrm{GRR}(\tau)$

Writing the image side only (the text side is defined analogously by replacing $(\vV,\vPhi_{\mathrm S})$ with $(\vW,\vPsi_{\mathrm S})$):
```latex
\mathrm{GRR}_{\mathrm I}(\tau)
\;:=\;
\frac{1}{n_{\mathrm S}}\sum_{i\in[n_{\mathrm S}]}
\mathbf{1}\!\left[\,
\max_{j\in[m]} \big|\cos\!\big([\vV]_{[:,j]},\,[\vPhi_{\mathrm S}]_{[:,i]}\big)\big| \;>\; \tau
\,\right],
```
where $\cos(\cdot,\cdot)$ is the cosine similarity between two vectors and $\tau\in[0,1]$. We report the two-modality average $\mathrm{GRR}(\tau):=\tfrac{1}{2}(\mathrm{GRR}_{\mathrm I}(\tau)+\mathrm{GRR}_{\mathrm T}(\tau))$, since the image and text sides are nearly symmetric in our experiments.

### 3. Mean Correlation Coefficient $\mathrm{MCC}$ (Hungarian matching)

$\mathrm{GRR}(\tau)$ allows multiple shared GTs to be "recovered" by the same dictionary column, which over-counts when one latent collapses two GTs. To rule this out, take the optimal *injective* assignment $\pi:[n_{\mathrm S}]\hookrightarrow[m]$ via the Hungarian algorithm and report the average matched cosine. On the image side:
```latex
\mathrm{MCC}_{\mathrm I}
\;:=\;
\frac{1}{\min(m,\,n_{\mathrm S})}
\max_{\pi:[n_{\mathrm S}]\hookrightarrow[m]}
\sum_{i\in[n_{\mathrm S}]}
\big|\cos\!\big([\vV]_{[:,\pi(i)]},\,[\vPhi_{\mathrm S}]_{[:,i]}\big)\big|.
```
The text side $\mathrm{MCC}_{\mathrm T}$ is defined analogously with $(\vW,\vPsi_{\mathrm S})$, and we report the two-modality average $\mathrm{MCC}:=\tfrac{1}{2}(\mathrm{MCC}_{\mathrm I}+\mathrm{MCC}_{\mathrm T})$.

### 4. Feature Uniqueness $\mathrm{Uniq}$ (latent diversity)

$\mathrm{MCC}$ rewards good per-pair alignment but does not directly penalise *how many* distinct GTs the dictionary actually claims. Following Chanin & Garriga-Alonso (2026), we additionally measure: for each learned latent $j\in[m]$, take its best-matching shared GT
```latex
i^\star_{\mathrm I}(j)
\;:=\;
\argmax_{i\in[n_{\mathrm S}]}
\big|\cos\!\big([\vV]_{[:,j]},\,[\vPhi_{\mathrm S}]_{[:,i]}\big)\big|,
```
and count how many distinct GTs are claimed across all latents:
```latex
\mathrm{Uniq}_{\mathrm I}
\;:=\;
\frac{\big|\{\,i^\star_{\mathrm I}(j) : j\in[m]\,\}\big|}{\min(m,\,n_{\mathrm S})}
\;\in\;[0,1].
```
A value of $1$ means every shared GT is claimed by at least one latent (no collapse); a small value means many latents pile onto a few GTs. The text side $\mathrm{Uniq}_{\mathrm T}$ is defined analogously with $(\vW,\vPsi_{\mathrm S})$, and we report $\mathrm{Uniq}:=\tfrac{1}{2}(\mathrm{Uniq}_{\mathrm I}+\mathrm{Uniq}_{\mathrm T})$.

### 5. Joint Ground-truth Recovery Rate $\mathrm{JGRR}(\tau)$

Per-modality $\mathrm{GRR}$ only asks whether $\vV$ and $\vW$ each *individually* recover the shared GT; it does not require the *same latent index* $j$ in $\vV$ and $\vW$ to represent the *same* shared atom. $\mathrm{JGRR}$ strengthens $\mathrm{GRR}$ by requiring both modalities to approximate a shared GT at a single shared index:
```latex
\mathrm{JGRR}(\tau)
\;:=\;
\frac{1}{n_{\mathrm S}}\sum_{i\in[n_{\mathrm S}]}
\mathbf{1}\!\left[\;
\max_{j\in[m]}\,
\tfrac{1}{2}\!\left(
\big|\cos\!\big([\vV]_{[:,j]},\,[\vPhi_{\mathrm S}]_{[:,i]}\big)\big|
\;+\;
\big|\cos\!\big([\vW]_{[:,j]},\,[\vPsi_{\mathrm S}]_{[:,i]}\big)\big|
\right)
\;>\;\tau
\,\right].
```
For single-SAE methods ($\vV=\vW$), this reduces to asking whether a *single* dictionary atom approximates *both* modality directions of the same shared GT simultaneously — a geometric constraint that is generally unsatisfiable when $\cos([\vPhi_{\mathrm S}]_{[:,i]},[\vPsi_{\mathrm S}]_{[:,i]})$ is far from $1$.

**Post-hoc re-indexed variant.** For a trained $(\vV,\vW)$, let $C\in\sR^{m\times m}$ be the Pearson correlation matrix between the dense top-$k$ latents $\sigma(\vV^{\!\top}\vx)$ and $\sigma(\vW^{\!\top}\vy)$ on the training set, and let
```latex
\pi^\star \;:=\; \argmax_{\pi\in\mathfrak{S}_m}\sum_{j\in[m]}\big|C_{j,\pi(j)}\big|
```
be the optimal $1$-to-$1$ relabelling (Hungarian assignment on $-|C|$). Writing $\vW^{\pi^\star}$ for $\vW$ with its columns permuted by $\pi^\star$, set
```latex
\mathrm{JGRR}^{\mathrm{Hung}}(\tau) \;:=\; \mathrm{JGRR}(\tau)\ \text{computed with}\ (\vV,\,\vW^{\pi^\star}).
```
This is the best joint recovery achievable by any post-hoc index relabelling, and serves as the "trivial alignment" baseline for any training-time alignment method.

### 6. Mean Joint Match $\overline{\mathrm{JM}}$ (threshold-free)

$\mathrm{JGRR}$ is a hard-thresholded count and tends to cluster near $0$ or near a random-pairing baseline $\sim\!0.5$, which compresses smooth differences into a bimodal picture. A threshold-free continuous summary is useful:
```latex
\overline{\mathrm{JM}}
\;:=\;
\frac{1}{n_{\mathrm S}}\sum_{i\in[n_{\mathrm S}]}
\max_{j\in[m]}\,
\tfrac{1}{2}\!\left(
\big|\cos\!\big([\vV]_{[:,j]},\,[\vPhi_{\mathrm S}]_{[:,i]}\big)\big|
\;+\;
\big|\cos\!\big([\vW]_{[:,j]},\,[\vPsi_{\mathrm S}]_{[:,i]}\big)\big|
\right)
\;\in\;[0,1].
```
The post-hoc variant $\overline{\mathrm{JM}}^{\mathrm{Hung}}$ is defined with $\vW$ replaced by $\vW^{\pi^\star}$.

### 7. Cross-Modal Alignment $\mathrm{XMA}$ (GT-probe encoder test)

$\mathrm{JGRR}$ and $\overline{\mathrm{JM}}$ test cross-modal agreement through decoder dictionary geometry. Complementarily, we probe whether the *encoders* $\sigma(\vV^{\!\top}\cdot)$ and $\sigma(\vW^{\!\top}\cdot)$ route the same shared concept to the same latent slot, by feeding an idealised single-atom stimulus to each side. For every shared GT index $i\in[n_{\mathrm S}]$, we compute the two SAE responses
```latex
\tilde{\vz}_{\mathrm I}^{(i)} \;:=\; \sigma\!\big(\vV^{\!\top}[\vPhi_{\mathrm S}]_{[:,i]}\big)\in\sR^{m},
\qquad
\tilde{\vz}_{\mathrm T}^{(i)} \;:=\; \sigma\!\big(\vW^{\!\top}[\vPsi_{\mathrm S}]_{[:,i]}\big)\in\sR^{m}.
```
Here $\tilde\vz_{\mathrm I}^{(i)}$ and $\tilde\vz_{\mathrm T}^{(i)}$ are the top-$k$ SAE latents that each modality assigns to the pure shared concept $i$ (cf.\ the paper's $\tilde\vz_{\mathrm I},\tilde\vz_{\mathrm T}$ in Section~\ref{sec:setup}, specialised to the single-atom input $\vx=[\vPhi_{\mathrm S}]_{[:,i]}$, $\vy=[\vPsi_{\mathrm S}]_{[:,i]}$). The three $\mathrm{XMA}$ variants below aggregate over the $n_{\mathrm S}$ probes.

**Cross-Modal Alignment (top-$1$)** — fraction of shared concepts for which both modality SAEs fire strongest at the same latent slot:
```latex
\mathrm{XMA}
\;:=\;
\frac{1}{n_{\mathrm S}}\sum_{i\in[n_{\mathrm S}]}
\mathbf{1}\!\left[\,
\argmax_{j\in[m]}\big[\tilde{\vz}_{\mathrm I}^{(i)}\big]_{j}
\;=\;
\argmax_{j\in[m]}\big[\tilde{\vz}_{\mathrm T}^{(i)}\big]_{j}
\,\right].
```
A value of $1$ means every shared concept is routed to a consistent latent index across modalities; values near $1/m$ indicate random pairing.

**Cross-Modal Alignment (cosine)** — a continuous full-vector version of $\mathrm{XMA}$:
```latex
\overline{\mathrm{XMA}}^{\cos}
\;:=\;
\frac{1}{n_{\mathrm S}}\sum_{i\in[n_{\mathrm S}]}
\cos\!\big(\tilde{\vz}_{\mathrm I}^{(i)},\,\tilde{\vz}_{\mathrm T}^{(i)}\big).
```

**Cross-Modal Alignment (top-$k$ Jaccard)** — how much of each sparse active set is shared between modalities:
```latex
\mathrm{XMA}^{\mathrm{Jacc}}
\;:=\;
\frac{1}{n_{\mathrm S}}\sum_{i\in[n_{\mathrm S}]}
\frac{\big|\,\supp\tilde{\vz}_{\mathrm I}^{(i)}\cap\supp\tilde{\vz}_{\mathrm T}^{(i)}\,\big|}
     {\big|\,\supp\tilde{\vz}_{\mathrm I}^{(i)}\cup\supp\tilde{\vz}_{\mathrm T}^{(i)}\,\big|},
```
where $\supp\vz:=\{j\in[m]\,:\,[\vz]_j\neq 0\}$ and $|\supp\tilde\vz_{\mathrm I}^{(i)}|=|\supp\tilde\vz_{\mathrm T}^{(i)}|=k$ by top-$k$ sparsity.

These three answer the functional question: *when an isolated shared concept is presented to both modalities, do the two SAEs agree on which latent slot represents it?*. Unlike $\mathrm{JGRR}$ and $\overline{\mathrm{JM}}$, $\mathrm{XMA}$ depends only on *actively firing* slots under the GT stimulus, so it is insensitive to dead latents or idle shared-block slots.

### 8. Merge Rate $\mathrm{MR}$ (shared-pair partition indicator)

$\mathrm{MR}$ directly measures whether the learned dictionary uses a single column to reconstruct *both* modality directions of a shared atom (i.e., collapses the pair $([\vPhi_{\mathrm S}]_{[:,i]},[\vPsi_{\mathrm S}]_{[:,i]})$ onto one latent), as opposed to allocating two distinct columns at the two endpoints. For each shared GT index $i\in[n_{\mathrm S}]$, define the decoder-side best-match indices using signed cosine (consistent with the top-$k$ ReLU encoder, under which a column anti-aligned with an atom cannot fire on it):
```latex
j^\star_{\mathrm I}(i)
\;:=\;
\argmax_{j\in[m]}\,
\cos\!\big([\vV]_{[:,j]},\,[\vPhi_{\mathrm S}]_{[:,i]}\big),
\qquad
j^\star_{\mathrm T}(i)
\;:=\;
\argmax_{j\in[m]}\,
\cos\!\big([\vW]_{[:,j]},\,[\vPsi_{\mathrm S}]_{[:,i]}\big),
```
and set
```latex
\mathrm{MR}
\;:=\;
\frac{1}{n_{\mathrm S}}\sum_{i\in[n_{\mathrm S}]}
\mathbf{1}\!\left[\, j^\star_{\mathrm I}(i) \;=\; j^\star_{\mathrm T}(i) \,\right]
\;\in\;[0,1].
```
The interpretation depends on the architecture:
- **Single-decoder methods** ($\vV=\vW$): a high value means the same column is the top match for both endpoints, i.e., that column sits near the bisector of $([\vPhi_{\mathrm S}]_{[:,i]},[\vPsi_{\mathrm S}]_{[:,i]})$ instead of at either endpoint. In this architecture $\mathrm{MR}$ is the direct signature of Theorem 2 partition merging.
- **Two-SAE without training-time pairing**: $\vV$ and $\vW$ have independent slot orderings, so two endpoint-locked columns landing on the *same* index number is a chance event with probability $\approx 1/m$. $\mathrm{MR}\approx 0$ here even when both endpoints are recovered.
- **Two-SAE with matching (ours)**: Algorithm 1's Hungarian permutation relabels slots so that column $j$ of $\vV$ and column $j$ of $\vW$ represent the same shared concept. A high value here means the matching succeeded, i.e., endpoint-locked columns share a common index — the desired state.

To resolve the "high $\mathrm{MR}$" ambiguity, $\mathrm{MR}$ must be read together with $\mathrm{GRR}(\tau)$ at a strict threshold (e.g., $\tau=0.99$):
- $\mathrm{MR}$ high with $\mathrm{GRR}(0.99)\approx 0$ ⇒ bisector merge (single-decoder failure mode).
- $\mathrm{MR}$ high with $\mathrm{GRR}(0.99)$ nontrivial ⇒ correctly matched endpoint recovery.
