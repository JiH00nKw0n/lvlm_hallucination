# Theorem 2 / Algorithm 1 실험 보고서 — latent_size = 2048

> 본 보고서는 `.claude/skills/experiment-report.md` 규약을 따르며, §1–§3은 main /
> ablation 결과를 확인하기 전에 작성되었다. §4 이후는 결과가 나온 뒤 채운다.

## §0. 한 줄 결론

*(결과 확인 후 작성)*

---

## §1. Background / Motivation

Vision-language 모델의 공동 임베딩 공간을 Sparse Autoencoder(SAE)로 분해하려는
접근은 최근 활발하지만, **두 modality의 latent가 같은 index에서 같은 concept를
가리키도록 정렬되지 않는다**는 문제가 반복적으로 보고된다. 본 프로젝트의 이론
파트(Theorem 1)는 **단일 SAE(shared decoder)** 와 **독립 학습된 두 SAE** 모두
multi-modal 설정에서 구조적인 실패 모드를 가진다는 점을 synthetic 생성 모델 위에서
보였다.

Theorem 2 / Algorithm 1은 이 실패 모드를 **두 단계 학습 + greedy permutation**
으로 교정하는 방법을 제안한다:

1. **Stage 1** — 두 modality의 SAE $V, W$를 `L_rec`만으로 독립 학습.
2. **Permutation** — 학습 데이터 전체에서 latent correlation 행렬
   $C = \mathrm{Corr}(\tilde z_{\mathrm I}, \tilde z_{\mathrm T})$를 계산하고, greedy
   알고리즘으로 대각을 최대화하도록 row/col permutation을 적용.
3. **Stage 2** — 권장 $m_S$개 대각을 1로, 나머지 대각을 0으로 미는
   diagonal-only auxiliary loss와 함께 $V, W$를 joint fine-tune:
   $\mathcal L_\text{aux} = \sum_{i\in[m_S]} (C_{ii}-1)^2 + \sum_{i\notin[m_S]} C_{ii}^2$.

본 실험의 목적은 이 알고리즘이 Theorem 1의 실패 모드를 실제로 교정하는지,
그리고 기존의 prior-art auxiliary loss (group-sparse L$_{2,1}$ / trace alignment)
들과 비교해 정량적 이득이 있는지를 **합성 생성 모델** 위에서 검증하는 것이다.

---

## §2. Hypothesis + Pre-registered Predictions

### Hypothesis (H)

> **H (Algorithm 1):** 주어진 synthetic 생성 모델 하에서, Algorithm 1의
> `Stage 1 → 탐욕적 permutation → Stage 2 (diag-only $\mathcal L_\text{aux}$)`
> 은 독립 학습된 두 SAE의 **cross-modal shared latent 정렬** 을
> 같은 학습 예산을 쓰는 baseline (single-recon / two-recon /
> group-sparse / trace-align) 보다 유의하게 높이며, 이 과정에서
> reconstruction 품질은 두 개 독립 SAE 수준을 유지한다.

latent_size $m = 2048$, ground-truth $n_{\mathrm S} = 512$, $n_{\mathrm I}=n_{\mathrm T}=256$,
$d=768$, $k=16$, $s=0.99$, paired batch 256, 10 epochs, 3 seeds, $\alpha \in
\{0.3, 0.6, 0.9, 1.0\}$라는 실험 scope 하에서만 검증한다 (§3 참조).

### Notation for the predictions

모든 prediction은 **eval 분할** 에서 계산된 값으로 판정한다:
- `avg_eval_loss` — 두 modality의 recon loss 평균.
- `top_mS` ≔ `cross_cos_top_mS_mean` — permutation 적용 후 대각 correlation의
  상위 $m_S$ 평균. Ours는 `m_S=512` supplied를, baseline은 oracle
  `m_S = n_shared = 512`를 사용해 동일 기준으로 측정한다.
- `rest` ≔ `cross_cos_rest_mean` — 나머지 대각 평균.
- `img_mgt_shared`, `txt_mgt_shared` — decoder 행이 $\phi_S, \psi_S$ GT 원자와
  $\tau=0.8$ 기준 cosine 매칭되는 비율.

### P1 [main] — Ours가 top-$m_S$ 대각 정렬을 가장 높게 달성한다.

- **측정:** 각 $\alpha$, seed에서 `ours::lam1.0_mS512_k4 / cross_cos_top_mS_mean`
  vs. 네 baseline의 동일 metric.
- **Pass:** $\alpha \in \{0.6, 0.9, 1.0\}$의 모든 cell에서
  `ours`의 seed-평균이 다른 네 method의 seed-평균보다 **절대치 +0.10 이상** 크다.
  (Lower bound는 $\alpha = 0.3$처럼 shared atom cosine이 매우 낮은 극단 케이스에서는
  ours도 permutation 정확도가 떨어질 수 있다고 예상하므로 완화한다.)
- **Fail:** 위 세 $\alpha$ 중 하나라도 `ours` mean이 baseline 최대 mean 대비
  +0.10 이하인 경우.
- **Failure scenario:** Stage 2의 diag-only loss가 Stage 1이 만든 배치 상관에
  오버핏해 eval 상관은 올리지 못하거나, permutation 단계가 학습되지 않은 잡음
  latent들을 매칭해 top-$m_S$ 자체가 무의미해지는 경우.
- **Intended interpretation:** Algorithm 1의 `permutation + diag-only loss`
  조합이 정렬을 직접 유도한다 (H).
- **Alternative explanations:**
  (a) Ours는 Stage 1 end에서 permutation으로 `top_mS` 값을 이미 최대치 근처까지
  끌어올리므로, Stage 2는 실제 학습 이득이 아닌 **relabel만으로도** 이 metric을
  올릴 수 있다. → P5로 분리 검증.
  (b) Ours의 두 SAE는 각각 `latent=1024` 이므로 shared-decoder baseline
  (`latent=2048` 단일 SAE) 대비 **총 decoder 파라미터는 동일** (2 × 1024 = 2048).
  따라서 단순 파라미터 수 우위 설명은 이 실험에서는 배제된다. 다만 **per-SAE
  표현력** 이 낮아지는 trade-off는 존재한다.
  (c) Ours의 Stage 2는 두 SAE가 본래 잘 매칭되는 `m_S=512`를 공급받는다
  (oracle). oracle 정보 유출이 이득의 원인일 수 있다. → L=2048에서는 ablation
  으로 측정, 별도 `m_S` 오라클 제거 실험은 추후 범위 밖.

### P2 [main] — Ours의 reconstruction은 two_recon 대비 저하되지 않는다.

- **측정:** 각 $\alpha$에서 `ours::lam1.0_mS512_k4 / avg_eval_loss` vs.
  `two_recon / avg_eval_loss`.
- **Pass:** 모든 $\alpha \in \{0.3, 0.6, 0.9, 1.0\}$에서
  `ours`의 seed-평균이 `two_recon` seed-평균의 **1.10배 이내**.
- **Fail:** 어느 한 cell이라도 1.10배 초과.
- **Failure scenario:** Stage 2의 `λ·L_aux`가 recon과 경쟁해 정렬은 얻되 recon이
  심하게 손상되는 경우 (즉, alignment와 recon 사이 trade-off가 실제로 1 이상
  발생).
- **Intended interpretation:** Algorithm 1이 `추가 정렬` 을 무료로 얻는다
  (reconstruction cost 거의 없음) — H의 부차 주장.
- **Alternative explanations:**
  (a) λ=1이 작아 aux가 거의 0이라 recon에 영향 없을 수 있다. → λ ablation으로
  확인.
  (b) paired batch의 recon loss가 batch 단위로 평균되므로, 동일 배치 모두에
  노출된 두 recon term이 서로 regularize하는 효과가 이득으로 나타날 수 있다. →
  two_recon도 같은 batch를 쓰므로 baseline과의 차이는 최소 1 이상 나와야 함.

### P3 [main] — Ours는 shared atom GT 복원(`mgt_shared`)에서 two_recon 이상이다.

- **측정:** `ours/img_mgt_shared` + `ours/txt_mgt_shared` 평균 vs.
  `two_recon/img_mgt_shared` + `two_recon/txt_mgt_shared` 평균.
- **Pass:** $\alpha \in \{0.6, 0.9, 1.0\}$의 모든 cell에서 ours 평균이
  two_recon 평균보다 **+0.02 이상** 크다.
- **Fail:** 위 세 $\alpha$ 중 하나라도 +0.02 이하.
- **Failure scenario:** Stage 2의 diag-only loss는 latent 공간의 정렬만 건드리고
  decoder row와 GT atom 사이 geometry는 건드리지 않는다. 따라서 cos(dec, GT)를
  높이지 못하는 경우.
- **Intended interpretation:** 정렬 개선이 GT feature 복원도 개선한다 (decoder가
  permutation 기준으로 재학습되면서 shared atom을 더 뚜렷하게 분리).
- **Alternative explanations:**
  (a) Ours와 two_recon은 동일한 Stage 1 경로를 공유하므로 P3이 차이를
  보지 못하면 이는 "Stage 2는 MGT 향상에 무력" 을 의미. 실패해도 P1, P2는 살 수
  있음 — 이 경우 H의 recon+정렬 이득은 유지되나 feature-level 해석성 이득은
  기각.
  (b) MGT 임계 $\tau=0.8$은 angular ~36.9°로 상당히 느슨하므로 두 method 모두
  같은 값을 가질 수 있다. → P3 실패 시 §6에 완화 조건으로 기재.

### P4 [main] — Ours는 group_sparse / trace_align baseline 대비 정렬 우위.

- **측정:** $\alpha=1.0$에서 ours `cross_cos_top_mS_mean` vs. group_sparse,
  trace_align의 동일 metric.
- **Pass:** ours seed-mean − group_sparse seed-mean $\ge 0.10$ **AND** ours
  seed-mean − trace_align seed-mean $\ge 0.10$.
- **Fail:** 둘 중 하나라도 <0.10.
- **Failure scenario:** group-sparse의 joint-sparsity나 trace loss의 normalized
  cosine maximization이 이미 paired 위치의 latent를 강하게 정렬하므로 ours와
  차이가 없는 경우.
- **Intended interpretation:** permutation을 *학습 중*에 수행하는 설계가 L2,1 /
  trace 같은 *index-aligned* loss들보다 우수하다 (shared decoder가 학습 초기에
  index를 섞는 문제를 명시적으로 풀기 때문).
- **Alternative explanations:**
  (a) group_sparse/trace_align의 baseline $\lambda$ 값 (0.05, 1e-4)이 그 논문들의
  대형 실험에 튜닝된 것이라 synthetic scale에 안 맞을 수 있다. → §6에 한계로
  기재.
  (b) shared decoder (V=W) 는 애초에 두 modality의 embedding geometry가
  `Φ_S ≠ Ψ_S`일 때 표현력이 부족해서 정렬 이전에 recon 자체가 실패할 수 있다.
  Theorem 1은 이를 보였다. → P1·P4가 통과해도, 이 alternative는 부분적으로만
  배제됨 (scale matched).

### P5 [sub] — Stage 2 학습 이득은 permutation-only 이득을 초과한다.

- **측정:** 학습 데이터 기반 `diag_top_mS_diag_mean` (Stage 1 직후 permutation
  시점 값) vs. eval 기반 `cross_cos_top_mS_mean` (Stage 2 후 값).
- **Pass:** eval 후 `top_mS` 평균 $\ge$ Stage 1 permutation 시점 `top_mS`
  평균 − 0.05 (Stage 2가 유의하게 퇴화시키지 않음) **AND** eval 후 `rest`
  평균이 Stage 1의 대각 $[m_S+1:]$ 평균보다 낮다 (nulling이 유지됨).
- **Fail:** 위 두 조건 중 하나라도 위반.
- **Failure scenario:** Stage 2가 diag 1/0 분리 자체는 배치 기준으로만 유지하고
  eval에서 전체 대각이 0 근처로 붕괴 (활성화 자체가 사라지는 trivial optimum).
- **Label:** [sub]. 이 prediction은 "L_aux가 학습 가능한 유의미한 신호"인지
  확인하는 sanity check이며, 실패 시 P1·P2·P3이 어떻게 나왔든 H는 살 수 있지만
  Stage 2의 필요성을 주장할 수 없다.

### P6 [sub] — λ ablation에서 `top_mS`가 λ 상승에 대해 대체로 단조.

- **측정:** ablation λ sweep의 $\lambda \in \{2^{-4},\dots,2^{8}\}$ 에서
  `cross_cos_top_mS_mean` seed-mean.
- **Pass:** 7점 중 **적어도 5점에서** 상관이 양의 Spearman (rank corr ≥ 0.5).
  동시에 λ=2⁸과 λ=2⁻⁴ 사이 차이가 +0.05 이상.
- **Fail:** Spearman < 0.3 또는 양 끝점 차이 < 0.05.
- **Failure scenario:** $\lambda$가 작을 때 이미 saturation, 큰 λ에서도 정렬
  상승 없이 recon만 망가지는 plateau + crash 패턴.
- **Label:** [sub]. λ 민감도가 무시 가능할 수 있음 (작은 λ로도 충분) — 이는
  H 검증에 무해하나, 논문 recommendation에는 영향.

### P7 [sub] — m_S sweep에서 최적점이 GT oracle 근방.

- **측정:** ablation m_S sweep의 `cross_cos_top_mS_mean` 값, 각 $m_S \in
  \{256, 384, 512, 640, 768, 896, 1024\}$에서 seed-mean.
- **Pass:** argmax seed-mean이 $\{512, 640\}$ 중 하나 (oracle 주변).
- **Fail:** argmax가 extreme ($m_S=256$ 또는 1024).
- **Failure scenario:** top_mS metric이 자기 자신이 supplied `m_S`에 의해
  정의되므로 $m_S$를 크게 할수록 정의상 값이 **자동으로 감소**해야 한다 —
  그 경우 작은 $m_S$가 항상 이기는 trivial 패턴. 이 편향 해소를 위해 §4에서는
  `top_mS`가 아닌 **oracle 고정 기준** (항상 top-512) 으로 추가 측정해서
  같이 보고한다.
- **Label:** [sub]. `top_mS` 기준 자체의 편향이 있어 argmax 해석에 한계.

---

## §3. Experiment Design

### 3.1 Synthetic generative model (data)

- `SyntheticTheoryFeatureBuilder`
  (`src/datasets/synthetic_theory_feature.py`)
- Ground-truth latent:
  $n = n_{\mathrm I} + n_{\mathrm S} + n_{\mathrm T} = 256 + 512 + 256 = 1024$
- Embedding dim $d = 768$ (CLIP ViT-B 수준)
- Sparsity $s = 0.99$ (1% 활성), `min_active=1`
- `max_interference=0.1` (unmatched atom 간 내적 상한)
- Sample 수: train 50,000 / eval 10,000 / test 10,000
- $\alpha$-calibration:
  - $\alpha \in \{0.6, 0.9\}$ → `shared_mode="range"`, per-atom target cos
    $\alpha \pm 0.03$ 달성할 때까지 gradient descent (tol 0.01, 3000 iter)
  - $\alpha = 1.0$ → `shared_mode="identical"`, $\Phi_S = \Psi_S$
  - $\alpha = 0.3$ → `shared_mode="range"`, per-atom $0.27 \sim 0.33$
- 각 modality의 embedding: $x = \Phi_I z_I + \Phi_S z_S$,
  $y = \Psi_T z_T + \Psi_S z_S$

### 3.2 SAE configuration

- **공정성 규약 (Theorem 1과 동일):** 각 method에 동일한 총 latent budget
  $L=2048$을 배정한다.
  - `single_recon / group_sparse / trace_align` (shared encoder $V=W$):
    `TopKSAE(latent_size=2048)` 1개.
  - `two_recon / ours` (독립 2개 SAE): 각각
    `TopKSAE(latent_size=2048 // 2 = 1024)`, 합쳐서 2048.
- `hidden_size = 768`, `k = 16`, `normalize_decoder = True`.
- Decoder row unit-norm 유지: `set_decoder_norm_to_unit_norm()` 매 optimizer
  step 직후.
- Gradient projection: `remove_gradient_parallel_to_decoder_directions()` 매
  backward 직후 (`src/models/modeling_sae.py:472–501`).
- Optimizer: AdamW, `lr=2e-4`, `weight_decay=0`, grad clip off.

### 3.3 Five method definitions

| # | 이름 | Arch | per-SAE latent | Loss | 논문값 |
|---|---|---|---|---|---|
| 1 | `single_recon` | 1 SAE, shared $V=W$ | 2048 | recon only | — |
| 2 | `two_recon` | 2 SAEs $V, W$ (독립) | 1024 each | recon only | — |
| 3 | `group_sparse` | 1 SAE, shared $V=W$, paired | 2048 | recon + $\lambda \cdot L_{2,1}(\tilde z_I, \tilde z_T)$ | $\lambda = 0.05$ |
| 4 | `trace_align` | 1 SAE, shared $V=W$, paired | 2048 | recon + $\beta \cdot (-\mathrm{Tr}(\hat Z_I \hat Z_T^\top)/b)$ | $\beta = 10^{-4}$ |
| 5 | `ours` | 2 SAEs $V, W$ | 1024 each | Stage 1 recon → greedy permutation → Stage 2 recon + $\lambda \cdot L_\text{aux}(m_S)$ | $\lambda = 1$, $m_S = 512$, $k_\text{align}=4$ |

구현 참조:
- Method 1–4: `synthetic_theorem2_method.py` lines 130–265.
- Auxiliary losses 3, 4: `synthetic_theory_simplified.py:284–307`에서 import.
- Method 5: `synthetic_theorem2_method.py :: _train_ours` (lines 336–460).
- Greedy permutation: `_greedy_permutation_match_full` (lines 294–333) —
  $n \times n$ correlation 행렬에서 $k=0,\dots,n-1$ 단계마다 sub-행렬 argmax를
  top-left로 swap.

### 3.4 Sweeps

**Main (method comparison):**
- methods: 5개 모두
- $\alpha \in \{0.3, 0.6, 0.9, 1.0\}$
- seeds: 1, 2, 3 (3개)
- 총 training runs: $5 \times 4 \times 3 = 60$

**Ablation ($\alpha = 1.0$ 고정, method = ours):**
- λ sweep: $\lambda \in \{2^{-4}, 2^{-2}, 1, 4, 16, 64, 256\}$ (7점),
  $m_S=512$, $k_\text{align}=4$ 고정 → $7 \times 3 = 21$ runs.
- $m_S$ sweep: $m_S \in \{256, 384, 512, 640, 768, 896, 1024\}$
  (`n_shared`의 0.5×, 0.75×, 1×, 1.25×, 1.5×, 1.75×, 2×),
  $\lambda=1$, $k_\text{align}=4$ 고정 → 21 runs.
- $k_\text{align}$ sweep: $k \in \{2, 4, 6, 8\}$ (stage1 에폭 수;
  stage2 $= 10 - k$), $\lambda=1$, $m_S=512$ 고정 → $4 \times 3 = 12$ runs.
- 모든 ablation은 $\alpha = 1.0$ 한 점에서만 수행.

**총 runs (L=2048):** $60 + 21 + 21 + 12 = 114$ training runs.

### 3.5 Metrics (definition + source location)

- `avg_eval_loss` = (img+txt) / 2, normalized recon loss
  (`_eval_loss_single_modality` in `synthetic_theorem1_unified.py:135–156`).
- `cross_cos_top_mS_mean`, `cross_cos_rest_mean`
  (`_cross_corr_mean_parts` in `synthetic_theorem2_method.py:512–530`,
  paired signed Pearson 후 상위 $m_S$ / 나머지 대각 평균).
- `img_mgt_shared`, `txt_mgt_shared` — cosine 기반 GT recovery, $\tau = 0.8$
  (`_compute_recovery_metrics` in `synthetic_sae_theory_experiment.py:63–79`).
- Ours diagnostics: `m_S_hat_rho`, `first_drop_below_rho`, `top_mS_diag_mean`,
  `ordered_diag_head` (`_train_ours` lines 402–414).

### 3.6 Hypothesis scope vs Experiment scope

| | |
|---|---|
| Hypothesis 원래 scope | 실제 VL 모델의 image / text encoder embedding 위에서 학습한 SAE |
| Experiment scope | Linear sparse generative model, identical z_S co-activation, matched sparsity/magnitude, $d=768$, $k=16$, paired batch, 10 epochs, 3 seeds, $\alpha \in \{0.3, 0.6, 0.9, 1.0\}$, latent size $m=2048$ |
| **Not guaranteed in scope** | 실제 VLM embedding이 linear sparse generative model을 따르는지, 곡률 있는 manifold에서의 동작, z_I·z_S·z_T 간 co-activation 상관, magnitude 비대칭, encoder 초기 정렬 |
| **이 보고서에서 주장 불가** | 전체 VLM alignment 문제 해결 / 알고리즘 1의 일반 우위 / SAE 해석성 전반 개선 |

### 3.7 Reproducibility

- Code: `synthetic_theorem2_method.py`, commit `a8f66bc` 이후.
- 실행: `DEVICE=cuda STAGE=main bash scripts/synthetic_theorem2_method.sh` +
  `DEVICE=cuda STAGE=ablation bash scripts/synthetic_theorem2_method.sh`.
- 결과 JSON: `outputs/synthetic_theorem2/runs/method_ns512_k16_*_<ts>/result.json`.
- Seeds: 1, 2, 3 (`--seed-base 1 --num-seeds 3`). 5개 미만이므로 §6에 한계로
  기재.

---

## §4. Results

*(main + ablation 완료 후 작성)*

### 4.1 Main comparison table (`avg_eval_loss`)

| α | single_recon | two_recon | group_sparse | trace_align | ours |
|---|---|---|---|---|---|
| 0.3 | | | | | |
| 0.6 | | | | | |
| 0.9 | | | | | |
| 1.0 | | | | | |

### 4.2 Main comparison table (`cross_cos_top_mS_mean`, $m_S=512$)

| α | single_recon | two_recon | group_sparse | trace_align | ours |
|---|---|---|---|---|---|
| 0.3 | | | | | |
| 0.6 | | | | | |
| 0.9 | | | | | |
| 1.0 | | | | | |

### 4.3 Main comparison table (`img_mgt_shared` + `txt_mgt_shared`) / 2

| α | single_recon | two_recon | group_sparse | trace_align | ours |
|---|---|---|---|---|---|
| 0.3 | | | | | |
| 0.6 | | | | | |
| 0.9 | | | | | |
| 1.0 | | | | | |

### 4.4 Ablation: λ sweep

### 4.5 Ablation: m_S sweep

### 4.6 Ablation: k_align sweep

### 4.7 Pass/fail verdict per prediction

- **P1 [main]**: _(TBD)_
- **P2 [main]**: _(TBD)_
- **P3 [main]**: _(TBD)_
- **P4 [main]**: _(TBD)_
- **P5 [sub]**: _(TBD)_
- **P6 [sub]**: _(TBD)_
- **P7 [sub]**: _(TBD)_

---

## §5. Verdict

*(결과 확인 후 작성)*

> This experiment is an existence proof. The conclusions hold only within the
> assumptions stated above.

---

## §6. Limitations and Scope

### 6.1 Strong assumptions of the experiment setting

- **Paired co-activation:** 모든 paired 샘플에서 image와 text가 *동일한*
  $z_S$를 공유한다. 실제 VLM에서는 image와 text가 같은 concept에 대해 서로 다른
  크기/활성 패턴을 가질 수 있다.
- **Linear sparse combination:** $x = \Phi z$ (곡률/비선형 없음).
- **Matched sparsity and magnitude:** 두 modality가 같은 $s$, 같은 top-k,
  같은 magnitude 스케일. Asymmetry는 모델링되지 않음.
- **Decoder unit-norm 상수화:** SAE의 decoder row norm을 매 step 1로 강제.
  실제 학습에서는 norm이 drift할 수 있음.
- **Paired batch 256 (drop_last=True):** Stage 2 배치 단위 Pearson이 $B=256$에
  의존. 더 작은 batch에서는 $L_\text{aux}$ 분산이 커질 수 있음.

### 6.2 Limitations of the metric

- `cross_cos_top_mS_mean`은 `m_S` 자체가 정의에 포함되어, 작은 `m_S`로 갈수록
  자동으로 값이 커지는 편향이 있다. **m_S ablation (P7)** 은 이 편향에 영향을
  받으므로 argmax 해석에 주의. §4에서는 보조로 `m_S_supplied`와 무관한 oracle
  기준 (항상 top-512)으로도 보고한다.
- `img_mgt_shared`의 $\tau = 0.8$은 angular ~36.9°의 느슨한 임계. 두 method가
  같은 bin에 들어가는 경우 해상도가 부족할 수 있다.
- Ours의 `cross_cos_*`는 permutation을 적용한 `ours` 파라미터에서 계산되므로,
  baseline과 비교할 때 **baseline 쪽에도 동일 oracle `m_S=512` 기준**을
  사용한다 — 이 공정성은 보고서 §4에서 명시적으로 검증한다.

### 6.3 Extrapolation / generalization scope

- 실제 VLM encoder embedding에서 같은 결과가 재현되는지는 본 실험으로 답할 수
  없다.
- latent_size $m = 2048$ 한 점만 — 더 큰 / 작은 latent 공간에서의 동작은 별도
  보고서 (`report_L4096.md`, `report_L8192.md`) 에서 다룬다.
- $\alpha \in \{0.3, 0.6, 0.9, 1.0\}$ — 중간값 ($\alpha = 0.5, 0.8$ 등) 이나
  $\alpha < 0.3$은 다루지 않음.
- seeds = 3 은 `.claude/skills/experiment-report.md §7`의 "≥ 5 seeds" 권장보다
  적음. 본 보고서의 표준편차는 작은 seed 수 하에서의 추정치이며, 5+ seed
  재검증이 필요하다.

### 6.4 Un-ruled-out alternative mechanisms

아래는 §2에서 열거된 대안 중 본 실험만으로는 배제할 수 없는 것들이다.

- **Per-SAE 표현력 trade-off** — two_recon/ours는 per-SAE latent=1024로 shared-
  decoder baseline의 절반이다. 총 파라미터 수는 같지만, 각 SAE가 한 modality의
  (private + shared) feature를 모두 담아야 해서 capacity 압박이 발생한다.
  이 trade-off가 결과에 미치는 영향은 L ∈ {2048, 4096, 8192} 비교로 따로 평가.
- **`m_S=512` oracle 정보 유출** — ablation은 측정하지만, oracle 없이 ρ-기반
  auto-detection 만 쓸 때의 성능은 별도 실험 필요.
- **Baseline $\lambda/\beta$ 튜닝 부족** — 각 논문 값은 CLIP-scale에 튜닝된 값.
  synthetic scale에 맞춘 재튜닝은 수행하지 않았다.
- **Encoder 초기 정렬** — 두 SAE를 독립 seed로 초기화한 것의 영향 (모든 method가
  같은 seed 배치를 쓰지만, two_recon과 ours는 동일 Stage 1 경로를 공유).
- **m_S sweep 상한 degeneracy** — `m_S=1024`는 per-SAE latent 전체와 같아
  `rest` 영역이 빈 슬라이스가 된다. aggregator는 NaN을 drop하므로 이 셀은
  `rest` metric이 결측치로 남는다.

이 대안들은 본 실험에서 직접 측정되지 않았으며, 후속 실험 항목이다.
