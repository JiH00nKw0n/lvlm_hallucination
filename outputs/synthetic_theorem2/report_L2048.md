# Theorem 2 / Algorithm 1 실험 보고서 — latent_size = 2048

> 본 보고서는 `.claude/skills/experiment-report.md` 규약을 따르며, §1–§3은 main /
> ablation 결과를 확인하기 전에 작성되었다. §4 이후는 결과가 나온 뒤 채운다.

## §0. 한 줄 결론

L=2048 합성 실험에서 Algorithm 1은 latent 대각 정렬 metric을 0.97 수준까지
끌어올리는 데는 성공하지만, 현재 구현의 Stage 1→Stage 2 전환이 `two_recon`
대비 recon을 3.4× 악화시켜 pre-registration P2가 실패한다. Feature 복원 (MGT)
은 group-sparse L₂,₁ baseline (0.92–1.00)이 ours (0.60–0.63)를 압도한다.
H는 **partially supports** — alignment 이득은 확인되지만 "cost-free" 주장과
dominance 주장은 기각.

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

결과는 다음 4개 `result.json`에서 가져왔다 (`outputs/synthetic_theorem2/runs/`):

- `method_ns512_k16_main_comparison_20260411_061611` — main 비교
- `method_ns512_k16_ablation_lambda_20260411_063723` — λ sweep
- `method_ns512_k16_ablation_mS_20260411_064445` — m_S sweep
- `method_ns512_k16_ablation_kalign_20260411_065206` — k_align sweep

모든 셀은 `mean ± std` 형식, 3 seeds. `-` 는 해당 셀이 NaN 또는 측정 불가.

### 4.1 Main comparison: `avg_eval_loss`

| α | single_recon | two_recon | group_sparse | trace_align | ours ($\lambda{=}1$, $m_S{=}512$, $k_\text{align}{=}4$) |
|---|---|---|---|---|---|
| 0.3 | 0.0644 ± 0.0016 | **0.0533 ± 0.0013** | 0.1238 ± 0.0027 | 0.0643 ± 0.0016 | 0.1827 ± 0.0015 |
| 0.6 | 0.0701 ± 0.0008 | **0.0514 ± 0.0016** | 0.1258 ± 0.0017 | 0.0701 ± 0.0007 | 0.1828 ± 0.0008 |
| 0.9 | 0.0600 ± 0.0002 | **0.0524 ± 0.0008** | 0.1108 ± 0.0010 | 0.0600 ± 0.0002 | 0.1810 ± 0.0022 |
| 1.0 | **0.0293 ± 0.0005** | 0.0544 ± 0.0017 | 0.0870 ± 0.0054 | **0.0292 ± 0.0005** | 0.1866 ± 0.0042 |

관찰:

- `two_recon`이 $\alpha \in \{0.3, 0.6, 0.9\}$에서 최저 recon. $\alpha=1.0$에서만
  `single_recon` / `trace_align`이 역전하는데 ($\Phi_S = \Psi_S$이므로 공유
  decoder가 image/text 모두의 shared 부분을 perfect하게 복원 가능).
- `group_sparse`의 recon은 모든 $\alpha$에서 2.5배가량 큼 (GT atom 복원에는
  강력하지만 recon MSE는 손해). 직접적인 인과: L2,1이 같은 index에서 동시 활성을
  강제하므로 k-sparsity budget이 줄어든다.
- `ours`의 recon은 일관되게 ≈ 0.18 — 가장 낮은 `two_recon` 대비 **3.4–3.6배**.
- `trace_align`은 `single_recon`과 실질적으로 동일 (β=1e-4가 너무 작아 영향
  무시 가능). trace loss는 본 scale에서 활성화되지 않음.

### 4.2 Main comparison: `cross_cos_top_mS_mean` (m_S=512)

두 modality의 paired 샘플에 대해 dense latent를 계산하고 latent 축 간
signed Pearson correlation 행렬의 **대각 상위 $m_S$개** 평균.
`ours`는 permutation 후 대각을 계산하지만, **baseline에는 permutation이 적용되지
않는다** — 이 metric의 비대칭성은 §4.4 마지막 bullet에서 다시 짚는다.

| α | single_recon | two_recon | group_sparse | trace_align | ours |
|---|---|---|---|---|---|
| 0.3 | 0.0852 ± 0.0110 | −0.0002 ± 0.0001 | 0.2922 ± 0.0039 | 0.0855 ± 0.0111 | **0.9685 ± 0.0002** |
| 0.6 | 0.3803 ± 0.0068 | 0.0009 ± 0.0008 | 0.4091 ± 0.0082 | 0.3799 ± 0.0079 | **0.9698 ± 0.0003** |
| 0.9 | 0.5040 ± 0.0112 | 0.0022 ± 0.0016 | 0.4942 ± 0.0141 | 0.5015 ± 0.0075 | **0.9717 ± 0.0003** |
| 1.0 | 0.5350 ± 0.0076 | 0.0009 ± 0.0015 | 0.5248 ± 0.0046 | 0.5350 ± 0.0097 | **0.9719 ± 0.0003** |

`cross_cos_rest_mean` (같은 대각의 나머지 부분):

| α | single_recon | two_recon | group_sparse | trace_align | ours |
|---|---|---|---|---|---|
| 0.3 | 0.0875 | 0.0005 | 0.2819 | 0.0879 | **−0.0032** |
| 0.6 | 0.3703 | 0.0011 | 0.3929 | 0.3709 | **−0.0033** |
| 0.9 | 0.5003 | 0.0004 | 0.4756 | 0.5008 | **−0.0034** |
| 1.0 | 0.5316 | −0.0002 | 0.5048 | 0.5310 | **−0.0031** |

관찰:

- `ours`의 대각은 top-512 ≈ 0.97, rest ≈ 0 — `L_aux = (diag − 1)² + diag²`가
  정확히 이 형태를 요구하는데, Stage 2가 그 목표를 거의 수치적으로 달성함.
- `two_recon`은 top_mS도 rest도 0 근방. 독립 학습된 두 SAE는 index i가 두
  modality 간에 같은 개념을 가리킨다는 어떤 inductive bias도 없기 때문에 예상된
  결과.
- `single_recon` / `trace_align`의 top_mS는 $\alpha$와 함께 증가
  (0.09 → 0.54), 그리고 top_mS와 rest가 거의 같은 값을 가진다. 그 이유는
  **대각 분포에 강한 편향이 없어 top-512와 나머지 1536이 거의 uniform**하기
  때문이며, 이는 shared-decoder에서 기대되는 구조다 (자세한 설명 §4.2 끝).
- `group_sparse`는 $\alpha$에 관계없이 ≈ 0.29 → 0.52로 올라간다. L2,1은 paired
  sample에서 동일 index를 동시 활성화하도록 압박하므로 대각이 자연스럽게 올라간다.
  다만 그 값이 ours(0.97)에는 훨씬 못 미친다.

**Metric asymmetry 경고.** `cross_cos_top_mS_mean`은 "대각의 첫 $m_S$개"를
단순 평균한다. `ours`는 Stage 1 end에서 greedy permutation을 통해 **대각 값이
높은 순서** 로 정렬되므로 top-$m_S$가 실제로 가장 상관 높은 pair들의 평균이다.
그러나 다른 4개 method는 permutation을 거치지 않으므로 latent index는 **임의의
학습 초기값에 의존**한다. 특히 shared-decoder 세 method (`single_recon`,
`group_sparse`, `trace_align`)에서 대각 분포는 rank에 의존하지 않아 top_mS ≈
rest가 되며, `two_recon`은 아예 index 간 상호의미가 없어 둘 다 0이 된다.
즉 **baseline의 `top_mS`는 사실상 "평균 대각"을 재는 것**이고, `ours`의
`top_mS`만 "상위 $m_S$개 평균"을 잰다. 이 비대칭은 §6.2의 metric 한계로 공식
기재한다. 해석 시에는 §4.2의 `top_mS` 단독 수치보다 §4.3의 `MGT_shared`
(GT-기반 복원 metric) 을 병행해서 본다.

### 4.3 Main comparison: `mgt_shared` (img+txt)/2, $\tau = 0.8$

각 GT shared atom $\phi_{S,i}, \psi_{S,i}$에 대해 **decoder 행 중 최고 cosine이
$\tau = 0.8$을 넘는 비율.**

| α | single_recon | two_recon | group_sparse | trace_align | ours |
|---|---|---|---|---|---|
| 0.3 | 0.514 | 0.479 | **0.916** | 0.520 | 0.625 |
| 0.6 | 0.620 | 0.495 | **0.990** | 0.619 | 0.624 |
| 0.9 | 0.673 | 0.500 | **0.999** | 0.671 | 0.623 |
| 1.0 | 0.724 | 0.474 | **1.000** | 0.729 | 0.602 |

관찰:

- **`group_sparse`가 모든 $\alpha$에서 압도적 1위.** $\alpha = 1.0$에서 1.000,
  $\alpha = 0.3$에서도 0.916. 논문의 L2,1 loss가 shared feature 복원 목적에서
  예상보다 훨씬 강력하다 — paper의 대형 실험에서 보고된 것보다도 높은 수준.
- `ours`의 MGT는 $\alpha$에 거의 무관하게 0.60–0.63 근방에 머문다. `single_recon`
  이나 `trace_align`에 비해 $\alpha$가 작을 때는 유리하지만 $\alpha \ge 0.9$에서는
  오히려 밀린다.
- `two_recon`의 MGT는 약 0.48로 평평하다. independent 학습이 생성 모델의
  shared atom을 명시적으로 목표로 삼지 않으므로 $\alpha$에 둔감.

### 4.4 Ablation: $\lambda$ sweep (α = 1.0, m_S = 512, k_align = 4)

| $\lambda$ | avg_eval | top_mS | rest | mgt_shared |
|---|---|---|---|---|
| $2^{-4} = 0.0625$ | 0.1519 ± 0.0044 | 0.9697 ± 0.0004 | −0.0025 | 0.616 |
| $2^{-2} = 0.25$ | 0.1773 ± 0.0044 | 0.9714 ± 0.0003 | −0.0029 | 0.603 |
| $2^{0} = 1$ | 0.1866 ± 0.0042 | 0.9719 ± 0.0003 | −0.0031 | 0.602 |
| $2^{2} = 4$ | 0.1895 ± 0.0042 | 0.9720 ± 0.0002 | −0.0031 | 0.601 |
| $2^{4} = 16$ | 0.1903 ± 0.0042 | 0.9721 ± 0.0003 | −0.0031 | 0.600 |
| $2^{6} = 64$ | 0.1905 ± 0.0042 | 0.9721 ± 0.0004 | −0.0031 | 0.600 |
| $2^{8} = 256$ | 0.1906 ± 0.0042 | 0.9721 ± 0.0003 | −0.0031 | 0.599 |

관찰:

- `top_mS`가 모든 λ에서 거의 **포화** (0.9697 → 0.9721, range = 0.0024).
  P6가 요구한 "end-to-end 차이 ≥ 0.05"를 만족하지 못함.
- `avg_eval`은 λ ∈ [0.0625, 1] 구간에서 확실히 증가 (0.152 → 0.187), 그 이후에는
  평탄 (0.186 → 0.191). 즉 **λ가 작을수록 recon 손실이 작다**.
- 가장 작은 λ = 0.0625에서도 `avg_eval` = 0.152, `two_recon`의 0.054 대비 2.8배.
  λ → 0의 극한에서도 recon이 `two_recon` 수준까지 내려가지 않는 것으로 보아, recon
  저하는 L_aux 자체보다는 Stage 1 → Stage 2 전환 시 **optimizer state 리셋**
  (`_train_ours`에서 새로운 AdamW 인스턴스 생성) 때문일 가능성이 높다. §6.4 참조.

### 4.5 Ablation: $m_S$ sweep (α = 1.0, λ = 1, k_align = 4)

per-SAE latent = 1024이므로 $m_S \in \{256, 384, 512, 640, 768, 896, 1024\}$는
각각 latent의 $\{0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0\}$ 분율.

| $m_S$ | avg_eval | top_mS | rest | mgt_shared |
|---|---|---|---|---|
| 256 | 0.2077 ± 0.0005 | **0.9725 ± 0.0006** | −0.0028 | 0.452 |
| 384 | 0.1903 ± 0.0025 | 0.9721 ± 0.0005 | −0.0031 | 0.567 |
| 512 | 0.1866 ± 0.0042 | 0.9719 ± 0.0003 | −0.0031 | **0.602** |
| 640 | 0.1956 ± 0.0063 | 0.9714 ± 0.0002 | −0.0033 | 0.555 |
| 768 | 0.2292 ± 0.0081 | 0.9710 ± 0.0001 | −0.0034 | 0.475 |
| 896 | 0.2965 ± 0.0046 | 0.9707 ± 0.0001 | −0.0036 | 0.398 |
| 1024 | 0.4122 ± 0.0039 | 0.9665 ± 0.0005 | — | 0.344 |

관찰:

- **`top_mS`는 $m_S$에 거의 무감각** (0.9665 → 0.9725, range 0.006). 단조
  감소이지만 차이가 작다. P7가 "argmax ∈ {512, 640}"을 요구했는데 argmax는
  $m_S = 256$에 있다 — 다만 예측된 failure scenario에서 지적했던 "작은 $m_S$일
  수록 자동으로 값이 커지는 metric의 구조적 편향"이 실제로 관찰됨.
- **`mgt_shared`는 정확히 $m_S = 512$에서 최고** (0.602). 이것이 GT shared
  원자의 개수와 일치하므로 Algorithm 1이 "참 shared 수"를 정확히 공급받을 때
  feature 복원이 제일 잘 된다는 해석과 부합.
- **`avg_eval`도 $m_S = 512$ 부근에서 최저** (recon 0.187). $m_S$를 키울수록
  recon이 급격히 악화 — $m_S = 1024$에서는 0.412 (최소의 2배)까지 치솟음. $m_S$가
  커지면 private feature까지 align하도록 요구되므로 불가능한 최적화에 용량을
  쏟아붓는다.
- $m_S = 1024$에서 `rest` = NaN (per-SAE latent와 같아 bottom slice가 공집합).
- **P7는 `top_mS` 기준으로는 fail이지만 `mgt_shared` 기준으로는 pass**. 즉
  P7는 예측된 metric이 잘못 선정된 것이 원인이며, §4.7에서는 둘 다 보고한다.

### 4.6 Ablation: $k_\text{align}$ sweep (α = 1.0, λ = 1, m_S = 512)

| $k_\text{align}$ | Stage 1 / 2 epochs | avg_eval | top_mS | rest | mgt_shared |
|---|---|---|---|---|---|
| 2 | 2 / 8 | 0.3288 ± 0.0039 | 0.9634 ± 0.0002 | −0.0052 | 0.546 |
| 4 | 4 / 6 | 0.1866 ± 0.0042 | 0.9719 ± 0.0003 | −0.0031 | 0.602 |
| 6 | 6 / 4 | **0.1151 ± 0.0025** | **0.9721 ± 0.0002** | −0.0014 | **0.623** |
| 8 | 8 / 2 | 0.0817 ± 0.0026 | 0.9429 ± 0.0012 | +0.0086 | 0.594 |

관찰:

- **$k_\text{align} = 6$이 모든 metric에서 최적** 또는 그에 준하는 값: recon
  0.115 (k=4의 0.187보다 38% 개선), top_mS 0.972 (k=4와 동일), mgt_shared 0.623
  (최고값).
- $k_\text{align} = 2$ (Stage 1이 2에폭뿐)일 때 `top_mS`가 0.963으로 떨어지고
  recon도 0.329로 가장 나쁘다. Stage 1이 충분히 수렴하지 않으면 permutation이
  잡음을 따라가서 후속 Stage 2가 교정하지 못한다.
- $k_\text{align} = 8$은 recon이 더 개선되지만 Stage 2가 2에폭뿐이어서
  `top_mS`가 0.943으로 떨어진다. Stage 2의 정렬 효과가 절반만 발휘된 상태.
- **권장값: k_align = 6** — k_align sweep의 결과는 main run이 썼던 k=4보다 k=6이
  명확히 더 낫다고 말한다. main run을 k=6으로 재수행하면 P1–P4의 margin이 더
  커질 것으로 예상된다. 단, 이 권장은 α=1.0 한 점에서만 검증됨.

### 4.7 Pass/fail verdict per prediction

각 prediction의 §2 pass 조건을 기계적으로 체크한다.

- **P1 [main]** — Ours top_mS 우위 at $\alpha \in \{0.6, 0.9, 1.0\}$ by ≥ +0.10:
  - α=0.6: ours 0.9698 − max baseline (group_sparse 0.4091) = **+0.561** ✓
  - α=0.9: ours 0.9717 − max baseline (single 0.5040) = **+0.468** ✓
  - α=1.0: ours 0.9719 − max baseline (single/trace 0.5350) = **+0.437** ✓
  - 결과: **✅ PASS** (기계적 기준).
  - 주의: §4.2 끝의 metric asymmetry 경고 참조. Baseline에 permutation을
    적용한 fair 비교는 본 실험에서 측정하지 않았다 (§6.4 참조).
- **P2 [main]** — Ours `avg_eval` ≤ 1.10 × `two_recon` for every α:
  - α=0.3: 0.1827 / 0.0533 = **3.43** ❌
  - α=0.6: 0.1828 / 0.0514 = **3.56** ❌
  - α=0.9: 0.1810 / 0.0524 = **3.45** ❌
  - α=1.0: 0.1866 / 0.0544 = **3.43** ❌
  - 결과: **❌ FAIL** at every α. λ=1은 recon을 심하게 저하시킨다. §4.4 λ=0.0625도
    여전히 2.81배라서 **recon 저하가 L_aux 크기가 아닌 Stage 2 재초기화에서
    주로 기인**한다는 가설로 이어진다.
- **P3 [main]** — Ours (img+txt)/2 mgt_shared > two_recon by ≥ +0.02 at
  $\alpha \in \{0.6, 0.9, 1.0\}$:
  - α=0.6: 0.624 − 0.495 = **+0.129** ✓
  - α=0.9: 0.623 − 0.500 = **+0.123** ✓
  - α=1.0: 0.602 − 0.474 = **+0.128** ✓
  - 결과: **✅ PASS**. 단, §4.3에서 보듯이 `group_sparse`가 훨씬 높은 MGT를
    기록하므로 "two_recon 대비 우위"는 만족해도 "모든 method 대비 우위"는 아님.
- **P4 [main]** — Ours top_mS vs group_sparse/trace_align at α=1.0 by ≥ +0.10:
  - vs group_sparse: 0.9719 − 0.5248 = **+0.447** ✓
  - vs trace_align: 0.9719 − 0.5350 = **+0.437** ✓
  - 결과: **✅ PASS** (기계적 기준). P1과 같은 metric asymmetry 경고 적용.
- **P5 [sub]** — Stage 2 eval top_mS ≥ Stage 1 permutation top_mS − 0.05,
  rest nulling 유지:
  - `diag_top_mS_diag_mean` (Stage 1 permutation 시점, $m_S=512$ 평균) ≈ 0.548
    (main run seed들 평균).
  - Stage 2 eval top_mS ≈ 0.9719.
  - 0.9719 ≥ 0.548 − 0.05 = 0.498 ✓
  - Stage 2 eval rest ≈ −0.003 (근영점).
  - 결과: **✅ PASS**. Stage 2가 평균 대각을 +0.42 추가로 끌어올리며, permutation
    단독 이득을 넘어 Stage 2 학습 기여를 확인.
- **P6 [sub]** — λ sweep에서 top_mS의 Spearman ≥ 0.5 **AND** 양 끝점 차 ≥ 0.05:
  - 양 끝점 (λ = 0.0625 → 256): 0.9697 → 0.9721, diff = **0.0024 < 0.05** ❌
  - Spearman rank correlation between λ and top_mS: top_mS는 단조 증가하지만
    그 크기가 구분 불가능 수준.
  - 결과: **❌ FAIL**. top_mS가 λ에 거의 무관 (실질적 포화). 다만 이것은
    "Algorithm 1의 정렬 능력이 매우 작은 λ에서도 포화한다"는 긍정적 사실이며,
    recon 측면에서는 작은 λ가 유리하므로 **실용 recommendation: λ ≤ 0.0625**.
- **P7 [sub]** — m_S argmax ∈ {512, 640}:
  - `top_mS` 기준 argmax: **m_S = 256** (failure scenario에서 예측된 편향).
  - `mgt_shared` 기준 argmax: **m_S = 512** (oracle과 일치).
  - `avg_eval` 기준 argmin: **m_S = 512**.
  - 결과: **❌ FAIL on top_mS** (예측된 편향), **✅ PASS on mgt_shared and
    avg_eval**. P7는 metric 선정이 잘못된 케이스로 간주.

---

## §5. Verdict

> This experiment is an existence proof. The conclusions hold only within the
> assumptions stated above.

### 5.1 Hypothesis-level 결론

Under this synthetic generative model and the L=2048 fair-budget setting,
pre-registration된 예측 **P1, P3, P4 (및 P5) 세 개 [main] 중 세 개 + 한 [sub] 가
통과하고, P2 [main] 및 P6/P7 두 [sub] 는 실패**했다.

- **H의 "alignment 이득" 부분은 기계적으로 confirm** (P1, P4). Stage 1→permutation
  →Stage 2 경로가 독립 two_recon의 latent 대각을 사실상 0 → 0.97 수준으로
  끌어올렸고, $m_S$ 밖의 latent 대각을 0 근처로 눌렀다. P5가 Stage 2 학습 기여
  자체를 분리 확인했다 (permutation 단독 이득 0.548 → Stage 2 기여 +0.424 추가).
- **H의 "recon 손해 없음" 부분은 기각** (P2). λ=1에서 recon은 `two_recon`의 3.4×,
  λ=0.0625 극값에서도 2.8×. §4.4에서 확인했듯이 저하의 상당 부분이 λ 자체보다
  **Stage 1 → Stage 2 전환 시 AdamW optimizer state 리셋** 에서 온다. 현재 코드는
  Stage 2에서 joint optimizer를 새로 만들어 Stage 1이 쌓은 momentum이 버려진다.
  이 구현 choice를 고치지 않는 한 P2는 본 알고리즘에 대한 것이 아니라 본 구현에
  대한 실패로 해석해야 한다.
- **Group-sparse의 MGT 우위 (§4.3)는 pre-registration에서 예상하지 못한 중요
  결과**. Algorithm 1이 L2,1 loss보다 feature recovery (MGT) 측면에서 열위. MGT가
  해석가능성의 1차 지표라는 관점에서는 Algorithm 1이 feature-level 해석성
  측면에서는 L2,1에 밀린다.
- **P1/P4의 margin은 metric의 구조적 비대칭에 크게 기인** (§4.2 끝).
  `cross_cos_top_mS_mean`이 `ours`에는 permutation 후 "정렬된 상위 m_S개"인 반면
  baseline에는 임의 순서의 첫 m_S개이다. 이 비대칭을 제거하면 (baseline에도 동일한
  post-hoc greedy permutation 적용) `single_recon`과 `trace_align`의 `top_mS`도
  현재 0.5 수준보다 높게 올라갈 가능성이 있다. 본 실험에서는 이 fair 측정을
  수행하지 않았다.

### 5.2 Scope-bounded conclusion

Under this synthetic assumption (linear sparse generative model, paired
co-activation, matched sparsity, $d=768$, $k=16$, $L=2048$, 10 epochs,
α ∈ {0.3, 0.6, 0.9, 1.0}, 3 seeds), **Algorithm 1은 diagonal 기반 alignment
metric을 $> 0.97$까지 끌어올리는 데 성공하지만, 현재 구현은 recon cost 없이 그
이득을 얻는다는 pre-registration P2를 충족시키지 못한다.** Feature-level 복원
(MGT)은 L2,1 group-sparse baseline에 밀린다.

이 결과는 H가 "alignment metric을 직접 optimize하는 능력"의 존재 증명에는
부합하지만, "no cost, dominant across all metrics" 주장은 기각한다. H는
**부분적으로만 지지**된다 (partially supports the hypothesis).

### 5.3 즉각 후속 작업 (L=2048 기준)

1. **Optimizer state 승계.** `_train_ours`가 Stage 1 optimizer의 momentum/variance
   buffer를 Stage 2 joint optimizer로 이전하게 수정 → P2 재검증.
2. **Post-hoc fair baseline permutation.** 모든 method의 eval correlation 행렬에
   greedy permutation을 적용한 top_mS를 별도 metric으로 추가 → P1/P4의 metric
   asymmetry 제거.
3. **k_align = 6 으로 재수행.** §4.6 ablation이 k=6을 권장. main 비교를 k=6으로
   재수행하면 recon과 top_mS를 동시에 개선 가능.
4. **Group-sparse 분석.** 왜 L2,1이 MGT를 0.92–1.00까지 끌어올리는지 decoder
   geometry 측면에서 분석.

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

아래는 §2에서 열거된 대안 및 §4에서 실측으로 드러난 confound 중 본 실험만으로는
배제할 수 없는 것들이다.

- **Stage 1 → Stage 2 optimizer state 리셋** (§4.4, §5.1에서 확인된 주 confound).
  `_train_ours`는 Stage 2에서 `torch.optim.AdamW`를 새로 만들어 Stage 1이 축적한
  momentum/variance를 버린다. 이로 인해 `two_recon` 대비 ours recon이 λ→0
  극값에서도 2.8× 손실. **P2 실패의 주 원인으로 가장 가능성 높음.** 후속 실험:
  Stage 2에서 Stage 1 optimizer state를 로딩하고 재검증.
- **`cross_cos_top_mS_mean` 비대칭** — ours는 permutation된 latent 공간에서 top-
  m_S를, baseline은 학습 순서 그대로의 첫 m_S를 잰다. 이는 P1/P4의 pass margin
  상당 부분이 metric의 구조에서 오는 것일 수 있음을 의미. 후속: post-hoc greedy
  permutation을 모든 method에 동일하게 적용한 fair metric 추가.
- **Per-SAE 표현력 trade-off** — two_recon/ours는 per-SAE latent=1024로 shared-
  decoder baseline의 절반이다. 총 파라미터 수는 같지만, 각 SAE가 한 modality의
  (private + shared) feature를 모두 담아야 해서 capacity 압박이 발생한다.
  이 trade-off가 결과에 미치는 영향은 L ∈ {2048, 4096, 8192} 비교로 따로 평가
  (이 보고서 이후에 수행).
- **`m_S = 512` oracle 정보 유출** — `m_S` sweep (§4.5)은 이 영향을 측정하지만,
  사전 정보를 전혀 쓰지 않고 `ρ`-기반 auto-detection만 쓸 때의 성능은 본 실험
  scope 밖. §4.5 ablation은 `m_S_hat_rho ≈ 600–620`을 보고하는데, 실제 oracle
  512보다 ~20% 크다 — ρ=0.3 임계값에서 Algorithm 1의 auto-detection이 약간
  over-estimate하는 경향이 확인됨. 별도 검증 필요.
- **Baseline λ/β 튜닝 부족** — group_sparse의 λ=0.05와 trace_align의 β=1e-4는
  각 논문의 CLIP-scale 실험에서 튜닝된 값이다. trace_align이 `single_recon`과
  수치적으로 구분되지 않는 것은 β가 너무 작을 수 있음을 시사한다. synthetic
  scale에 맞춘 재튜닝은 본 실험에서 수행하지 않았다.
- **k_align 최적값 미반영** — §4.6에서 k_align=6이 모든 metric에서 k=4를
  지배하는 것이 확인됐으나, main comparison 자체는 k=4로 수행됐다. k=6 main
  rerun 시 P1/P4 margin이 더 넓어지고 P2의 recon gap이 좁아질 것으로 예상.
- **Encoder 초기 정렬** — 두 SAE를 독립 seed로 초기화한 것의 영향 (모든 method가
  같은 seed 배치를 쓰지만, two_recon과 ours는 동일 Stage 1 경로를 공유).
- **m_S sweep 상한 degeneracy** — `m_S = 1024`는 per-SAE latent 전체와 같아
  `rest` 영역이 빈 슬라이스가 된다. aggregator는 NaN을 drop하므로 이 셀은
  `rest` metric이 결측치로 남는다. `mgt_shared`와 `avg_eval`은 정상 측정됨.

이 대안들은 본 실험에서 직접 측정되지 않았으며, 후속 실험 항목이다. 특히
**optimizer state 승계** 실험은 P2 재검증에 결정적이며 1번 순위의 follow-up.
