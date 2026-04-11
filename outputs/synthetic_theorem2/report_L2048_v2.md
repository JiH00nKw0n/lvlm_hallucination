# Theorem 2 / Algorithm 1 실험 보고서 — latent_size = 2048 (**v2**: k_align=6, multi-τ MGT)

> 본 보고서는 `.claude/skills/experiment-report.md` 규약을 따르며, §1–§3은
> v2 run의 결과를 보기 전에 작성되었다. §4 이후는 결과가 나온 뒤 채운다. v1
> 보고서 `report_L2048.md` 의 결과는 §1에서 "v2 motivating prior" 로만 참조하며,
> 본 보고서의 §2 pre-registration에는 결과 편향 없이 새 기준으로 등록된다.

## §0. 한 줄 결론

L=2048 v2 실험 (k_align=6, multi-τ MGT) 에서 Algorithm 1은 **alignment
metric에서는 압도적 우위 재확인, recon gap은 v1 대비 38% 좁혀졌으나 여전히
two_recon의 2.1×, group_sparse는 τ=0.95까지는 MGT 지배적이지만 τ=0.99에서는
α=0.9 구조적 붕괴**. `ours`와 `two_recon` 만이 α=0.9, τ=0.99 구간에서 0이
아닌 값을 유지해 **tight threshold + moderate α 구간**에서 독립 2개 SAE
구조의 실제 기하적 이점이 처음으로 정량 확인됨. 9개 pre-reg 중 **7개 pass,
2개 (P2, P6, P7) fail**. H는 **partially supports** (alignment 이득은
확인, recon 동등성과 universality는 기각).

---

## §1. Background and Motivation for v2

### 1.1 v1 요약

`report_L2048.md` 에 기록된 v1 run (`main_comparison_20260411_061611` +
ablation 3개)의 pre-registration 결과:

- ✅ **P1, P3, P4, P5** — alignment metric 및 Stage 2 학습 기여가 예측대로 통과.
- ❌ **P2** — ours 의 `avg_eval_loss` 가 `two_recon` 대비 3.4× 수준까지 악화.
  원인 가설: Stage 1→Stage 2 전환 시 AdamW optimizer state 리셋.
- ❌ **P6** — λ ∈ [2⁻⁴, 2⁸] 전범위에서 `top_mS`가 0.0024 범위로 거의 포화.
- ❌ **P7 on `top_mS`** (예측된 metric 편향), ✅ **on `mgt_shared`, avg_eval**.
- **예상 밖 결과 1** — `group_sparse` (L2,1 loss)가 τ=0.8 기준 `mgt_shared`
  0.92–1.00 로 모든 method를 압도.
- **예상 밖 결과 2** — `k_align` ablation에서 **k=6이 k=4 을 모든 metric에서
  지배** (recon 0.115 vs 0.187, mgt 0.623 vs 0.602, top_mS 0.972 동률).

### 1.2 v2에서 바뀌는 점

v1에서 **결과가 아닌 ablation 출력을 통해 관찰된 두 가지 사실** 을 v2 main에
반영한다. `.claude/skills/experiment-report.md §2.3` 이 허용하는 "결과가 아닌
새 정보 획득에 근거한 revision"에 해당.

1. **`k_align` default 4 → 6**: §4.6 v1 ablation 결과가 k=6을 모든 metric에서
   우세로 확인. main 비교에 반영해 recon gap이 좁혀지는지 / alignment가 유지
   되는지를 새로 검증.
2. **MGT τ ∈ {0.8, 0.95, 0.99} 동시 측정**: v1은 단일 τ=0.8만 저장. 이
   보고서의 feedback 중 `group_sparse` MGT 압도가 threshold-dependent인지
   확인하려면 tighter τ 가 필요.
3. **m_S sweep을 fine scan으로 변경**: `{256, 384, 512, 640, 768, 896, 1024}`
   → `{384, 448, 512, 576, 640}` (n_shared 주변 ±25% 에 5점). v1에서 optimum
   plateau 가 매우 좁다는 것을 확인했으므로 근방 해상도를 높인다.

v2는 v1과 **같은 하드웨어, 같은 generative model, 같은 seed 할당** 을 쓰므로,
metric 확장 + config 변경 효과만 분리할 수 있다.

---

## §2. Hypothesis + Pre-registered Predictions (v2)

### Hypothesis (H, v1과 동일)

> **H (Algorithm 1):** 주어진 synthetic 생성 모델 하에서, Algorithm 1의
> `Stage 1 → 탐욕적 permutation → Stage 2 (diag-only $\mathcal L_\text{aux}$)`
> 은 독립 학습된 두 SAE의 **cross-modal shared latent 정렬** 을 baseline
> (single-recon / two-recon / group-sparse / trace-align) 보다 유의하게 높이며,
> 이 과정에서 reconstruction 품질은 두 개 독립 SAE 수준을 유지한다.

Scope: latent_size $m = 2048$, per-SAE (two/ours) 1024, $d=768$, $k=16$,
$s=0.99$, 10 epochs, 3 seeds, $\alpha \in \{0.3, 0.6, 0.9, 1.0\}$.

### Notation

- τ = 0.8, 0.95, 0.99 동시 측정. 주요 pre-reg 표는 τ=0.95 기준(타이트).
- `top_mS` ≔ `cross_cos_top_mS_mean` — m_S=512 기준.
- 모든 comparison은 seed 3개의 mean 기준이며 pass/fail은 mean으로 판정.
  표준편차는 §4에서 함께 보고.

### P1 [main, v1 유지] — Ours `top_mS` 우위

- **Pass:** $\alpha \in \{0.6, 0.9, 1.0\}$의 모든 cell에서 `ours` mean −
  max(baseline) mean ≥ **+0.10**.
- **Failure scenario:** Stage 2 loss가 Stage 1 배치 상관에 overfit해 eval 대각을
  올리지 못함. v1 에서 확실히 통과한 예측이지만 metric 구조적 편향으로 해석
  주의 필요 (아래 P10 참조).
- **Alternatives**:
  (a) Permutation 단독 이득일 수 있음 → P5에서 분리.
  (b) Per-SAE 용량 절반이지만 총 파라미터 동일 → single-decoder baseline 4개가
  공정 비교. metric asymmetry → P10.

### P2 [main, revised] — Ours `avg_eval` vs `two_recon`, k_align=6 기준

v1에서 λ=1, k_align=4 일 때 ratio 3.4× (pass threshold 1.10) 로 실패. v2는
k_align=6으로 Stage 1이 더 오래 돌고 Stage 2가 4 epoch만 돌므로 recon gap이
줄어들 것으로 예상되지만, v1 ablation이 시사하는 바에 따르면 여전히 fail일
가능성이 높다. **v2에서도 동일한 1.10 threshold를 유지하되**, 추가로 "개선
방향 정량" 을 기록한다.

- **Pass (v1과 동일):** 모든 $\alpha$에서 `ours/two_recon` ratio ≤ 1.10.
- **Fail:** 어느 cell이라도 > 1.10.
- **추가 기록:** v1 `ratio_v1` vs v2 `ratio_v2` 개선 폭 (음수면 v2가 더 좋음).
- **Failure scenario:** Stage 2가 4 epoch만으로는 recon 회복 불충분.
- **Alternatives**:
  (a) optimizer state 리셋이 여전히 작동 → 여전히 2×+ ratio 예상.
  (b) 작은 stage2 budget이 recon 개선에 오히려 도움 → ratio 감소.

### P3 [main, **revised**] — Ours MGT τ=0.95 vs two_recon, at α ∈ {0.6, 0.9, 1.0}

- **측정:** `(img_mgt_shared_tau0.95 + txt_mgt_shared_tau0.95) / 2` 기준 ours
  mean − two_recon mean.
- **Pass:** 세 α 모두에서 차이 ≥ **+0.02**.
- **Failure scenario:** τ=0.95 에선 ours의 decoder row가 GT atom에 충분히
  근접하지 못하여 회복률이 two_recon 수준 이하로 떨어짐.
- **Alternatives**:
  (a) τ=0.95 strict 구간에서는 ours와 two_recon 모두 낮아서 우열이 noise 수준
  이 될 수 있음. → 그 경우 P3 변형 P3b 가 결정.
  (b) ours의 Stage 2가 Phi_S/Psi_S atom 방향을 오히려 평준화시켜 tight τ 기준
  recovery가 감소할 수 있음 → v2에서 observability 포함.

### P3b [sub] — Ours MGT τ=0.99 vs two_recon

- **측정:** τ=0.99 기준 동일 비교.
- **Pass:** α=1.0 셀에서 ours mean > two_recon mean (방향 일치만 요구).
- **Fail:** α=1.0 에서 ours mean ≤ two_recon mean.
- **Label:** [sub]. τ=0.99 는 8.1° 각도로 매우 엄격하므로 두 method 모두 낮을
  것으로 예상. 방향성만 본다.

### P4 [main, v1 유지] — Ours `top_mS` vs group_sparse / trace_align

- **Pass:** α=1.0 에서 ours mean − group_sparse mean ≥ +0.10 **AND** ours mean
  − trace_align mean ≥ +0.10.

### P5 [sub, v1 유지] — Stage 2 학습 이득

- **Pass:** eval `top_mS` ≥ Stage 1 `diag_top_mS_diag_mean` − 0.05, rest ≈ 0.

### P6 [sub, revised] — λ sweep 민감도 (k_align=6)

v1 에서 top_mS가 λ에 거의 무반응 (diff 0.0024). v2에서도 같은 양상이 나올
가능성이 높지만 k_align 변경이 감도에 영향을 줄 수 있으므로 재측정.

- **Pass:** λ = 2⁻⁴ → 2⁸ 양 끝 top_mS 차이 ≥ 0.05, Spearman ≥ 0.5.
- **Label:** [sub]. v1에서 실패했으며 v2에서도 실패할 가능성이 높음을 사전에
  기록.

### P7 [sub, revised — metric을 `mgt_shared_tau0.95` 로 교체] — m_S argmax

v1 의 P7은 `top_mS` 기반이었으나 metric 구조적 편향으로 기각. v2는 tight τ=0.95
MGT 를 criterion으로 사용.

- **측정:** `(img_mgt_shared_tau0.95 + txt_mgt_shared_tau0.95) / 2` 기준, m_S
  sweep `{384, 448, 512, 576, 640}` 에서 argmax.
- **Pass:** argmax ∈ {448, 512, 576} (n_shared 중심 ±1 grid).
- **Failure scenario:** Fine scan에서도 plateau가 너무 평탄해 argmax가 grid
  끝점에 위치.
- **Label:** [sub]. Theorem 1 이론이 예측하는 "m_S = n_shared 에서 최적" 이
  경험적으로 관찰되는지를 확인.

### P8 [main, **new**] — `group_sparse` MGT 지배가 τ-robust

v1 에서 `group_sparse` 가 τ=0.8 MGT를 0.92–1.00 까지 끌어올리는 예상 밖의 결과.
v2는 τ=0.95 / 0.99 에서 이 지배가 유지되는지, 아니면 τ가 올라가면 붕괴하는지를
직접 검증한다. 이 예측은 **Algorithm 1에 대한 것이 아니라, ours와 함께 비교될
경쟁 baseline의 양상** 을 묻는다.

- **측정:** `(img+txt)/2` MGT τ=0.95 기준, α=1.0 셀.
- **Pass:** group_sparse mean ≥ 0.80 (즉 τ=0.95 에서도 80% 이상 GT atom 회복).
- **Fail:** group_sparse mean < 0.80.
- **Intended interpretation:** L2,1 로 학습된 decoder row 가 GT atom 방향에
  "거의 정확하게" 정렬되는가, 아니면 느슨한 τ 범위에서만 통과하는 부분 정렬인가.
- **Alternatives:**
  (a) τ=0.8 에서의 지배는 각 decoder row가 GT atom 방향에 0.8–0.9 구간에 몰려
  있어서 0.95로 올리면 대부분 탈락 → **P8 Fail** 쪽 시나리오.
  (b) L2,1 이 decoder row를 GT atom에 "거의 exact" 하게 정렬 → **P8 Pass**.

### P9 [main, **new**] — v1 vs v2 recon 개선 관찰

v1 ablation에서 k_align 증가에 따라 ours recon이 개선됨을 봤다. v2 main 은
k_align=6 을 기본으로 쓰므로 v1 (k_align=4) 대비 `avg_eval_loss` 가 개선되어야
한다.

- **측정:** v2 `ours avg_eval_loss` vs v1 `ours avg_eval_loss`, α=1.0.
- **Pass:** v2 ratio `v2/v1` ≤ 0.70 (최소 30% 개선).
- **Fail:** v2 ratio > 0.70.
- **Failure scenario:** k_align 변경의 실제 효과가 작음.
- **Label:** [main]. v1 ablation의 k_align=6 결과가 얼마나 main run에 그대로
  이식되는지를 직접 재확인.

### P10 [sub, new] — Metric asymmetry documentation

v1 §4.2 에서 지적한 `cross_cos_top_mS_mean` 비대칭 (ours에만 permutation 적용)
은 본 실험의 구조적 한계. v2에서는 post-hoc greedy permutation 후의 baseline
`top_mS` 추가 측정을 계획했으나 본 run scope에 포함하지 않음. §6.2/§6.4에
한계로 남기고 후속 작업으로 지정. **이 P10 은 numerical prediction이 아닌
"기록 의무"** 로 분류.

---

## §3. Experiment Design (v2)

### 3.1 Synthetic generative model

v1과 동일 (`SyntheticTheoryFeatureBuilder`, n_image=256, n_shared=512, n_text=256,
$d=768$, s=0.99, max_interference=0.1, train 50k / eval 10k).

### 3.2 SAE configuration (v1과 동일한 공정성 규약)

- 총 latent budget $L=2048$ 각 method에 동일 배정.
- single-decoder methods (`single_recon`, `group_sparse`, `trace_align`):
  `TopKSAE(latent_size=2048)` 1개.
- two-SAE methods (`two_recon`, `ours`): per-SAE `latent_size=1024`.
- $k=16$, `normalize_decoder=True`, AdamW `lr=2e-4`, 10 epochs.

### 3.3 Methods

v1의 5개 method 와 동일. 변경점은 **ours 의 main `k_align` default 만**:
4 → **6** (즉 Stage 1 6 epoch, Stage 2 4 epoch).

### 3.4 Sweeps (v2)

**Main:**
- α ∈ {0.3, 0.6, 0.9, 1.0}, seeds 1–3
- ours: (λ=1, m_S=512, **k_align=6**)
- 총 60 training runs.

**Ablation (α=1.0, method=ours):**
- λ sweep: {2⁻⁴, 2⁻², 1, 4, 16, 64, 256}, m_S=512, **k_align=6**, 3 seeds → 21
- m_S fine scan: {384, 448, 512, 576, 640}, λ=1, **k_align=6**, 3 seeds → 15
- k_align sweep: {2, 4, 6, 8}, λ=1, m_S=512, 3 seeds → 12

**총 v2 runs:** 60 + 21 + 15 + 12 = **108**.

### 3.5 Metrics (v2 변경 사항만)

- `img_mgt_shared_tauX`, `txt_mgt_shared_tauX` for X ∈ {0.8, 0.95, 0.99},
  `_compute_recovery_metrics_multi_tau` at `synthetic_theorem2_method.py:498–523`.
- Backward compat: `img_mgt_shared` = `img_mgt_shared_tau0.8` (v1 리포트와
  직접 비교 가능).
- 나머지 metric은 v1과 동일.

### 3.6 Hypothesis scope vs Experiment scope

v1과 동일. `report_L2048.md §3.6` 참조.

---

## §4. Results

출처 JSON (`outputs/synthetic_theorem2/runs/`):

- `method_ns512_k16_main_comparison_20260411_072737` — main, k_align=6
- `method_ns512_k16_ablation_lambda_20260411_074527` — λ sweep, k_align=6
- `method_ns512_k16_ablation_mS_20260411_075238` — m_S fine scan, k_align=6
- `method_ns512_k16_ablation_kalign_20260411_075754` — k_align sweep

모든 셀은 3 seeds 평균 ± std.

### 4.1 Main comparison: `avg_eval_loss`

| α | single_recon | two_recon | group_sparse | trace_align | ours ($k_\text{align}{=}6$) |
|---|---|---|---|---|---|
| 0.3 | 0.064 ± 0.002 | **0.053 ± 0.001** | 0.124 ± 0.003 | 0.064 ± 0.002 | 0.113 ± 0.004 |
| 0.6 | 0.070 ± 0.001 | **0.051 ± 0.002** | 0.126 ± 0.002 | 0.070 ± 0.001 | 0.110 ± 0.004 |
| 0.9 | 0.060 ± 0.000 | **0.052 ± 0.001** | 0.111 ± 0.001 | 0.060 ± 0.000 | 0.111 ± 0.002 |
| 1.0 | **0.029 ± 0.001** | 0.054 ± 0.002 | 0.087 ± 0.005 | **0.029 ± 0.000** | 0.115 ± 0.003 |

**v1 대비**: `ours avg_eval` (α=1.0) 0.1866 → 0.1151 (**-38%**). 다른 method 값은 변동 없음 (random seed까지 동일하므로 기대된 결과).

`ours/two_recon` ratio:

| α | v1 (k=4) | v2 (k=6) | Δ |
|---|---|---|---|
| 0.3 | 3.43 | 2.13 | **-38%** |
| 0.6 | 3.56 | 2.14 | -40% |
| 0.9 | 3.45 | 2.14 | -38% |
| 1.0 | 3.43 | 2.13 | -38% |

### 4.2 Main comparison: `cross_cos_top_mS_mean` (m_S=512) + `rest`

| α | metric | single | two | group_sparse | trace | ours |
|---|---|---|---|---|---|---|
| 0.3 | top_mS | 0.085 | −0.000 | 0.292 | 0.086 | **0.969** |
| 0.3 | rest | 0.088 | 0.001 | 0.282 | 0.088 | **−0.002** |
| 0.6 | top_mS | 0.380 | 0.001 | 0.409 | 0.380 | **0.971** |
| 0.6 | rest | 0.370 | 0.001 | 0.393 | 0.371 | **−0.001** |
| 0.9 | top_mS | 0.504 | 0.002 | 0.494 | 0.502 | **0.972** |
| 0.9 | rest | 0.500 | 0.000 | 0.476 | 0.501 | **−0.002** |
| 1.0 | top_mS | 0.535 | 0.001 | 0.525 | 0.535 | **0.972** |
| 1.0 | rest | 0.532 | −0.000 | 0.505 | 0.531 | **−0.001** |

v1과 실질적으로 동일 (alignment는 k_align 변경에 거의 무감각). `top_mS`의 **metric 비대칭** 문제는 v2에서도 그대로 존재 — 자세한 논의는 §5.1과 §6.2.

### 4.3 Main comparison: multi-τ MGT

**(img + txt) / 2**, 세 τ 모두:

| α | τ | single | two | group_sparse | trace | ours |
|---|---|---|---|---|---|---|
| 0.3 | 0.80 | 0.514 | 0.479 | **0.916** | 0.520 | 0.634 |
| 0.3 | 0.95 | 0.182 | 0.174 | **0.745** | 0.184 | 0.261 |
| 0.3 | 0.99 | 0.064 | 0.074 | **0.295** | 0.070 | 0.084 |
| 0.6 | 0.80 | 0.620 | 0.495 | **0.990** | 0.619 | 0.652 |
| 0.6 | 0.95 | 0.125 | 0.180 | **0.444** | 0.127 | 0.273 |
| 0.6 | 0.99 | 0.021 | 0.078 | **0.163** | 0.020 | 0.094 |
| 0.9 | 0.80 | 0.673 | 0.500 | **0.999** | 0.671 | 0.642 |
| 0.9 | 0.95 | 0.165 | 0.173 | **0.990** | 0.171 | 0.254 |
| 0.9 | 0.99 | **0.000** | 0.076 | **0.000** | **0.000** | 0.086 |
| 1.0 | 0.80 | 0.724 | 0.474 | **1.000** | 0.729 | 0.623 |
| 1.0 | 0.95 | 0.186 | 0.155 | **0.999** | 0.183 | 0.229 |
| 1.0 | 0.99 | 0.076 | 0.062 | **0.717** | 0.074 | 0.075 |

**두 가지 주목할 관찰.**

(1) **α = 0.9, τ = 0.99의 "shared decoder 구조적 붕괴"**. `single_recon`,
`group_sparse`, `trace_align` 세 shared-decoder method가 **모두 정확히 0.000**.
원인은 기하적: α=0.9는 $\cos(\Phi_{S,i}, \Psi_{S,i}) \approx 0.9$ (각도 ~25°).
두 서로 다른 방향을 **한 개의 decoder row가 동시에 cos > 0.99 (각도 ≤ 8°)
이내로 커버할 수 없다** — 중간 방향을 택하면 양쪽 모두로부터 ~12.5°씩 떨어지기
때문. α=1.0이면 $\Phi_S = \Psi_S$라 한 row로 가능, α=0.3이면 두 방향이 너무
달라서 **별도 row로 학습** (τ=0.99 기준 group_sparse 0.295). α=0.6 역시 이미
shared decoder의 한계가 일부 보임 (group_sparse 0.163).

`ours`와 `two_recon`은 **독립적인 두 SAE**를 사용하므로 이 실패 지점이 없다 —
`ours` τ=0.99 at α=0.9 = 0.086, `two_recon` = 0.076. 이는 **본 실험에서
ours 가 baseline 전체에 대해 처음으로 유의한 우위**를 보이는 지점이다 (비록
절대값은 작지만).

(2) **group_sparse 가 τ=0.95에서도 α ∈ {0.9, 1.0} 에서 0.99 이상을 유지.**
L2,1 loss가 유도하는 decoder row가 GT atom에 거의 exact 하게 정렬됨. α < 0.9
에서는 threshold가 올라갈수록 급락 (τ=0.8 0.92 → τ=0.95 0.74 → τ=0.99 0.29).

### 4.4 Ablation: λ sweep (α=1.0, m_S=512, k_align=6)

| λ | avg_eval | top_mS | MGT τ=0.8 | MGT τ=0.95 | MGT τ=0.99 |
|---|---|---|---|---|---|
| $2^{-4} = 0.0625$ | **0.1047** | 0.9700 | **0.633** | **0.241** | **0.081** |
| $2^{-2} = 0.25$ | 0.1124 | 0.9716 | 0.626 | 0.231 | 0.077 |
| $2^{0} = 1$ | 0.1151 | 0.9721 | 0.623 | 0.229 | 0.075 |
| $2^{2} = 4$ | 0.1159 | 0.9723 | 0.622 | 0.229 | 0.075 |
| $2^{4} = 16$ | 0.1161 | 0.9724 | 0.621 | 0.230 | 0.074 |
| $2^{6} = 64$ | 0.1161 | 0.9724 | 0.621 | 0.231 | 0.074 |
| $2^{8} = 256$ | 0.1162 | 0.9723 | 0.622 | 0.231 | 0.074 |

- **`top_mS` saturation**: $\lambda = 0.0625 \to 256$ 범위에서 0.9700 → 0.9723,
  diff = 0.0023. **P6 fail 재확인**.
- **작은 λ가 모든 다른 metric에서 우세**: `avg_eval`, MGT τ=0.8/0.95/0.99 모두
  $\lambda = 0.0625$ 에서 최적. λ를 더 작게 (e.g. $2^{-6}$) 해도 포화 유지될 것으로
  추정.
- **v1 대비**: k_align=4 v1에선 avg_eval 범위 0.152 → 0.191 이었는데 v2에서는
  0.105 → 0.116. **k_align 6이 모든 λ에서 recon을 크게 개선**.

### 4.5 Ablation: m_S fine scan (α=1.0, λ=1, k_align=6)

| m_S | avg_eval | top_mS | MGT τ=0.8 | MGT τ=0.95 | MGT τ=0.99 |
|---|---|---|---|---|---|
| 384 | 0.1331 | **0.9756** | 0.549 | 0.156 | 0.046 |
| 448 | 0.1228 | 0.9738 | 0.589 | 0.192 | 0.064 |
| 512 | 0.1151 | 0.9721 | **0.623** | 0.229 | 0.075 |
| 576 | **0.1118** | 0.9708 | 0.628 | 0.272 | 0.079 |
| 640 | 0.1181 | 0.9692 | 0.594 | **0.292** | **0.087** |

관찰:

- `avg_eval` argmin: **m_S = 576** (0.1118), 오라클 512와 거의 동률.
- MGT τ=0.8 argmax: **m_S = 576** (0.628), 512와 거의 동률 (0.623).
- **MGT τ=0.95 argmax: m_S = 640** (0.292). n_shared보다 25% 더 큼.
- **MGT τ=0.99 argmax: m_S = 640** (0.087). 같은 결론.
- 즉 **엄격한 τ가 요구될수록 최적 m_S 가 n_shared 보다 위쪽으로 shift**.
  직관: tight τ에서 decoder row 가 GT atom 방향으로 "정확히" 수렴하려면 더
  많은 anchor가 필요하기 때문으로 보인다 (n_shared 개보다 많은 latent 에
  정렬 압력을 가하면 Stage 2 learning이 더 많은 row를 GT-like 방향으로 끌어당김).
- **P7** (argmax ∈ {448, 512, 576}): 기준 metric이 MGT τ=0.95 → argmax가 **640**
  → **fail**. 단, 576 도 0.272로 매우 근소한 2위. grid를 {640, 704, 768} 까지
  확장해보면 어디서 peak가 생기는지 알 수 있을 것이다 (§5.3).

### 4.6 Ablation: k_align sweep (α=1.0, λ=1, m_S=512)

| $k_\text{align}$ | Stage 1 / 2 ep | avg_eval | top_mS | MGT τ=0.8 | MGT τ=0.95 | MGT τ=0.99 |
|---|---|---|---|---|---|---|
| 2 | 2 / 8 | 0.3288 | 0.9634 | 0.546 | 0.173 | 0.001 |
| 4 | 4 / 6 | 0.1866 | 0.9719 | 0.602 | 0.201 | 0.026 |
| **6** | **6 / 4** | **0.1151** | **0.9721** | **0.623** | 0.229 | 0.075 |
| 8 | 8 / 2 | **0.0817** | 0.9429 | 0.594 | 0.222 | **0.079** |

관찰:

- **$k_\text{align} = 6$ 이 여전히 균형점**. top_mS 0.972 (k=4와 동률), recon
  0.115 (k=4 대비 38% 개선), MGT τ=0.8 0.623 (최고).
- **$k_\text{align} = 8$ 이 recon(0.082) 과 MGT τ=0.99(0.079) 에서 최고**
  지만, Stage 2가 2 epoch만 돌아 **top_mS가 0.943으로 떨어짐**. 즉 "완전 two_recon
  쪽" 으로 이동하는 중. alignment를 포기할 수 있다면 k=8, 아니면 k=6.
- k=2는 모든 metric에서 최악.

### 4.7 Pass/fail verdict per prediction

아래 모든 verdict는 §2의 기계적 pass 조건으로 판정한다. 예측 자체는 수정하지
않음.

- **P1 [main]** — Ours `top_mS` > max(baseline) + 0.10 at α ∈ {0.6, 0.9, 1.0}:
  - α=0.6: ours 0.971 − max baseline (group_sparse 0.409) = **+0.562** ✓
  - α=0.9: ours 0.972 − single 0.504 = **+0.468** ✓
  - α=1.0: ours 0.972 − single 0.535 = **+0.437** ✓
  - 결과: **✅ PASS**. metric asymmetry caveat (§6.2) 유지.
- **P2 [main]** — Ours `avg_eval` / `two_recon` ≤ 1.10:
  - α=0.3: 2.13 ❌ / α=0.6: 2.14 ❌ / α=0.9: 2.14 ❌ / α=1.0: 2.13 ❌
  - 결과: **❌ FAIL** at every α. v1의 3.43–3.56 → v2의 2.13–2.14로 **38% 수준
    개선** 됐으나 pass threshold 1.10은 여전히 멀다. 잔여 gap은 (a) Stage 2가
    4 epoch밖에 안 돌아 converged 상태가 아니거나, (b) Stage 2 optimizer reset
    의 구조적 문제에서 기인. §5.1 및 §6.4 참조.
- **P3 [main]** — Ours (img+txt)/2 MGT **τ=0.95** > two_recon + 0.02 at α ∈
  {0.6, 0.9, 1.0}:
  - α=0.6: 0.273 − 0.180 = **+0.093** ✓
  - α=0.9: 0.254 − 0.173 = **+0.081** ✓
  - α=1.0: 0.229 − 0.155 = **+0.074** ✓
  - 결과: **✅ PASS**. ours 가 τ=0.95 strict 기준에서도 two_recon 대비 명확한
    우위를 유지.
- **P3b [sub]** — Ours (img+txt)/2 MGT **τ=0.99** > two_recon @ α=1.0:
  - ours 0.075 vs two_recon 0.062 → **+0.013** (방향만 요구) ✓
  - 결과: **✅ PASS** (방향 일치).
- **P4 [main]** — Ours top_mS − {group_sparse, trace_align} ≥ +0.10 @ α=1.0:
  - vs group_sparse: 0.972 − 0.525 = **+0.447** ✓
  - vs trace_align: 0.972 − 0.535 = **+0.437** ✓
  - 결과: **✅ PASS**. metric asymmetry caveat 적용.
- **P5 [sub]** — Stage 2 eval top_mS ≥ Stage 1 perm top_mS − 0.05, rest≈0:
  - v2 main seed 1: `diag_top_mS_diag_mean` ≈ 0.549 (Stage 1 end);
    eval top_mS ≈ 0.972. 0.972 ≥ 0.549 − 0.05 = 0.499 ✓
  - rest ≈ −0.002 (near zero) ✓
  - 결과: **✅ PASS**. Stage 2 학습 기여 +0.42가 확인됨.
- **P6 [sub]** — λ sweep top_mS end-to-end diff ≥ 0.05, Spearman ≥ 0.5:
  - λ=2⁻⁴ → 2⁸: 0.9700 → 0.9723, diff = **0.0023 < 0.05** ❌
  - Spearman 약 +1 이지만 크기가 무시 수준.
  - 결과: **❌ FAIL**. v1 에서 fail, v2 에서도 같은 패턴. top_mS 가 매우 작은 λ에서
    이미 saturate. 실용 recommendation: **λ ≤ 0.0625**.
- **P7 [sub]** — m_S argmax (MGT τ=0.95 기준) ∈ {448, 512, 576}:
  - argmax = **m_S = 640** (0.292) ❌
  - 576 이 0.272로 2위.
  - 결과: **❌ FAIL**. `(img+txt)/2` MGT τ=0.95 기준 최적점이 oracle 512보다
    약간 위쪽 (640) 으로 shift. 그러나 `avg_eval` (576 argmin) 과 MGT τ=0.8
    (576 argmax) 기준으로는 oracle 근방이다. m_S sweep grid 자체를 더 넓혀
    (e.g. {512, 640, 768, 896, 1024}) 엄격 τ 기준 optimum plateau 형태를 확인할
    필요 있음 (§5.3 follow-up).
- **P8 [main]** — group_sparse MGT τ=0.95 ≥ 0.80 @ α=1.0:
  - α=1.0 group_sparse MGT τ=0.95 = **0.999** ≥ 0.80 ✓
  - 결과: **✅ PASS** with overwhelming margin. L2,1 이 α=1.0 에서 decoder row를
    GT atom에 거의 exact하게 정렬함이 확인됨.
  - 추가 관찰: α=0.9 에서도 0.990 으로 유지. α=0.6 에서 0.444로 급락, α=0.3
    에서 0.745 로 반등 (비단조 — §4.3 참조).
- **P9 [main]** — v2 ours `avg_eval` / v1 ours `avg_eval` ≤ 0.70 @ α=1.0:
  - v1: 0.1866 / v2: 0.1151 → ratio = **0.617 < 0.70** ✓
  - 결과: **✅ PASS**. k_align 4→6 변경만으로 recon 38% 개선.
- **P10 [sub]** — Metric asymmetry documentation: §4.2, §4.3, §6.2 에 기록 완료.

---

## §5. Verdict

> This experiment is an existence proof. The conclusions hold only within the
> assumptions stated above.

### 5.1 Hypothesis-level 결론

Under this synthetic setup (L=2048, per-SAE 1024, $k_\text{align}=6$, $d=768$,
$k=16$, 10 epochs, 3 seeds, α ∈ {0.3, 0.6, 0.9, 1.0}), pre-registration
**9개 + 1 기록의무 중** 기계적 통과:

- ✅ **P1, P3, P3b, P4, P5, P8, P9** — alignment + Stage 2 학습 기여 + τ=0.95
  strict 기준 two_recon 우위 + group_sparse τ-robustness + v2 recon 개선
- ❌ **P2** (recon ≤ 1.10×two_recon) — 2.13×로 개선했으나 여전히 fail
- ❌ **P6** (λ 민감도) — top_mS가 λ=2⁻⁴에서 이미 saturate
- ❌ **P7** (m_S argmax at τ=0.95) — argmax가 oracle 512가 아닌 640
- 🗒️ **P10** — metric asymmetry 문서화 완료

### 5.2 v1 → v2 비교 및 새로 얻어진 결론

1. **k_align = 6이 recon 문제의 상당 부분을 완화**. Stage 1을 6 epoch 돌리면
   optimizer state reset이 일어나는 시점이 이미 수렴에 가까워져, Stage 2 4 epoch
   로도 recon을 `two_recon` 대비 2.13×까지만 저하로 유지. v1의 3.43×에서 -38%.
   남은 2.13×는 구조적(optimizer reset) 문제이거나 Stage 2 budget이 4 epoch로
   부족한 것 중 하나로 추정된다. 두 가설의 분리는 §5.3 follow-up 항목.

2. **L2,1 group_sparse 의 MGT 지배가 threshold-robust하지만 α에 비단조 의존**.
   τ=0.8 기준 모든 α에서 0.92+ 이지만, τ=0.95 / 0.99 기준으로 들여다보면:
   - α=1.0: τ=0.95 0.999, τ=0.99 0.717 — 거의 exact alignment.
   - α=0.9: τ=0.95 0.990, **τ=0.99 0.000** — 극단적 threshold 붕괴.
   - α=0.6: τ=0.95 0.444, τ=0.99 0.163 — 중간.
   - α=0.3: τ=0.95 0.745, τ=0.99 0.295 — 반등.
   이 비단조성의 원인은 "shared decoder row 하나가 $\Phi_{S,i}$와 $\Psi_{S,i}$
   양쪽 모두를 cos>τ 이내로 커버할 수 있느냐"는 기하적 문제. 자세한 분석은
   §5.4.

3. **Shared decoder 구조의 geometric cliff at α=0.9, τ=0.99**. 세 shared-decoder
   method (single, group_sparse, trace) 가 **모두 정확히 0.00**인 지점. 이는
   본 synthetic setup에서 ours / two_recon (독립 2 SAE) 이 가질 수 있는 **기하적
   독점 영역**을 정확히 드러낸다 — 독립 SAE는 Φ와 Ψ를 각자 별도 decoder로
   학습할 수 있으므로 α=0.9에서도 0.08-0.09 수준으로 생존. 절대값은 작지만
   "baseline 전부가 0이고 우리는 0이 아닌" 지점이 처음 나왔다는 점에서 중요.

4. **`top_mS` 가 λ에 매우 무감각 (≤ 0.003 over $2^{-4}$ to $2^{8}$)**.
   "Algorithm 1 의 diagonal alignment 능력은 아주 작은 λ만 있어도 발휘된다"는
   뜻이며, 실용 관점에서는 λ를 크게 할 이유가 없음 (오히려 recon에 해로움).

### 5.3 Follow-ups to L=2048

남은 문제 / 다음에 확인할 것:

1. **Optimizer state 승계 실험** — `_train_ours`에서 Stage 1 AdamW moment/variance
   state를 Stage 2 joint optimizer로 전이. 이렇게 하면 P2 (recon ≤1.10×) 가
   통과할 수 있는지 확인.
2. **Post-hoc fair baseline permutation** — 모든 baseline 의 eval correlation
   행렬에 greedy permutation을 적용한 `top_mS` 를 계산해 §4.2 테이블에 추가.
   이건 재학습 없이 decoder-only 후처리로 가능.
3. **m_S sweep 확장** — 현재 fine scan은 {384, ..., 640}. τ=0.95/0.99 argmax가
   오른쪽 끝 640에서 나왔으므로 grid를 {640, 704, 768, 832} 까지 연장해 실제
   peak를 확인.
4. **L2,1 decoder geometry 분석** — 왜 L2,1 이 τ=0.95까지는 α=0.9에서 거의
   perfect, 그러나 τ=0.99에서 정확히 0으로 붕괴하는지 "compromise row 평균
   각도 분포"를 직접 측정.
5. **k_align = 8 재평가** — recon (0.082) 과 τ=0.99 MGT (0.079) 에서 최고지만
   top_mS가 0.943으로 떨어짐. "alignment를 조금 포기하는 대신 더 잘 수렴하는"
   operating point로서 가치가 있는지 check.

### 5.4 Appendix: Shared decoder at α=0.9, τ=0.99 collapse의 기하

**Setup.** Single shared SAE가 2048개 decoder row를 가지고 있을 때, GT 원자
$\Phi_{S,i}$와 $\Psi_{S,i}$에 대해 $\cos(\Phi_{S,i}, \Psi_{S,i}) = \alpha$. SAE
가 "같은 latent index i"로 두 원자를 모두 표현하려 한다고 가정 (L2,1 loss의
joint support 압박 + top-k sparsity 예산).

**Option A (compromise)**: 하나의 decoder row $\mathbf{w}_i$ 를 두 원자의
기하적 중점 방향으로 학습.
$$\mathbf{w}_i \approx \frac{\Phi_{S,i} + \Psi_{S,i}}{\|\Phi_{S,i} + \Psi_{S,i}\|}$$

$\cos(\Phi_{S,i}, \Psi_{S,i}) = \alpha$ 이면 두 원자 사이 각도 $\theta =
\arccos(\alpha)$. compromise row가 각각에서 벗어나는 각도는 $\theta/2$,
대응 cosine:
$$\cos\!\left(\frac{\theta}{2}\right) = \sqrt{\frac{1+\alpha}{2}}.$$

$\alpha = 0.9$ 이면 $\sqrt{0.95} \approx 0.9747$ — **τ=0.95(≈18°)는 통과**
하지만 **τ=0.99(≈8°)는 실패**.

$\alpha = 1.0$ 이면 $\cos(0°) = 1.0$ — τ=0.99도 pass.

$\alpha = 0.6$ 이면 $\sqrt{0.8} \approx 0.894$ — τ=0.95도 fail (0.894 < 0.95).

**Option B (separate rows)**: $\mathbf{w}_i \approx \Phi_{S,i}$, 다른 row
$\mathbf{w}_j \approx \Psi_{S,i}$. L2,1 penalty 큼 (같은 pair에서 두 다른
index가 fire). 그러나 recon은 양쪽 모두 perfect.

SAE의 최적해는 α에 따라 A↔B를 trade-off:

- $\alpha \to 1$: A가 cost-free (중점이 두 원자와 같음). A 선택.
- $\alpha \to 0$: A의 compromise row가 양쪽 모두에서 심하게 벗어남 → recon이
  나빠져서 B가 유리. B 선택.
- **$\alpha \approx 0.9$**: A의 compromise가 τ=0.95는 넘지만 τ=0.99는
  실패. L2,1 penalty는 B가 내야 해서 A가 trade-off에서 이김.

따라서 α=0.9, τ=0.99 에서 shared decoder 방법은 **구조적으로 MGT=0**.

관찰된 수치:
- group_sparse α=0.9, τ=0.95: 0.990 (compromise가 τ=0.95는 통과 — 예측과 일치)
- group_sparse α=0.9, τ=0.99: **0.000** (compromise가 τ=0.99 실패 — 예측과 일치)
- group_sparse α=0.6, τ=0.95: 0.444 (일부 row는 compromise + 일부 separate —
  혼합 전략)
- group_sparse α=1.0, τ=0.99: 0.717 (compromise가 곧 exact이므로 τ=0.99 pass,
  72% 원자가 수치적 수렴에 성공)

**왜 `ours` / `two_recon` 은 이 cliff 를 겪지 않는가.** 독립 2개 SAE이므로
image 쪽 SAE는 Φ_S에, text 쪽 SAE는 Ψ_S에 각각 별도로 decoder row를 학습 — 즉
option B를 강제로 쓴다. 따라서 α=0.9에서도 τ=0.99 원자 복원 자체는 가능
(`ours` 0.086, `two_recon` 0.076). 절대값이 작은 건 Stage 2 budget, capacity,
top-k sparsity 등의 다른 제약 때문.

### 5.5 Scope-bounded conclusion

Under this synthetic assumption, **Algorithm 1의 alignment 이득은 재현되고
v2에서 recon gap은 상당히 좁혀졌다 (v1 3.4× → v2 2.1×)**. 그러나 `two_recon`
수준의 recon 동등성 주장(P2)은 여전히 기각되며, feature-level recovery(MGT)
에서 `group_sparse` L2,1의 압도는 **α=0.9에서 tight τ=0.99 에서만 기하적으로
붕괴** — 이 지점이 independent 2 SAE 구조의 고유 이점이 수치적으로 드러나는
유일한 구간.

H는 **partially supports the hypothesis**: alignment metric 우위는 supported,
cost-free / dominant 주장은 rejects, group_sparse 대체 가능성은 "특정
(α, τ) 구간에서만 우리가 유일한 선택지" 로 제한적으로만 지지됨.

---

## §6. Limitations and Scope (v2)

v1과 동일한 §6.1/§6.2/§6.3. §6.4 는 v1 한계에 아래 항목을 추가.

### 6.4 Un-ruled-out alternative mechanisms (v1 + v2 추가)

- **Optimizer state 리셋 (v1 §5.1에서 진단됨).** v2는 k_align 조정만으로 이
  문제를 완화하려 하지만 구조적 해결은 아니다. 후속으로 `_train_ours` 에서
  Stage 1 AdamW state 를 Stage 2 joint optimizer 로 승계하는 구현 필요.
- **Baseline post-hoc permutation 미적용.** P1/P4 pass margin이 metric 비대칭
  때문에 과대평가될 수 있음. v2 run 자체에서는 baseline 에 permutation을 적용한
  fair metric을 추가 측정하지 않는다 — 후속 작업 1순위.
- **m_S sweep 의 n_shared 근방 fine scan 은 여전히 oracle 정보 사용.** rho-기반
  auto-detection과의 비교는 별도 실험 필요.
- **v1 vs v2 교차 비교는 동일 seed 할당에서 가능** 하지만, AdamW 초기 상태가
  정확히 같은지는 가정. `_seed_everything` 이 모든 RNG 를 고정하므로 계통적
  비교 가능하지만, CUDA 비결정성의 잔여 영향이 있을 수 있음 (§6.2).
