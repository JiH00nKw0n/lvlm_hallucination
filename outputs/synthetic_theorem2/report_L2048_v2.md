# Theorem 2 / Algorithm 1 실험 보고서 — latent_size = 2048 (**v2**: k_align=6, multi-τ MGT)

> 본 보고서는 `.claude/skills/experiment-report.md` 규약을 따르며, §1–§3은
> v2 run의 결과를 보기 전에 작성되었다. §4 이후는 결과가 나온 뒤 채운다. v1
> 보고서 `report_L2048.md` 의 결과는 §1에서 "v2 motivating prior" 로만 참조하며,
> 본 보고서의 §2 pre-registration에는 결과 편향 없이 새 기준으로 등록된다.

## §0. 한 줄 결론

*(v2 결과 확인 후 작성)*

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

*(v2 main + ablation 완료 후 작성)*

### 4.1 Main comparison: `avg_eval_loss`

### 4.2 Main comparison: `cross_cos_top_mS_mean`, `rest`

### 4.3 Main comparison: multi-τ MGT (τ = 0.8, 0.95, 0.99)

### 4.4 Ablation: λ sweep (k_align=6)

### 4.5 Ablation: m_S fine scan

### 4.6 Ablation: k_align sweep

### 4.7 Pass/fail verdict per prediction

- **P1 [main]**: _(TBD)_
- **P2 [main]**: _(TBD)_
- **P3 [main]**: _(TBD)_
- **P3b [sub]**: _(TBD)_
- **P4 [main]**: _(TBD)_
- **P5 [sub]**: _(TBD)_
- **P6 [sub]**: _(TBD)_
- **P7 [sub]**: _(TBD)_
- **P8 [main]**: _(TBD)_
- **P9 [main]**: _(TBD)_
- **P10 [sub]**: _(기록 의무 — numerical verdict 없음)_

---

## §5. Verdict

*(v2 결과 확인 후 작성)*

> This experiment is an existence proof. The conclusions hold only within the
> assumptions stated above.

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
