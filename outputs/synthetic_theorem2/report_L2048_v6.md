# Theorem 2 / Algorithm 1 — L = 2048 보고서 (v6: dual aux_norm + refined sweeps)

> `.claude/skills/experiment-report.md` 규약 준수.
> §1–§3 은 v6 결과를 보기 **전** 에 작성됨 (pre-registration). §4 이후는
> 결과가 나온 뒤 채움. v6 pre-reg 은 L=2048/4096/8192 세 리포트가 공유하는
> 동일 hypothesis 를 갖고, 각 리포트는 자기 L 에서의 predictions 을 독립
> 평가한다. 이 파일은 L=2048 버전.

## §0. 한 줄 결론

*(v6 결과 확인 후 작성)*

---

## §1. Background and motivation for v6

### 1.1 v5 까지 얻은 시사점 요약 (Background)

`report_L2048.md`, `report_L2048_v2.md` 및 v3/v4/v5 run 을 합쳐 보면:

- **Ours 의 alignment 이득은 재현 가능**. `top_mS` 는 모든 (L, α) 에서
  0.97 근방 포화. 어떤 λ 나 latent size 에도 불변.
- **P2 recon ratio 가 L 에 따라 악화**. L=2048 에서 λ=2⁻⁸ 로 1.08× (pass),
  L=4096/8192 에서는 같은 λ 가 1.32-1.47× (fail). 원인: `L_aux` 가
  `.sum()` 이라 latent dim 에 비례해서 scale 이 커짐 → 큰 L 에서 동일 λ 가
  과도한 regularization 으로 작용.
- **`group_sparse` (L2,1) 가 MGT τ ∈ {0.8, 0.95} 에서 ours 압도**.
  `group_sparse` 의 joint sparsity bias 가 합성 생성 모델의 shared
  co-activation 가정과 정확히 일치해서 decoder row 가 GT atom 방향으로
  수렴.
- **Ours 의 유일 승리 구간 = tight τ × 중간 α (특히 α=0.9, τ=0.99)**:
  shared-decoder 방법이 기하적으로 불가능한 구간 (compromise row cosine
  $\sqrt{(1+\alpha)/2} < \tau$) 에서 ours 만 유의한 non-zero 값 유지.
- **`trace_align` baseline 의 paper β=1e-4 은 synthetic scale 에서 "no
  effect"**: 모든 metric 에서 `single_recon` 과 numerically identical.
  논문 Appendix B 가 이미 "below 1e-4 = no effect" 라고 명시.
- **IsoEnergy 논문 본문 Eq. (1) 과 공식 코드의 alignment loss 가 다름**:
  본문은 `-(1/b) Tr(Z̃₁ Z̃₂ᵀ)` 단순 trace, 공식 코드는 top-1 per modality
  를 masking 후 cosine similarity. v6 는 두 variant 를 별도 baseline
  (`trace_align`, `iso_align`) 으로 분리.

### 1.2 v6 에서 바뀌는 점 (Motivation)

1. **`L_aux` 정규화** (`.sum()` → `.mean()`). L-independent λ 확보.
2. **두 정규화 variant** (Option A = per-group mean, Option B = `/n` global
   mean) 을 동시에 실험 → 통계적으로 어느 쪽이 더 적절한지 정량 비교.
3. **`iso_align` baseline 신규 추가**: IsoEnergy 공식 코드의 `alignment_
   penalty(metric='cosim')` 를 정확히 복제 (top-1 masking + cosine sim),
   β=0.03 default.
4. **α sweep refined**: `{0.3, 0.5, 0.7, 0.9}` 로 α=1.0 identity case 제거
   하고 mid-α region uniform 하게.
5. **λ sweep 2의 제곱수 grid**: `{2⁻³, ..., 2³}` = `{0.125, ..., 8}`, 7
   points, factor-2 간격, 정규화된 L_aux 기준 meaningful 영역.
6. **τ set refined**: `{0.9, 0.95, 0.99}` (τ=0.8 drop, Theorem 1 이 다룸).
7. **새 metric `pair_cos_mean`**: eval 셋에서 per-paired-sample latent
   cosine similarity. per-dim diagonal correlation (`cross_cos_top_mS_mean`)
   과 complementary: "한 쌍의 샘플 이 latent 공간에서 얼마나 가까운가".

### 1.3 v5 → v6 수식 비교

```
v5 L_aux (raw sum)      = Σ_{i∈[m_S]}(C_ii − 1)² + Σ_{i∉[m_S]} C_ii²       # O(n)
v6 L_aux (group, A)     = (1/m_S) Σ_shared(C_ii − 1)² + (1/(n−m_S)) Σ_private C_ii²
v6 L_aux (global, B)    = (1/n)   [Σ_shared(C_ii − 1)² + Σ_private C_ii²]
```

v5 λ ≈ 2⁻⁸ 과 v6 λ ≈ 1 의 숫자 비교 불가능 (다른 정규화).

---

## §2. Hypothesis + pre-registered predictions (L=2048)

### Hypothesis H (v6)

> Under the synthetic generative model (sec. §3.1), Algorithm 1
> **(Ours)** — with mean-normalized `L_aux` and the unified sweep of α,
> λ, m_S, k_align — produces SAE decoders such that:
>
> (i) latent index alignment (`top_mS`) is near-saturate at 0.97 for
> both `aux_norm = group` and `global`, independent of λ in the main
> operating range;
>
> (ii) recon cost is within 1.10× two_recon baseline at L=2048, achieving
> the previously missing **P2 pass** with the L-independent formulation;
>
> (iii) per-pair eval latent cosine similarity (`pair_cos_mean`) is
> substantially higher than all four baselines (> +0.20 absolute
> separation) across all four α;
>
> (iv) at the tight threshold τ=0.99 Ours remains the only method with
> meaningfully non-zero MGT across **all four α** (extending the v5
> α=0.9 cliff observation to include α=0.3, 0.5, 0.7).

Baselines: `single_recon`, `two_recon`, `group_sparse` (λ=0.05),
`trace_align` (β=1e-4, paper formula), `iso_align` (β=0.03, code variant).

### Notation

All metrics are measured on the **eval set** (10k paired samples).

- `avg_eval_loss` ≔ $(L_\text{recon,img} + L_\text{recon,txt})/2$, FVU-normalized.
- `top_mS` ≔ `cross_cos_top_mS_mean` — first $m_S$ diagonal entries of
  paired signed Pearson on dense top-k latents.
- `pair_cos_mean` ≔ per-paired-sample cosine similarity of dense top-k
  latents, averaged over eval.
- `img_mgt_shared_tauX + txt_mgt_shared_tauX` / 2 ≔ fraction of GT shared
  atoms whose best-matching decoder row has cosine > X.

### P1 [main] — Ours top_mS 우위 전 α × 두 norm variant

- **측정:** `ours::lam1.0_mS512_k6_normgroup / top_mS` and
  `..._normglobal / top_mS`, at each α ∈ {0.3, 0.5, 0.7, 0.9}.
- **Pass:** both variants ≥ 0.90 at every α, AND ours mean − max(baseline)
  mean ≥ +0.30 at every α. (Both thresholds relaxed from v5 0.97/+0.10
  because α=0.3 is the hardest case; still dominant.)
- **Failure scenario:** Stage 2 L_aux 가 mean-normalized 형태로 너무
  약해서 baseline 수준으로 포화 못함.
- **Intended interpretation:** permutation + mean-normalized aux 가 top_mS
  를 여전히 saturate 시킴 (independent of the normalization choice).
- **Alternative explanations:**
  (a) Option A 와 B 가 서로 다른 수렴점에 도달해서 한쪽만 pass 할 수 있음.
  (b) top_mS 는 ours 에만 permutation 적용이라 metric asymmetry. →
  `pair_cos_mean` (P3) 으로 교차 검증.

### P2 [main] — Ours recon ≤ 1.10× two_recon at L=2048, both variants

- **측정:** `avg_eval_loss / two_recon:avg_eval_loss` ratio at each α, each
  aux_norm.
- **Pass:** ratio ≤ 1.10 at every α AND every aux_norm variant.
- **Fail:** any α × norm 셀이 1.10 초과.
- **Failure scenario:** 정규화된 L_aux 에 맞는 λ=1 이 그래도 과도한
  regularization 을 걸어서 recon 이 2× 이상.
- **Intended interpretation:** mean-normalized L_aux 가 L-independent 한
  sweet spot λ 를 제공 → P2 finally passes.
- **Alternative explanations:**
  (a) λ=1 이 Option A 와 B 의 sweet spot 이 다를 수 있음 (B 는 더 강해서
  λ=0.5 가 적절).
  (b) α=0.3 은 shared atom 이 너무 달라 recon 자체가 어려워서 ratio 커질
  수 있음.

### P3 [main] — Ours pair_cos_mean ≫ 전 baseline at every α

- **측정:** `pair_cos_mean` for ours (either variant) vs all 4 baselines.
- **Pass:** ours pair_cos − max(baseline) pair_cos ≥ **+0.20** at every α.
- **Fail:** any α 에서 margin < 0.20.
- **Failure scenario:** baseline 중 `iso_align` 이 top-1 masked cosine 을
  직접 optimize 해서 pair_cos 가 ours 와 comparable 해짐.
- **Intended interpretation:** per-sample pair cosine 도 ours 가 압도
  (top_mS 와 independent 한 증거).
- **Alternative explanations:**
  (a) `iso_align` 이 pair_cos 에서 경쟁적일 수 있음 (같은 per-sample
  cosine 을 직접 최적화). → margin 축소 예측.
  (b) shared-decoder baseline 들은 α=1.0 에선 자연스럽게 pair_cos 가 높지만
  α ≤ 0.9 에서는 compromise 가 생겨서 낮을 것.

### P4 [main] — Ours at α=0.9 × τ=0.99 is only non-zero MGT method

- **측정:** `(img+txt)/2 MGT τ=0.99` at α=0.9.
- **Pass:** ours ≥ 0.10 **AND** `single_recon`, `group_sparse`,
  `trace_align`, `iso_align` 모두 ≤ 0.02.
- **Failure scenario:** `iso_align` 의 top-1 masking 이 shared-decoder 의
  geometric cliff 를 풀어줘서 non-zero 값을 만들거나, `group_sparse` 의
  L2,1 이 α=0.9 에서 separate row (Option B) 로 전환되어 cliff 탈출.
- **Intended interpretation:** L=2048 에서 v5 에서 발견한 α=0.9, τ=0.99
  geometric cliff 가 v6 에서도 재현.
- **Alternative explanations:**
  (a) `two_recon` 도 independent 2-SAE 라 cliff 탈출 — 이건 v5 에서 이미
  관찰 (two_recon τ=0.99 = 0.077 at L=2048). P4 는 shared-decoder 3개만
  대상으로 함.

### P5 [main, **new**] — Option A (group) 과 Option B (global) 비교

- **측정:** two variants 의 `avg_eval_loss`, `top_mS`, `pair_cos_mean`,
  `MGT τ=0.95`, `MGT τ=0.99` 를 α=0.9 기준으로 비교.
- **Pass:** **하나의 variant 가 결정적 우위** 를 못 가짐 → Option A 와 B
  의 차이가 각 metric 에서 mean 기준 |Δ| ≤ 0.05 이내. (둘 다 valid 라는
  가설)
- **Fail (either direction):** 어느 한쪽이 모든 metric 에서 +0.05 이상
  우위면 → 둘 중 하나가 객관적으로 나음. 이 경우 §5 에서 recommendation.
- **Intended interpretation:** m_S = n/2 인 현재 세팅에선 Option A 와 B 가
  상수 배 관계라 수렴점이 비슷해야 함. 차이가 크면 이론과 실험의 괴리.
- **Alternative explanations:**
  (a) m_S 고정 + seed 단일 이라 noise 로 small Δ 발생 가능.
  (b) Option B 는 λ=1 이 너무 약할 수 있음 (shared 기여가 (m_S/n)=1/2 로
  wash out) → B 에 불리한 비교가 될 수 있음. → λ ablation (§4.4) 에서 재
  평가.

### P6 [sub] — λ sweep saturation

- **측정:** ablation_lambda (α=0.9, 7-point grid 2⁻³..2³, both norms).
- **Pass:** `top_mS` end-to-end 차이 (λ=2⁻³ vs λ=2³) ≤ 0.10. (v5 에서 본
  saturation 이 v6 에서도 유지되는지 확인.)
- **Failure scenario:** λ 가 너무 작거나 (under-regularization) 너무 크면
  (over-regularization) top_mS 가 움직일 수 있음. 예상: 양끝에서 0.5 이하.
- **Label:** [sub]. Saturation 자체는 실용 recommendation 용.

### P7 [sub] — m_S argmax on MGT τ=0.95

- **측정:** ablation_mS, `(img+txt)/2 MGT τ=0.95` 기준 argmax over
  `{384, 448, 512, 576, 640}`.
- **Pass:** argmax ∈ {512, 576}.
- **Fail:** argmax ∈ {384, 448, 640}.
- **Failure scenario:** m_S optimum 이 oracle (512) 보다 살짝 벗어나는
  것은 v5 에서도 관찰됨 (특히 τ=0.95 strict 조건에서).
- **Label:** [sub]. `m_S = n_shared` 직관 검증.

### P8 [sub] — k_align sweet spot

- **측정:** ablation_kalign `{2, 4, 6, 8, 10}`, `avg_eval_loss`,
  `top_mS`, `MGT τ=0.9`.
- **Pass:** k=6 이 `avg_eval_loss` 와 `top_mS` 두 metric 모두에서 top-2
  (best 또는 second best).
- **Fail:** k=6 이 둘 다 3위 이하.
- **Label:** [sub].

---

## §3. Experiment design

### 3.1 Synthetic generative model

v2 와 동일 (`SyntheticTheoryFeatureBuilder`, n_shared=512, n_image=256,
n_text=256, d=768, sparsity=0.99, max_interference=0.1, train 50k / eval
10k / test 10k). α 는 `shared_mode="range"` 로 `[α ± 0.03]` 범위 per-atom.
α=1.0 identity case 는 drop.

### 3.2 SAE configuration

- **공정성 규약 그대로**: single-decoder baselines (single_recon,
  group_sparse, trace_align, iso_align) 은 latent=2048; two_recon 과 ours
  는 per-SAE latent=1024 (= 2048/2).
- `k=16`, `normalize_decoder=True`, `AdamW(lr=2e-4)`, `num_epochs=10`.

### 3.3 Methods

| # | 이름 | Arch | per-SAE latent | β/λ default |
|---|---|---|---|---|
| 1 | `single_recon` | 1 SAE | 2048 | — |
| 2 | `two_recon` | 2 SAE | 1024 | — |
| 3 | `group_sparse` | 1 SAE + L2,1 | 2048 | λ=0.05 (paper) |
| 4 | `trace_align` | 1 SAE + paper Eq.1 trace | 2048 | β=1e-4 (paper) |
| 5 | **`iso_align`** | 1 SAE + IsoEnergy code loss (top-1 masked cosim) | 2048 | **β=0.03** (code default) |
| 6 | `ours (group)` | 2 SAE + Alg.1 + Option A | 1024 | λ=1, m_S=512, k=6 |
| 7 | `ours (global)` | 2 SAE + Alg.1 + Option B | 1024 | λ=1, m_S=512, k=6 |

### 3.4 Main sweep

- α ∈ {0.3, 0.5, 0.7, 0.9}
- seeds = 1 (single seed for speed; 5+ seeds 재검증은 §6 한계)
- 총 main runs: 7 method-configs × 4 α × 1 seed = **28 configs**

### 3.5 Ablations (all at α=0.9, method=ours only)

- **λ sweep**: `{0.125, 0.25, 0.5, 1, 2, 4, 8}` (7 points), m_S=512,
  k_align=6, both aux_norm → 7 × 2 = 14 configs
- **m_S sweep**: `{384, 448, 512, 576, 640}` (5 points), λ=1, k_align=6,
  both aux_norm → 5 × 2 = 10 configs
- **k_align sweep**: `{2, 4, 6, 8, 10}` (5 points), λ=1, m_S=512, both
  aux_norm → 5 × 2 = 10 configs

총 **34 ablation configs + 28 main configs = 62 model trainings** for
L=2048.

### 3.6 Metrics

- `avg_eval_loss` — FVU-normalized recon
- `cross_cos_top_mS_mean`, `cross_cos_rest_mean` — per-dim diagonal correlation (existing)
- **`pair_cos_mean`** — per-paired-sample latent cosine (**new v6**)
- MGT τ ∈ {0.9, 0.95, 0.99} — GT atom recovery rates
- `mip_shared` — mean best cosine with GT atoms

Code refs: `_eval_pair_latent_cosine` (new), `_compute_recovery_metrics_
multi_tau` (τ 변경), `_auxiliary_alignment_loss` (정규화 + dual variant).

### 3.7 Hypothesis scope vs experiment scope

| | |
|---|---|
| Hypothesis 원래 scope | 실제 VLM image/text encoder embedding |
| Experiment scope | Linear sparse generative model, paired z_S co-activation, matched sparsity, d=768, k=16, paired batch, 10 epochs, single seed, α ∈ {0.3, 0.5, 0.7, 0.9}, L = **2048** |
| Not guaranteed | 실제 VLM 의 paired co-activation 구조, magnitude asymmetry, curved manifold, encoder 초기 정렬 |

---

## §4. Results

*(v6 L=2048 main + ablation 완료 후 작성)*

---

## §5. Verdict

*(결과 확인 후 작성)*

> This experiment is an existence proof. The conclusions hold only within
> the assumptions stated above.

---

## §6. Limitations and Scope

### 6.1 Strong assumptions

- Paired co-activation of z_S, linear sparse combination, matched sparsity
- Decoder unit-norm 강제, drop_last=True 로 마지막 batch drop (<0.5% sample loss)
- Single seed (목표 5 seeds 를 못 채움, v1-v5 report 와 동일 한계)

### 6.2 Metric limitations

- `cross_cos_top_mS_mean` 은 permutation 을 ours 에만 적용 → baseline 과의
  직접 비교는 asymmetric. `pair_cos_mean` (new) 이 이 비대칭을 부분적으로
  해소 — permutation 없이 per-sample cosine 을 재므로 baseline 과 ours 가
  같은 조건에서 비교됨.
- MGT τ ∈ {0.9, 0.95, 0.99} 는 strict angular thresholds. τ=0.8 loose
  버전은 Theorem 1 레포트에서 이미 다뤘으므로 v6 에서 drop.
- `pair_cos_mean` 은 bounded [-1, 1] 이라 scale 해석이 clean 하지만 "어떤
  sample 에서 어떤 latent dim 이 align 되는가" 는 측정 못함.

### 6.3 Extrapolation / generalization

- 본 결과는 L=2048 만 다룸. L=4096/8192 는 sibling reports 에 독립 기재.
- 실제 VLM embedding 으로 transfer 여부는 별도 real-data 실험 필요.

### 6.4 Un-ruled-out alternative mechanisms

- **Option A/B 우열이 m_S sweep 에 dependent**: main 은 m_S=512 고정이라
  두 norm variant 의 차이가 크지 않을 가능성. m_S ≠ n/2 인 경우 차이가
  드러날 수 있음 → §5 에서 ablation_mS 결과로 추가 검증.
- **`iso_align` β=0.03 은 IsoEnergy 코드의 default** 지만 synthetic scale
  에서 optimum 인지 미검증. 논문 main text 가 제시한 β range (1e-5 ~ 1e-2)
  중 가장 큰 값 근처라 optimum 에 가까울 것으로 예상.
- **`trace_align` β=1e-4 는 paper text 값** 이지만 synthetic scale 에선
  almost no effect (v5 까지 관찰됨). §6 에 한계로 유지.
- **Per-SAE capacity 차이**: two/ours 는 per-SAE = L/2 = 1024, shared
  decoder baseline 은 2048. 총 파라미터 수는 같지만 per-modality
  표현력이 절반. 이 trade-off 의 영향은 L=4096/8192 sibling report 에서
  재검증.
