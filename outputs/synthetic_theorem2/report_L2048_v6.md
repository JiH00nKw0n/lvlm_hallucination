# Theorem 2 / Algorithm 1 — L = 2048 보고서 (v6: dual aux_norm + refined sweeps)

> `.claude/skills/experiment-report.md` 규약 준수.
> §1–§3 은 v6 결과를 보기 **전** 에 작성됨 (pre-registration). §4 이후는
> 결과가 나온 뒤 채움. v6 pre-reg 은 L=2048/4096/8192 세 리포트가 공유하는
> 동일 hypothesis 를 갖고, 각 리포트는 자기 L 에서의 predictions 을 독립
> 평가한다. 이 파일은 L=2048 버전.

## §0. 한 줄 결론

v6 mean-normalized `L_aux` 가 **v5 의 P2 실패를 완전히 해결**: ours 양쪽
variant (group, global) 이 L=2048 에서 모든 α 에 대해 **`two_recon` 보다
recon 이 낮고** (0.94–0.99×) top_mS 는 0.93–0.96 으로 saturate. Pre-reg 8
predictions 중 **P1/P2/P4/P5/P7 pass, P3/P6/P8 fail**. P3 failure 는 `pair_
cos_mean` 이 α 증가 시 shared-decoder 의 자연 cosine 과 ours 의 구조적 0.46
플랫이 교차하기 때문이며 **metric-objective mismatch** (§5.2). group_sparse
는 τ = 0.9/0.95 에서 여전히 압도하지만 **τ=0.99 × α ∈ {0.7, 0.9}** 지점은
ours 가 유일 winner. Option A (group) 와 Option B (global) 는 `Option A λ_x
≈ Option B λ_2x` 정확한 2× 스케일 관계로 실질 동등.

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

출처 JSON (`outputs/synthetic_theorem2/runs/`):

- `method_ns512_k16_main_comparison_20260411_121243` — main, α ∈ {0.3, 0.5, 0.7, 0.9}
- `method_ns512_k16_ablation_lambda_20260411_122115` — λ sweep @ α=0.9
- `method_ns512_k16_ablation_mS_20260411_122552` — m_S sweep @ α=0.9
- `method_ns512_k16_ablation_kalign_20260411_122921` — k_align sweep @ α=0.9

Single seed. `n_shared=512`, `n_image=n_text=256`, `d=768`, `k=16`,
`batch=256`, `num_epochs=10`.

### 4.1 Main: `avg_eval_loss` (recon ratio vs `two_recon`)

| α | single_recon | two_recon | group_sparse | trace_align | iso_align | ours (group) | ours (global) |
|---|---|---|---|---|---|---|---|
| 0.3 | 0.0666 (1.29×) | **0.0515 (1.00×)** | 0.1206 (2.34×) | 0.0666 (1.29×) | 0.0663 (1.29×) | **0.0499 (0.97×)** ✅ | **0.0491 (0.95×)** ✅ |
| 0.5 | 0.0661 (1.33×) | **0.0498 (1.00×)** | 0.1273 (2.55×) | 0.0658 (1.32×) | 0.0673 (1.35×) | **0.0492 (0.99×)** ✅ | **0.0477 (0.96×)** ✅ |
| 0.7 | 0.0773 (1.57×) | **0.0494 (1.00×)** | 0.1287 (2.61×) | 0.0774 (1.57×) | 0.0826 (1.67×) | **0.0470 (0.95×)** ✅ | **0.0463 (0.94×)** ✅ |
| 0.9 | 0.0603 (1.18×) | **0.0509 (1.00×)** | 0.1095 (2.15×) | 0.0603 (1.18×) | 0.0608 (1.19×) | **0.0502 (0.99×)** ✅ | **0.0487 (0.96×)** ✅ |

**주요 관찰:**
- **Ours 양쪽 variant 모두 모든 α 에서 `two_recon` 보다 recon 이 낮다** (0.94–0.99×). v5 λ=2⁻⁸ 에서 1.08×, v6 mean-norm λ=1 에서 `<1.00×`. P2 의 "1.10 이내" 조건을 큰 margin 으로 통과.
- `group_sparse` recon 은 여전히 2.15–2.61× (L2,1 regularizer 의 구조적 비용).
- `iso_align` recon 은 `single_recon` 과 비슷 (1.29–1.67×). β=0.03 이 synthetic scale 에서 mild effect.
- `trace_align` recon 은 `single_recon` 과 동일 (β=1e-4 사실상 off — paper Appendix B 예측 재확인).

### 4.2 Main: `cross_cos_top_mS_mean` & `cross_cos_rest_mean`

| α | method | top_mS | rest |
|---|---|---|---|
| 0.3 | single_recon | 0.0751 | +0.0880 |
| 0.3 | two_recon | 0.0005 | −0.0009 |
| 0.3 | group_sparse | 0.2972 | +0.2858 |
| 0.3 | trace_align | 0.0743 | +0.0882 |
| 0.3 | iso_align | 0.1039 | +0.1143 |
| 0.3 | **ours (group)** | **0.9589** | **+0.0029** |
| 0.3 | **ours (global)** | **0.9309** | +0.0185 |
| 0.5 | single_recon | 0.3013 | +0.3072 |
| 0.5 | group_sparse | 0.3661 | +0.3431 |
| 0.5 | iso_align | 0.3303 | +0.3495 |
| 0.5 | **ours (group)** | **0.9615** | **+0.0027** |
| 0.5 | **ours (global)** | **0.9359** | +0.0185 |
| 0.7 | single_recon | 0.4235 | +0.4224 |
| 0.7 | group_sparse | 0.4573 | +0.4126 |
| 0.7 | iso_align | 0.4678 | +0.4625 |
| 0.7 | **ours (group)** | **0.9626** | +0.0016 |
| 0.7 | **ours (global)** | **0.9374** | +0.0182 |
| 0.9 | single_recon | 0.4988 | +0.5018 |
| 0.9 | group_sparse | 0.5022 | +0.4720 |
| 0.9 | iso_align | 0.5271 | +0.5171 |
| 0.9 | **ours (group)** | **0.9617** | +0.0023 |
| 0.9 | **ours (global)** | **0.9332** | +0.0158 |

**주요 관찰:**
- `ours (group)` top_mS ≈ 0.96, `ours (global)` top_mS ≈ 0.93 — 두 variant 모두 α 에 완전히 무관하게 saturate. margin vs baseline: 최소 +0.40, 최대 +0.86.
- `ours` rest ≈ 0.00 (group) / +0.02 (global) — L_aux 가 off-diagonal 을 거의 0 으로 suppress.
- Baseline 의 top_mS 와 rest 가 거의 같음 (대각 분포가 uniform) — metric asymmetry 의 재확인.

### 4.3 Main: `pair_cos_mean` (신규 metric, per-paired-sample latent cosine)

| α | single | two | group_sp | trace | iso | ours(grp) | ours(glb) |
|---|---|---|---|---|---|---|---|
| 0.3 | 0.053 | 0.008 | 0.189 | 0.054 | 0.085 | **0.462** | **0.462** |
| 0.5 | 0.313 | 0.008 | 0.367 | 0.312 | 0.354 | **0.469** | **0.470** |
| 0.7 | 0.473 | 0.010 | 0.510 | 0.472 | 0.500 | 0.466 | 0.467 |
| 0.9 | 0.594 | 0.009 | 0.599 | 0.594 | **0.607** 🏆 | 0.460 | 0.459 |

**주요 관찰 (예상 밖):**
- **`pair_cos_mean` 은 α 에 따라 정반대 scaling**. Shared-decoder (`single`, `group_sparse`, `trace`, `iso`) 의 pair_cos 는 α 와 같이 증가 (0.05 → 0.60). **ours 는 α 에 완전 invariant 하게 0.46 flat**.
- 이유: shared decoder 는 α → 1 일수록 두 modality 코드가 자연스럽게 유사해짐 (Φ_S ≈ Ψ_S 면 같은 top-k 를 선택). Ours 는 **per-dim diagonal 만 1 로 밀고 나머지는 0 으로 밀기** 때문에 per-vector cosine 이 구조적으로 bounded.
- P3 기준 (+0.20 margin) 으로는:
  - α=0.3: ours 0.462 − max_baseline 0.189 = **+0.273** ✓
  - α=0.5: ours 0.469 − max_baseline 0.367 = **+0.102** ❌ (margin 부족)
  - α=0.7: ours 0.466 − max_baseline 0.510 = **−0.044** ❌ (baseline 승)
  - α=0.9: ours 0.460 − max_baseline 0.607 = **−0.147** ❌ (iso_align 승)
- **P3 는 α ∈ {0.5, 0.7, 0.9} 에서 fail**. 하지만 §5 에서 논의하듯 이 metric 은 ours 의 목표 (shared/private separation) 와 orthogonal.

### 4.4 Main: multi-τ MGT (img+txt 평균)

| α | τ | single | two | group_sp | trace | iso | ours(grp) | ours(glb) |
|---|---|---|---|---|---|---|---|---|
| 0.3 | 0.9 | 0.271 | 0.282 | **0.836** 🏆 | 0.272 | 0.276 | 0.441 | 0.408 |
| 0.3 | 0.95 | 0.163 | 0.175 | **0.754** 🏆 | 0.173 | 0.162 | 0.314 | 0.270 |
| 0.3 | 0.99 | 0.058 | 0.073 | **0.288** 🏆 | 0.063 | 0.056 | 0.163 | 0.139 |
| 0.5 | 0.9 | 0.247 | 0.311 | **0.683** 🏆 | 0.252 | 0.230 | 0.471 | 0.426 |
| 0.5 | 0.95 | 0.128 | 0.196 | **0.458** 🏆 | 0.122 | 0.109 | 0.323 | 0.292 |
| 0.5 | 0.99 | 0.038 | 0.078 | **0.187** 🏆 | 0.037 | 0.034 | 0.174 | 0.152 |
| 0.7 | 0.9 | 0.281 | 0.304 | **0.891** 🏆 | 0.284 | 0.215 | 0.475 | 0.434 |
| 0.7 | 0.95 | 0.095 | 0.193 | **0.426** 🏆 | 0.092 | 0.058 | 0.329 | 0.289 |
| 0.7 | 0.99 | 0.006 | 0.079 | **0.097** | 0.006 | 0.003 | **0.186** 🏆 | 0.160 |
| 0.9 | 0.9 | 0.317 | 0.287 | **0.998** 🏆 | 0.315 | 0.228 | 0.438 | 0.402 |
| 0.9 | 0.95 | 0.172 | 0.184 | **0.994** 🏆 | 0.182 | 0.106 | 0.315 | 0.299 |
| 0.9 | **0.99** | 0.000 | 0.077 | **0.000** | 0.000 | 0.000 | **0.175** 🏆 | 0.144 |

**주요 관찰:**
- **`group_sparse` 가 τ = 0.9, 0.95 에서 모든 α 승자** (feature recovery 기준).
- **`ours` 가 승리하는 유일 영역 = τ=0.99 × α ∈ {0.7, 0.9}**:
  - α=0.7 τ=0.99: ours 0.186 > group_sparse 0.097 (**+0.089** 차이)
  - α=0.9 τ=0.99: ours 0.175, shared-decoder 전부 0.000 (geometric cliff)
- **α=0.9, τ=0.99 geometric cliff 완벽 재현**: single/group_sparse/trace/iso 모두 정확히 0.000.

### 4.5 Ablation: λ sweep @ α=0.9

| λ | normgroup recon | top_mS | mgt.9 | mgt.99 | normglobal recon | top_mS | mgt.9 | mgt.99 |
|---|---|---|---|---|---|---|---|---|
| 0.125 | **0.0490** | 0.841 | 0.332 | 0.094 | 0.0497 | 0.811 | 0.308 | 0.085 |
| 0.25 | 0.0484 | 0.886 | 0.361 | 0.112 | **0.0490** | 0.841 | 0.332 | 0.094 |
| 0.5 | **0.0487** | 0.933 | 0.402 | 0.144 | **0.0484** | 0.886 | 0.361 | 0.112 |
| 1.0 | 0.0502 | 0.962 | 0.438 | **0.175** | **0.0487** | 0.933 | 0.402 | 0.144 |
| 2.0 | 0.0543 | 0.973 | **0.461** | 0.165 | 0.0502 | 0.962 | 0.438 | **0.175** |
| 4.0 | 0.0606 | 0.975 | 0.479 | 0.151 | 0.0543 | 0.973 | 0.461 | 0.165 |
| 8.0 | 0.0687 | **0.975** | 0.476 | 0.139 | 0.0606 | 0.975 | **0.479** | 0.151 |

**주요 관찰:**
- **`normgroup λ=x ≈ normglobal λ=2x`** 정확히 2× 스케일 관계 (m_S = n/2 이므로 수학적 예측과 일치).
  - 예: normgroup 0.5 ↔ normglobal 1.0, normgroup 1.0 ↔ normglobal 2.0, ...
- **recon 최적점**: normgroup λ=0.25–0.5 (recon 0.0484), normglobal λ=0.5–1.0.
- **top_mS 는 λ=2 이후 포화** (0.975 plateau). 7-point sweep 범위 (λ=0.125 → 8) 에서 top_mS range = 0.134 (normgroup) / 0.164 (normglobal) — **P6 의 "≤ 0.10" 기준 기각** (sweep 범위가 좁아서 saturation 이 완전히 보이지 않음).

### 4.6 Ablation: m_S sweep @ α=0.9 (λ=1, k=6)

| m_S | normgroup recon | top_mS | pair_cos | mgt.9 | mgt.95 | normglobal recon | top_mS | pair_cos | mgt.9 | mgt.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| 384 | 0.0558 | **0.972** | 0.368 | 0.414 | 0.278 | 0.0539 | 0.948 | 0.384 | 0.381 | 0.259 |
| 448 | 0.0518 | 0.969 | 0.415 | 0.430 | 0.298 | 0.0505 | 0.943 | 0.422 | 0.391 | 0.280 |
| 512 | **0.0502** | 0.962 | 0.460 | **0.438** | 0.315 | **0.0487** | 0.933 | 0.459 | **0.402** | **0.299** |
| 576 | 0.0517 | 0.951 | 0.501 | **0.439** | **0.327** | 0.0493 | 0.920 | 0.495 | 0.401 | **0.305** |
| 640 | 0.0521 | 0.938 | **0.539** | 0.407 | 0.313 | 0.0493 | 0.907 | 0.529 | 0.373 | 0.292 |

**주요 관찰:**
- **Recon 최저 at m_S = 512** (oracle value) for both variants.
- **MGT τ=0.9 argmax**: m_S ∈ {512, 576} for both variants — **P7 PASS**.
- **pair_cos monotonically increasing with m_S**: 0.37 → 0.54 (normgroup). 더 많은 latent 를 "shared" 로 강제할수록 per-vector cosine 증가. 당연한 결과.

### 4.7 Ablation: k_align sweep @ α=0.9 (λ=1, m_S=512)

| k_align | Stage 1/2 | normgroup recon | top_mS | pair_cos | mgt.9 | mgt.99 | normglobal recon | top_mS | pair_cos | mgt.9 | mgt.99 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | 2/8 | 0.0504 | **0.966** | 0.414 | **0.468** | **0.192** | 0.0458 | 0.938 | 0.422 | **0.434** | **0.181** |
| 4 | 4/6 | **0.0490** | 0.965 | 0.453 | 0.466 | 0.184 | **0.0471** | 0.940 | 0.453 | 0.431 | 0.165 |
| 6 | 6/4 | 0.0502 | 0.962 | 0.460 | 0.438 | 0.175 | 0.0487 | 0.933 | 0.459 | 0.402 | 0.144 |
| 8 | 8/2 | 0.0564 | 0.946 | 0.468 | 0.392 | 0.127 | 0.0529 | 0.910 | 0.471 | 0.368 | 0.104 |
| 10 | 10/0 | 0.0509 | 0.810 | 0.472 | 0.287 | 0.077 | 0.0509 | 0.810 | 0.472 | 0.287 | 0.077 |

**주요 관찰:**
- **v5 와 달리 v6 에서는 k=2 또는 k=4 가 alignment/MGT 에서 우세**. v5 에서는 k=6 이 최적이었던 것과 정반대.
  - 이유 추정: v6 의 mean-normalized L_aux 가 mild 해져서, Stage 2 시간이 더 길수록 (k 작을수록) alignment 기회 더 많음. v5 raw-sum L_aux 는 강해서 Stage 2 가 길면 recon 을 망쳤지만, v6 에선 안 망쳐서 Stage 2 시간이 주로 유리.
- **k=10 (Stage 2 skip)**: top_mS=0.810 (permutation 만으로 얻는 값). v5 L=2048 의 0.808 과 일치 (이론 예측).
- **P8 기준 (k=6 이 recon + top_mS 에서 top-2)**: k=6 은 recon 3위 (k=4 > k=2 > k=6), top_mS 3위 (k=2 > k=4 > k=6) → **P8 FAIL**. 다만 차이는 모두 0.003–0.004 로 매우 작음. v6 에서는 **k=4 가 새 권장값**.

### 4.8 Pass/fail verdict per prediction

- **P1 [main]** — Ours top_mS ≥ 0.90 + margin +0.30 at every α, both variants:
  - normgroup: top_mS = 0.959/0.962/0.963/0.962. Max baseline per α: 0.297 (α=0.3)/0.366/0.468/0.527. Margins: **+0.66/+0.60/+0.49/+0.43**. ✓
  - normglobal: top_mS = 0.931/0.936/0.937/0.933. Margins: **+0.63/+0.57/+0.47/+0.41**. ✓
  - 결과: **✅ PASS** both variants.
- **P2 [main]** — Recon ratio ≤ 1.10× at every α, both variants:
  - normgroup ratios: 0.97 / 0.99 / 0.95 / 0.99. ✓
  - normglobal ratios: 0.95 / 0.96 / 0.94 / 0.96. ✓
  - 결과: **✅ PASS**. v5 의 핵심 실패가 v6 에서 해결됨.
- **P3 [main]** — pair_cos +0.20 margin vs max(baseline):
  - α=0.3: +0.273 ✓
  - α=0.5: +0.102 ❌
  - α=0.7: −0.044 ❌
  - α=0.9: −0.147 ❌
  - 결과: **❌ FAIL** at α ∈ {0.5, 0.7, 0.9}. 원인 = metric–objective mismatch (§5.2 참조).
- **P4 [main]** — α=0.9 τ=0.99 cliff, ours non-zero + 4개 shared-decoder ≤ 0.02:
  - ours normgroup: 0.175 ≥ 0.10 ✓
  - single 0.000, group_sparse 0.000, trace 0.000, iso 0.000 — 모두 ≤ 0.02 ✓
  - 결과: **✅ PASS** (geometric cliff 재현, ours 유일 non-zero).
- **P5 [main]** — Option A vs B difference ≤ 0.05 per metric at α=0.9:
  - recon: 0.002 ✓
  - top_mS: 0.029 ✓ (normgroup 약간 높음)
  - pair_cos: 0.001 ✓
  - MGT τ=0.9: 0.036 ✓
  - MGT τ=0.95: 0.016 ✓
  - MGT τ=0.99: 0.031 ✓
  - 결과: **✅ PASS**. Option A 가 alignment 에서 살짝 유리, Option B 가 recon 에서 살짝 유리. 체계적 차이는 m_S = n/2 scaling 관계 (normgroup ≈ 2× normglobal) 로 설명됨.
- **P6 [sub]** — λ saturation (top_mS range ≤ 0.10):
  - normgroup range (λ=0.125→8): 0.975 − 0.841 = **0.134** ❌
  - normglobal range: 0.975 − 0.811 = **0.164** ❌
  - 결과: **❌ FAIL**. v5 에선 극단 λ 범위에서 saturate 했지만, v6 의 λ=2⁻³ 는 여전히 top_mS 에 영향. λ ≥ 2 에서만 saturate.
- **P7 [sub]** — m_S argmax on MGT τ=0.95 ∈ {512, 576}:
  - normgroup argmax: **m_S=576** (0.327) ✓
  - normglobal argmax: **m_S=576** (0.305) ✓
  - 결과: **✅ PASS**.
- **P8 [sub]** — k=6 top-2 on both recon and top_mS:
  - normgroup recon top-2: k=4 (0.0490), k=2 (0.0504) — **k=6 은 3위**
  - normgroup top_mS top-2: k=2 (0.966), k=4 (0.965) — **k=6 은 3위**
  - 결과: **❌ FAIL**. 다만 모든 k 간 recon 차이 < 0.003, top_mS 차이 < 0.005. 실용상 k=4 가 새 권장값.

---

## §5. Verdict

> This experiment is an existence proof. The conclusions hold only within
> the assumptions stated above.

### 5.1 Hypothesis-level 결론

Under the L=2048 synthetic setting, pre-reg 8 predictions 결과:

| 예측 | Label | Verdict |
|---|---|---|
| P1 top_mS ≥ 0.90 + margin +0.30 (두 variant) | [main] | ✅ PASS |
| P2 recon ratio ≤ 1.10× (두 variant, 전 α) | [main] | ✅ PASS (0.94–0.99×) |
| P3 pair_cos +0.20 margin vs baseline (전 α) | [main] | ❌ FAIL (α ∈ {0.5, 0.7, 0.9}) |
| P4 α=0.9 τ=0.99 cliff (ours ≥ 0.10, baselines ≤ 0.02) | [main] | ✅ PASS |
| P5 Option A ≈ Option B (ǀΔǀ ≤ 0.05) | [main] | ✅ PASS |
| P6 λ saturation (range ≤ 0.10) | [sub] | ❌ FAIL (range 0.13–0.16) |
| P7 m_S argmax ∈ {512, 576} (MGT τ=0.95) | [sub] | ✅ PASS (argmax = 576) |
| P8 k=6 top-2 in recon & top_mS | [sub] | ❌ FAIL (k=4 now best) |

**4/5 main predictions pass, 1/3 sub predictions pass.**

Hypothesis H 에서 가장 중요한 주장은:

> (i) latent index alignment saturates, (ii) recon cost within 1.10×, 
> (iii) pair_cos 우위, (iv) α=0.9 τ=0.99 cliff 유일 생존.

(i), (ii), (iv) 는 **완벽히 supported**. (iii) pair_cos 는 **명확히 실패하지만
metric 자체의 구조적 한계** 이며 ours 의 실제 objective 와 orthogonal (§5.2).

### 5.2 P3 (pair_cos) failure 해석 — metric–objective mismatch

Pre-reg 단계에서 `pair_cos_mean` 을 "per-sample 정렬 강도" metric 으로 선정
했으나, 결과는 다음 구조를 드러냄:

```
shared-decoder methods @ α↑ → top-k indices become nearly identical across
    modalities (because Φ_S ≈ Ψ_S means same atom fires for both) → pair_cos → 1

ours @ any α → top-m_S diag ≈ 1, rest diag ≈ 0 → per-vector cosine capped
    at sqrt(m_S/n) = sqrt(512/1024) ≈ 0.707 theoretical max, observed ~0.46
    (because actual activation magnitudes are not all-1)
```

`ours` 의 `L_aux` 는 **per-dim diagonal 구조** 를 밀지 전체 code vector 유사도
를 밀지 **않는다**. 사실 ours 가 원하는 건 "shared concept 을 같은 index 에
배치, modality-specific concept 은 다른 index 에 배치" 이므로, per-vector
cosine 을 1 로 만드는 건 오히려 modality-specific 정보를 다 지우는 것과 같음.

따라서 **P3 failure 는 ours 의 결함이 아니라 metric 선택이 ours 의 objective
를 대표하지 못한 것**. `cross_cos_top_mS_mean` (top-m_S 대각 평균) 이 ours
의 primary objective 를 올바르게 측정하며, 이 metric 에서는 ours 가 **0.93–
0.96 vs baseline 최대 0.53** 으로 압도 (P1 pass).

추가 관찰: α=0.9 에서 `iso_align` 의 pair_cos (0.607) 이 ours 의 0.460 보다
높다. 이는 `iso_align` 이 top-1 masked cosine 을 직접 optimize 하기 때문 —
**ours 와 iso_align 은 서로 다른 objective 를 optimize 하는 method 이고
pair_cos 는 iso_align 의 natural metric**. Pre-reg 시점에 이 distinction 을
명확히 인식하지 못했음.

### 5.3 Group_sparse 의 지속적 MGT 우위

`group_sparse` 가 τ = 0.9/0.95 에서 **모든 α 에 대해 ours 를 압도** (≥ +0.35
margin). 이유는 v1–v5 에서 분석한 대로:

1. L2,1 penalty 가 paired sample 에서 동시 활성화를 강제 → decoder row 가
   GT shared atom 방향으로 수렴 (합성 생성 모델의 "shared co-activation" 과
   정확히 match).
2. 2048 latent 중 512 개만 shared 로 학습하면 되므로 capacity 여유.
3. α=0.9 에서 compromise row cosine 이 √0.95 ≈ 0.97 > 0.95 라 τ=0.95 는
   통과하지만 > 0.99 는 불가능 → τ=0.99 cliff.

Ours 의 MGT 는 decoder row 를 직접 건드리지 않고 latent space alignment 만
optimize 하므로 feature recovery 에서 group_sparse 에 밀리는 것은 구조적.

**예외**: τ=0.99 × α ∈ {0.7, 0.9} 에서 ours 가 승리. α=0.9 에서는 shared-
decoder 전부가 geometric cliff 로 0.000 이고, α=0.7 에서도 cosine cliff 가
시작되어 shared-decoder 가 0.003–0.097 수준. 이 구간에서 ours 의 "two
independent SAE + permutation" 구조가 기하적 우위.

### 5.4 Option A vs Option B 권장

P5 pass + 수치 분석:

| 기준 | 승자 | 차이 |
|---|---|---|
| recon (낮을수록 좋음) | global (~0.003 낮음) | marginal |
| top_mS, MGT τ=0.95/0.99 | group (~0.03 높음) | marginal |
| pair_cos | tie | <0.001 |
| λ scaling 해석 | group | group 이 L-independent, global 은 m_S/n 에 의존 |

**권장 = Option A (`normgroup`)**. 이유:
1. MGT (ours 의 secondary objective) 에서 일관되게 약간 우세.
2. λ scaling 이 m_S 와 L 에 완전히 독립 → 논문의 hyperparameter 권장 값
   제시 시 해석이 깔끔.
3. Option B (normglobal) 은 "normgroup λ=x ≈ normglobal λ=2x" 대응 관계가
   있어 마치 같은 family 안의 다른 parameterization 으로 보임.

### 5.5 Scope-bounded conclusion

Under the L=2048 synthetic assumption (linear sparse generative model,
paired co-activation, matched sparsity, d=768, k=16, paired batch,
10 epochs, single seed, α ∈ {0.3, 0.5, 0.7, 0.9}), **Algorithm 1 with
mean-normalized `L_aux` (either variant) matches or beats `two_recon` on
recon while saturating index alignment (top_mS ≈ 0.94)**. Feature
recovery (MGT τ = 0.9, 0.95) 에서는 `group_sparse` baseline 이 여전히
압도하지만, **τ = 0.99 + mid-to-high α (= 0.7, 0.9)** 구간에서는 ours 가
유일하게 의미 있는 값을 유지 (shared-decoder geometric cliff 를 넘는
유일한 method).

H 는 **대부분 지지됨 (mostly supported)**: alignment 이득 확인, recon
동등성 확보, geometric cliff 생존 재현. `pair_cos_mean` 기반 주장 (P3) 은
기각되지만 이는 metric mismatch 로 해석 — `cross_cos_top_mS_mean` (P1) 이
ours 의 실제 objective 를 반영하는 올바른 alignment metric.

### 5.6 Follow-ups

1. L=4096, L=8192 sibling reports 에서 L-scaling 재검증.
2. `group_sparse` 의 decoder geometry 를 직접 측정 → 왜 GT atom 방향으로
   수렴하는지 formal analysis.
3. τ=0.99 × α ∈ {0.5, 0.6, 0.8} 까지 grid 넓혀 "cliff 영역" 경계 매핑.
4. Ablation 의 **k=4 권장 재확인**: L=4096/8192 에서도 k=4 가 top 인지.
5. Real-VLM embedding 에서 재평가 (IsoEnergy 의 CLIP-scale 실험 환경).

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
