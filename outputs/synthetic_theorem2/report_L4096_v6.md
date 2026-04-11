# Theorem 2 / Algorithm 1 — L = 4096 보고서 (v6)

> `.claude/skills/experiment-report.md` 규약. §1–§3 은 v6 결과 전 작성,
> §4 이후는 결과 후 작성. L=2048 sibling report (`report_L2048_v6.md`) 와
> 동일한 v6 hypothesis 를 공유하며, L=4096 에만 해당하는 predictions 를
> 별도 기재한다.

## §0. 한 줄 결론

*(v6 결과 확인 후 작성)*

---

## §1. Background and motivation (shared with L=2048 v6)

`report_L2048_v6.md §1` 의 v5 까지 시사점 + v6 변경점 요약을 동일하게
공유한다. 핵심만 요약:

- v5 까지 ours 의 P2 (recon ratio ≤ 1.10×two_recon) 가 L=2048 에서만 pass
  하고 L=4096/8192 에서 1.32-1.47× 로 fail 했었음. 원인: `L_aux` sum-based
  정규화가 L 에 비례해서 커짐.
- v6 는 `.mean()` 정규화로 L-independent λ 를 만드는 것이 주 목표.
- L=4096 에서의 시험 = "v6 정규화가 L=2048 보다 더 큰 capacity 에서도
  P2 를 회복시키는가?" 가 이 리포트의 중심 질문.

추가로 L=4096 은 **per-SAE latent = 2048** (two_recon, ours 의 경우) 로
L=2048 의 1024 대비 2배 capacity. 이 추가 capacity 가 ours 의 MGT
feature recovery 를 `group_sparse` 쪽으로 밀어주는지 확인.

---

## §2. Hypothesis + pre-registered predictions (L=4096)

### Hypothesis (L-shared, identical to L=2048)

동일하게 H (v6) 을 따른다. Scope 는 latent_size=4096 만 다름.

### P1 [main] — top_mS 우위, 둘 다 variant, 전 α

- **Pass:** both `normgroup`, `normglobal` variants 가 `top_mS` ≥ 0.90
  at every α ∈ {0.3, 0.5, 0.7, 0.9}, AND margin vs max(baseline) ≥ +0.30.
- **Failure scenario:** L=4096 capacity 에서 permutation 품질 저하. 생성
  모델의 shared atom 수 (n_shared=512) 대비 per-SAE latent (2048) 비율이
  커서 many-to-one mapping 이 저절로 생길 수 있음.

### P2 [main] — Ours recon ≤ 1.10× two_recon at L=4096, **both variants**

- **Pass:** ratio ≤ 1.10 at every α AND both aux_norm variants.
- **Fail:** 어느 α × norm 셀이 1.10 초과.
- **Failure scenario:** 정규화가 부족해서 L=4096 에서 λ=1 이 여전히 과도
  (L=2048 v5 의 λ=2⁻⁸ 이 L=4096 에서 1.34× 였던 것과 비슷한 pattern).
- **Intended interpretation:** v6 의 L-independent normalization 이 실제로
  L=4096 에서 P2 를 회복시킴.
- **Alternative explanations:**
  (a) Per-SAE capacity 가 2배라서 two_recon 이 0.025 에 근접하게 수렴한
  반면 ours 는 L_aux 때문에 capacity 를 다 못 써서 0.03 근처. absolute
  gap 이 작아도 ratio 가 1.10 을 넘길 수 있음.
  (b) Option A 와 B 가 다른 수렴점에 도달해서 한쪽은 pass 한쪽은 fail.

### P3 [main] — pair_cos_mean 우위, +0.20 margin

- **Pass:** ours pair_cos − max(baseline) pair_cos ≥ +0.20 at every α.
- **Failure scenario:** L=4096 에서 `iso_align` 이 per-sample cosine 을
  직접 optimize 해서 ours 와 comparable 해지는 경우.

### P4 [main] — α=0.9 τ=0.99 geometric cliff 에서 ours 유일 non-zero

- **Pass:** ours MGT τ=0.99 at α=0.9 ≥ 0.10 AND shared-decoder 4개
  (single/group_sparse/trace/iso) 모두 ≤ 0.02.
- **Failure scenario:** L=4096 에서 shared decoder 가 separate row (Option
  B) 로 switch 하여 cliff 탈출. 특히 `group_sparse` 가 L 증가로 switch
  쉬워질 수 있음 (v5 에서 L=4096 α=0.3 에서 group_sparse τ=0.95 가 0.96
  으로 급상승하는 유사 현상 관찰됨).

### P5 [main] — Option A vs B 차이 ≤ 0.05 per metric

- 세부 내용 L=2048 과 동일. L=4096 에서도 같은 결론 (두 variant 가
  수렴점 유사) 이 유지되는지.

### P6 [sub] — λ saturation (ablation at α=0.9)

- L=2048 과 동일 기준.

### P7 [sub] — m_S argmax

- L=2048 과 동일 기준 (argmax ∈ {512, 576}).

### P8 [sub] — k_align sweet spot

- L=2048 과 동일 기준.

### P9 [main, **L-specific**] — L=4096 capacity benefit for Ours MGT τ=0.9

v5 에서 관찰: ours MGT τ=0.8 이 L=2048 → L=4096 에서 0.68-0.71 → 0.82-0.85
로 크게 상승. v6 의 τ=0.9 에서도 같은 L-scaling benefit 이 있을지 확인.

- **측정:** `(img+txt)/2 MGT τ=0.9` for ours at α=0.9.
- **Pass:** ours ≥ 0.80. (v5 에서 τ=0.8 기준 0.839 였으므로 v6 τ=0.9 에선
  0.80 근방 예상.)
- **Fail:** ours < 0.80.
- **Intended interpretation:** L=4096 capacity 가 feature recovery 에 도움.
- **Alternative explanations:**
  (a) mean-normalized L_aux 가 feature recovery 는 별로 안 돕고 recon
  만 개선할 수 있음.

---

## §3. Experiment design (shared with L=2048 except latent size)

### 3.1 Synthetic generative model

동일 (`SyntheticTheoryFeatureBuilder`, n_shared=512, ...).

### 3.2 SAE configuration — **latent_size = 4096**

- single-decoder: latent = 4096
- two_recon / ours: per-SAE latent = **2048** (= 4096 / 2)
- 나머지 동일.

### 3.3 Methods & sweep

L=2048 v6 와 동일 method 7 개, 동일 α/λ/m_S/k_align sweep. 총 62 trainings.

### 3.4 Hypothesis scope vs experiment scope

L=2048 과 동일, 단 L = **4096**.

---

## §4. Results

*(v6 L=4096 main + ablation 완료 후 작성)*

---

## §5. Verdict

*(결과 확인 후 작성)*

> This experiment is an existence proof. The conclusions hold only within
> the assumptions stated above.

---

## §6. Limitations and Scope

L=2048 report 의 §6 과 동일. 추가:

### 6.5 L=4096 specific

- Per-SAE latent = 2048 이 n_S + n_I = 768 GT atom 보다 훨씬 큼. 즉 ours
  는 n_I private + n_S shared 를 학습하는 데 여유 공간이 많음. 이 capacity
  redundancy 가 ours 의 L_aux 페널티를 상쇄할 수 있는지는 v6 에서 확인.
