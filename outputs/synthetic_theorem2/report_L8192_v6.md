# Theorem 2 / Algorithm 1 — L = 8192 보고서 (v6)

> `.claude/skills/experiment-report.md` 규약. §1–§3 은 v6 결과 전 작성,
> §4 이후는 결과 후 작성. L=2048/L=4096 sibling reports 와 동일 v6
> hypothesis 를 공유.

## §0. 한 줄 결론

*(v6 결과 확인 후 작성)*

---

## §1. Background and motivation (shared with L=2048/L=4096 v6)

`report_L2048_v6.md §1` 참조. L=8192 는 세 L 중 가장 큰 capacity 로, v5
에서 "ours MGT τ=0.8 이 L 증가에 따라 group_sparse 에 수렴" 했던 trend 가
L=8192 에서 극점을 찍는지 확인하는 역할.

v5 observed:
- L=2048: ours MGT τ=0.8 ≈ 0.68-0.71, group_sparse ≈ 0.92-1.00
- L=4096: ours ≈ 0.82-0.85, group_sparse ≈ 0.99-1.00
- **L=8192: ours ≈ 0.94-0.96, group_sparse ≈ 1.00** ← 0.04-0.06 gap 이내

v6 의 τ=0.9 기준으로는 더 strict 한데, 이 L-scaling benefit 이 여전히
유지되는지 확인.

또한 L=8192 는 v5 에서 `ours avg_eval` 이 0.033 → 0.035 로 **거꾸로** 증가
하는 현상 관찰됨 (λ=2⁻⁸ 튜닝이 맞지 않아서). v6 L-independent λ=1 으로
이 현상이 사라지는지가 P2 의 핵심 질문.

---

## §2. Hypothesis + pre-registered predictions (L=8192)

### Hypothesis

L=2048/4096 과 동일 v6 hypothesis.

### P1 [main] — top_mS 우위, 둘 다 variant, 전 α

- Pass/Fail 기준 L=2048 과 동일.
- **Failure scenario (L-specific):** per-SAE latent = 4096 인 상황에서
  m_S = 512 로만 align 요구 → 나머지 3584 개 latent 는 "private" 로
  취급되는데, greedy permutation 이 이 넓은 영역에서 정확한 matching 을
  못할 수도 있음.

### P2 [main] — Ours recon ≤ 1.10× two_recon at L=8192, **both variants**

- Pass 기준 L=2048 과 동일.
- **이 prediction 이 v6 에서 가장 중요**: v5 에서 1.44-1.47× 로 failed.
  v6 의 L-independent λ 가 실제로 효과가 있는지는 L=8192 에서 가장 잘 드러남.
- **Failure scenario:** `L_aux` 가 mean-normalized 로 bounded 됐어도
  per-SAE capacity 4096 내에서 4096 × latent 계산의 grad noise 가 커져서
  ours recon 이 two_recon 에 못 따라갈 수 있음.
- **Alternative explanations:**
  (a) two_recon recon 이 L=4096 에서 이미 포화 (0.024) 라 L=8192 에서도
  0.024 유지 → ours ratio 는 ours 의 recon 이 얼마나 포화 가까이 오느냐에
  달림.
  (b) Option A 와 B 의 dominance 가 L 따라 달라질 수 있음.

### P3 [main] — pair_cos_mean 우위 +0.20 margin

- L=2048 과 동일.

### P4 [main] — α=0.9 τ=0.99 cliff 에서 ours 유일 non-zero

- L=2048 과 동일. v5 에서 L=8192 에도 이 cliff 가 존재함을 확인했으므로
  여기서도 재현 기대.

### P5 [main] — Option A vs B diff ≤ 0.05 per metric

- L=2048 과 동일. L 이 가장 큰 L=8192 에서 difference 가 가장 두드러질
  가능성 → 이 shared prediction 이 L=8192 에서 falsify 될 확률이 가장
  높은 L.

### P6 [sub] — λ saturation

- L=2048 과 동일.

### P7 [sub] — m_S argmax

- L=2048 과 동일.

### P8 [sub] — k_align sweet spot

- L=2048 과 동일.

### P9 [main, L-specific] — MGT τ=0.9 at α=0.9 ≥ 0.85

v5 의 τ=0.8 기준으로는 L=8192 에서 ours ≈ 0.94-0.96 (group_sparse 의
0.998-1.000 에 거의 근접). v6 의 τ=0.9 는 약간 더 strict 하므로 예상값을
0.85 로 낮춤.

- **Pass:** ours MGT τ=0.9 at α=0.9 ≥ 0.85.
- **Fail:** < 0.85.
- **Intended interpretation:** L=8192 에서 ours 가 group_sparse 에 거의
  근접 (± 0.10 이내).
- **Alternative explanations:**
  (a) mean-normalized L_aux 가 feature recovery 를 오히려 약하게 만들 수
  있음 (λ=1 이 sum-based v5 λ=2⁻⁸ 보다 상대적으로 약할 수도).

### P10 [main, L-specific] — Ours-L scaling 유지

- **측정:** ours pair_cos_mean at α=0.9 for three Ls.
- **Pass:** pair_cos_mean at L=8192 > pair_cos_mean at L=2048 (monotone
  increasing with L).
- **Fail:** non-monotonic 또는 decreasing.
- **Intended interpretation:** capacity 증가가 ours 의 per-sample
  alignment 에도 유익.

---

## §3. Experiment design

### 3.1 Synthetic generative model

동일.

### 3.2 SAE configuration — **latent_size = 8192**

- single-decoder: latent = 8192
- two_recon / ours: per-SAE latent = **4096**

### 3.3 Methods & sweep

L=2048/L=4096 v6 와 동일. 총 62 trainings.

### 3.4 Hypothesis scope

동일, 단 L = **8192**, per-SAE latent 4096.

---

## §4. Results

*(v6 L=8192 main + ablation 완료 후 작성)*

---

## §5. Verdict

*(결과 확인 후 작성)*

> This experiment is an existence proof. The conclusions hold only within
> the assumptions stated above.

---

## §6. Limitations and Scope

L=2048/L=4096 report 의 §6 과 동일. 추가:

### 6.5 L=8192 specific

- Per-SAE latent = 4096 은 generative model 의 n_S + n_I = 768 GT atom
  보다 약 5배 크다. Ours 의 L_aux pressure 는 최초 512 개 latent 에만
  걸리므로 나머지 3584 latent 는 "자유" — private feature 를 학습하거나
  dead 될 가능성. Dead code rate 가 크면 per-SAE effective capacity 가 더
  낮아져 recon ratio 에 영향.
- L=8192 ablation 은 total run time 이 가장 길다 (~52 min). 단일 seed 라
  statistical noise 가장 크지만, 방향성만 보는 것이 목표이므로 허용.
