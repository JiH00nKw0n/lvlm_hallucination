# Paper Draft Skeleton

> Living skeleton. Each section holds only core bullets; prose is filled in later.

---

## 0. Logical spine (original, for reference)

1. **Setup**: VLM joint embedding은 poly-semantic, 해석 어려움.
2. **Framework**: SAE가 sparse + monosemantic direction으로 decompose. 기존 VLM SAE는 "shared concept = 단일 direction" 가정.
3. **Anomaly**: 실제로는 shared concept도 image/text 간 다른 direction에 실림. 기존에는 이를 "modality split"(다른 latent coord)로 해석하고 activation alignment로 해결 시도.
4. **Reframing [핵심]**: root cause는 direction mismatch (DDM). Modality split은 symptom. Activation alignment로는 구조적 해결 불가.
5. **Empirical**: 5 VLM diagnostic → shared concept도 partial alignment (α<1). [Fig: multi_model_boxplot]
6. **Theory + Synthetic**: α<1이면 shared decoder가 principal direction으로 collapse (Thm 2). Separated는 보존 (Thm 1). Synthetic에서 α sweep으로 검증. [Fig: shared_vs_separated]
7. **Method**: Separated SAE + post-hoc alignment.
8. **Real experiments**: 5 VLM에서 diagnostic + method 비교.

---

## Figure 1 (DDM overview)

`outputs/fig1_ddm_overview.pdf` with caption in `outputs/fig1_ddm_overview_caption.tex`.

---

## 1. Introduction

- VLM 임베딩은 poly-semantic; SAE가 monosemantic dictionary direction을 분해한다.
- Prior work는 공유 concept이 single modality-agnostic direction에 놓인다고 가정하고, 그로부터 벗어나는 현상(modality split)을 실패로 보고 alignment 목적을 추가해왔다.
- (Fig 1) 실제 VLM에서 공유 concept조차 modality마다 서로 다른 dictionary direction으로 표현된다 (**Dictionary Direction Mismatch, DDM**). 이 기하적 사실 아래에서 modality split은 자연 결과이며, 강제 정렬은 modality-specific 구조를 붕괴시킨다.
- 본 논문은 이 관찰을 이론적으로 formalize (Section 3), synthetic 실험으로 검증 (Section 4), direction을 보존하는 **post-hoc alignment** 방법을 제안 (Section 5)하고 real VLM에서 검증 (Section 6)한다.

---

## 2. Problem Setup

- 임베딩은 sparse latent code $\vz \in \mathbb{R}^n_+$와 modality-specific dictionary $\vPhi, \vPsi \in \mathbb{R}^{d\times n}$를 통해 $\vx = \vPhi \vz$, $\vy = \vPsi \vz$로 생성. $\phi_i \ne \psi_i$ 허용 (prior work와 다른 점).
- Sparsity $\Pr(z_i \ne 0) = s$ ($s \to 1$ regime) + 좌표별 독립.
- SAE $\vV, \vW \in \mathbb{R}^{d\times m}$; $\vV = \vW$이면 shared SAE. Activation $\sigma$는 top-1 (analytical convenience).

---

## 3. Theoretical Analysis

- **Main Lemma (Theorem 1)**: $s \to 1$ regime에서 $\loss_{\mathrm{rec}}(\vV; \vPhi) = s^{n-1}(1-s)\sum_i \mu'_i \|\phi_i - \vV\sigma(\vV^\top \phi_i)\|_2^2 + o(1-s)$. $m \ge n$일 때 $\hat{\vV} = [\vPhi\;\vzero]\vP$가 global minimizer이며 recon loss $= 0$ (permutation까지 exact recovery).
- **Compressed-Capacity Theorem (Theorem 2, $m < n$)**: 최적 SAE column은 partition $\{\mathcal A_j\}$에 대한 $\sum_{i \in \mathcal A_j} \mu'_i \phi_i\phi_i^\top$의 top eigenvector. → 한 column이 두 atom을 담당하면 두 atom의 1st principal direction (동일 가중치의 경우 bisector).
- **Two-to-One Lemma (Proposition 1)**: $\loss_{\mathrm{rec}}(\vV;\vPhi) + \loss_{\mathrm{rec}}(\vV;\vPsi) = \loss_{\mathrm{rec}}(\vV;[\vPhi\;\vPsi]) + o(1-s)$. 다중 modal 분석을 concat 문제로 환원.
- **Alignment-Loss Thresholds (Propositions 2–3)**: group-sparse 및 cosine-alignment 각각에 대해 critical $\lambda^\star(\rho)$ 존재. $\lambda < \lambda^\star$: separate 해가 최적; $\lambda > \lambda^\star$: **bisector merge**가 최적. 강한 정렬은 phase transition을 거쳐 direction을 붕괴시킨다는 formal statement.

---

## 4. Synthetic Experiments

- **Setup**: GT atom이 관측 가능한 generative model ($d=256$, $n_S=1024$, α 제어). Top-$k$ SAE 학습 후 RE / GRE / Embedding-SIM / Feature-SIM 측정.
- **Fig: Shared vs Separated across α (`fig1_v3`)** — Shared SAE는 $\alpha < 1$에서 CR↑, RE↑, GRE↑; Separated SAE는 α 전 구간에서 recovery flat. **Theorem 2의 bisector 예측을 직접 empirical 검증**.
- **Fig: λ sweep at α=0.5 (`fig2_v3`)** — Iso-Energy / Group-Sparse 모두 특정 λ 이후 recon / GRE cliff (Propositions 2–3의 $\lambda^\star$ phase transition과 일치). **Post-hoc Matching은 λ와 무관하게 RE/GRE를 baseline 수준 유지하면서 E-SIM / F-SIM을 모든 방법 중 최고**.

---

## 5. Method: Post-hoc Alignment

- **Stage 1 (independent training)**: 각 modality에서 $\loss_{\mathrm{rec}}$만으로 SAE 학습. Main Lemma에 의해 GT atom 복원은 이미 최적.
- **Stage 2 (alignment)**: train-set co-activation correlation $\vC$ 위에서 alive-restricted assignment로 permutation $\pi$를 구하고, shared slot 수 $m_S$는 threshold $\rho$를 넘는 diagonal 수로 자동 결정.
- **Development directions** (후보 — 최소 하나 이상 포함):
  - (a) **Threshold $\rho$ 기반 shared/specific slot 분리** — $\rho$보다 큰 diagonal만 shared로 선언, 나머지는 modality-specific으로 보존 (Algorithm 1)
  - (b) **Hungarian 후 aux-loss fine-tune** — shared slot만 correlation 1 쪽으로 당기고, modality-specific slot은 0 쪽으로 분리하는 보조 목적 추가
  - (c) **Matching score weighting** — correlation과 decoder cosine의 weighted combination으로 activation 강도와 direction 유사도 모두 반영

---

## 6. Real Experiments

- **Setup**: 5 VLMs (CLIP / MetaCLIP / OpenCLIP / MobileCLIP2 / SigLIP2). 각 모델에 Shared SAE / Separated SAE / Iso-Energy / Group-Sparse / **Post-hoc Matching (ours)** 학습 (동일 hyperparameter).
- **COCO Cross-Modal Retrieval**: Karpathy test split, sparse code 기반 image↔text retrieval R@1 / R@5 / R@10. Sparse code로도 의미 구조가 보존되는지 + method별 비교.
- **ImageNet-1K Image Classification**: class prototype (80 OpenAI templates 평균) sparse code와 val image sparse code의 cosine-nearest로 zero-shot top-1 accuracy. latent purity / class coverage 분석 함께.

---

## 7. Related Work

- Linear representation hypothesis + SAE for interpretability
- Modality gap / modality split 연구
- Multimodal SAE alignment 방법들 (align-regularized variants)

## 8. Conclusion

- DDM은 symptom이 아니라 VLM 임베딩의 기하적 성질
- 강제 정렬 vs direction 보존 + post-hoc matching: 후자가 이론과 실험 모두에서 우위
- Future work: downstream task에서 interpretable steering, 더 나은 matching objective
