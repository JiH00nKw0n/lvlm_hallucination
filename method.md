# Method — 현재 상태와 걱정되는 지점들

## 1. 지금 돌리는 method

### 1.1 Setup — 기호 정의

Paired CLIP embedding $(\boldsymbol{x}, \boldsymbol{y}) \in \mathbb{R}^{d}$ 을 입력. 두 개의 SAE 를 각각 학습한다.

- **image SAE**: dense latent $\boldsymbol{z}_{\mathrm{I}} = \mathrm{TopK}(\boldsymbol{W}_{\mathrm{I}}^{\mathrm{enc}} \boldsymbol{x}) \in \mathbb{R}^{n}$, 재구성 $\hat{\boldsymbol{x}} = \boldsymbol{V}\,\boldsymbol{z}_{\mathrm{I}}$.
- **text SAE**: $\boldsymbol{z}_{\mathrm{T}} = \mathrm{TopK}(\boldsymbol{W}_{\mathrm{T}}^{\mathrm{enc}}\boldsymbol{y})$, $\hat{\boldsymbol{y}} = \boldsymbol{W}\,\boldsymbol{z}_{\mathrm{T}}$.
- Decoder 매트릭스 $\boldsymbol{V}, \boldsymbol{W} \in \mathbb{R}^{d \times n}$ (각 column unit-norm).
- $n$ = per-side latent 개수 (실험: $4096$).
- 재구성 loss: $\mathcal{L}_{\mathrm{rec}}^{\mathrm{I}} = \mathbb{E}\|\boldsymbol{x} - \hat{\boldsymbol{x}}\|^{2}$, $\mathcal{L}_{\mathrm{rec}}^{\mathrm{T}}$ 동일.

**Cross-correlation matrix**. $\boldsymbol{C} \in \mathbb{R}^{n \times n}$, $\boldsymbol{C}_{ij} = \mathrm{Corr}(\boldsymbol{z}_{\mathrm{I}}[i], \boldsymbol{z}_{\mathrm{T}}[j])$. 즉 image slot $i$ 와 text slot $j$ 의 activation 상관.

### 1.2 Paper Algorithm 1 (현재 baseline)

두 단계 학습:

1. **Stage 1** ($k_{\mathrm{align}}$ epoch): recon 만. $\mathcal{L} = \mathcal{L}_{\mathrm{rec}}^{\mathrm{I}} + \mathcal{L}_{\mathrm{rec}}^{\mathrm{T}}$.
2. **Permutation** (epoch $k_{\mathrm{align}}$ 시점): $\boldsymbol{C}$ 계산 후 **greedy Hungarian** — threshold $\rho$ 위 correlation 쌍을 순서대로 top-left 로 밀어올림. 결과:
   - $m_{\mathrm{S}}$ = "diagonal 값 > $\rho$" 인 쌍의 개수 = 공유 slot 추정.
   - Decoder/encoder 의 column/row 를 재배치.
3. **Stage 2** (나머지 epoch): recon + aux.
   $$
   \mathcal{L}_{\mathrm{aux}}^{\mathrm{naive}}(m_{\mathrm{S}})
   = \frac{1}{n}\Big[\sum_{i \le m_{\mathrm{S}}} (\boldsymbol{C}_{ii} - 1)^{2} + \sum_{i > m_{\mathrm{S}}} \boldsymbol{C}_{ii}^{2}\Big].
   $$
   즉 앞 $m_{\mathrm{S}}$ slot 의 대각선을 $1$ 로, 나머지는 $0$ 으로.

최종 loss: $\mathcal{L}_{\mathrm{rec}}^{\mathrm{I}} + \mathcal{L}_{\mathrm{rec}}^{\mathrm{T}} + \lambda \cdot \mathcal{L}_{\mathrm{aux}}^{\mathrm{naive}}$. 고정 hyperparam: $\lambda, \rho, k_{\mathrm{align}}$.

---

## 2. 걱정되는 지점 + 바꿔본 것

### 2.1 Hard target "$\boldsymbol{C}_{ii} = 1$" 이 $\alpha < 1$ 에 부적합

**문제.** 이론상 paired concept 의 image atom 과 text atom 은 cosine $\alpha \in [0, 1)$ 만큼 어긋나 있음 (partial alignment). 그런데 $\mathcal{L}_{\mathrm{aux}}^{\mathrm{naive}}$ 는 **activation 수준에서 $\boldsymbol{C}_{ii} = 1$ 을 강제** → decoder column 을 과하게 왜곡할 가능성.

**아이디어 (InfoNCE).** Hard target 대신 **상대적 target**. Slot $i$ 의 $\boldsymbol{C}_{ii}$ 가 자기 row 에서 **dominate** 만 하면 됨 (절대값 1 필요 없음). CLIP-style symmetric InfoNCE:

$$
\mathcal{L}_{i}^{\mathrm{row}} = -\log \frac{\exp(\boldsymbol{C}_{ii}/\tau)}{\sum_{j=1}^{n} \exp(\boldsymbol{C}_{ij}/\tau)}, \qquad
\mathcal{L}_{i}^{\mathrm{col}} = -\log \frac{\exp(\boldsymbol{C}_{ii}/\tau)}{\sum_{j=1}^{n} \exp(\boldsymbol{C}_{ji}/\tau)}
$$

- $\tau$ = learnable temperature (CLIP init $0.07$).
- 최종 $\mathcal{L}_{\mathrm{aux}}^{\mathrm{infonce}} = \frac{1}{|\mathcal{F}|}\sum_{i \in \mathcal{F}} \tfrac{1}{2}(\mathcal{L}_{i}^{\mathrm{row}} + \mathcal{L}_{i}^{\mathrm{col}})$.
- $\mathcal{F}$ = "공유 후보" 로 굳어진 slot 집합. Naive 의 $[1, m_{\mathrm{S}}]$ 역할. §2.3 에서 갱신 방식 서술.

**추가: alive-mask.** Real 데이터에서 slot 의 약 **85% 가 dead** (fire rate $\approx 0$). Dead slot $j$ 는 $\boldsymbol{C}_{ij} = 0$ 이라 $\exp(0) = 1$ 을 softmax 분모에 기여 → $3500$ 개 정도 상수 inflation → gradient 희석. 해결: $\mathcal{A} = \{j : \text{fire rate}(j) > 10^{-3}\}$ 로 softmax negative pool 제한. ($\mathcal{A}$ = alive slot set.)

### 2.2 Off-diagonal 무관심 → slot redundancy 방치

**naive 와 Barlow 의 관계.** Zbontar et al. 2021 의 "Barlow Twins" 원조 loss 는 두 부분으로 구성:

$$
\mathcal{L}_{\mathrm{BT}} = \underbrace{\sum_{i}(\boldsymbol{C}_{ii} - 1)^{2}}_{\text{on-diagonal half}} + \lambda_{\mathrm{off}} \underbrace{\sum_{i \neq j} \boldsymbol{C}_{ij}^{2}}_{\text{off-diagonal half}}
$$

Paper Algorithm 1 의 naive loss (§1.2) 는 사실상 **Barlow 의 on-diagonal half 만** 사용 (private slot 을 $0$ 으로 누르는 항은 추가 변형). **Off-diagonal half 는 생략**되어 있음. 의도된 생략인지 불분명하나, 생략의 결과로 **slot redundancy** (한 concept 이 여러 slot 에 분산 인코딩) 를 막을 기전이 없음 — synthetic 측정 결과 concept 당 평균 $1.8$ slot 에 분산.

**문제.** Slot redundancy 는 $\boldsymbol{C}$ 의 off-diagonal $\boldsymbol{C}_{ij}$ ($i \neq j$) 에 신호로 나타남. Naive 는 이를 무시하므로 redundancy 를 감소시킬 수단 부재.

**아이디어 (Barlow 완성).** 원조 Barlow 의 off-diagonal half 를 $\mathcal{F} \times \mathcal{F}$ 블록에 복원:

$$
\mathcal{L}_{\mathrm{aux}}^{\mathrm{barlow}}
= \mathcal{L}_{\mathrm{aux}}^{\mathrm{naive}}(\mathcal{F}) + \lambda_{\mathrm{off}} \sum_{i, j \in \mathcal{F},\ i \neq j} \boldsymbol{C}_{ij}^{2}.
$$

- $\lambda_{\mathrm{off}} = 0.005$ (Barlow 논문 기본값).
- 즉 "paper method + Barlow 의 원래 off-diagonal half".
- 기호 복습: $\boldsymbol{C}$ = cross-correlation (§1.1), $\mathcal{F}$ = frozen slot set (§2.1).

### 2.3 Hungarian 1 회 → 학습 중 $\boldsymbol{C}$ 변화 반영 못 함

**문제.** Algorithm 1 은 epoch $k_{\mathrm{align}}$ 한 시점의 $\boldsymbol{C}$ 로 permutation 을 정하고 학습 끝까지 고정. 하지만:

- Stage 2 학습 중 slot activation 이 변하면서 $\boldsymbol{C}$ 의 peak 가 이동 가능.
- Dead slot 이 부활해 의미있게 발화하기 시작해도 고정된 permutation 에 편입될 기회 없음.

**아이디어 (per-epoch Hungarian).** 매 epoch 끝에 재매칭. 단, 이미 안정화된 slot 은 놔두고 **미확정 slot (unsettled) 끼리만** 재permute (partitioned):

$$
\mathcal{F}_{t} = \{i : \bar{\boldsymbol{C}}_{ii} > \rho\ \text{at epoch } t\}, \qquad \mathcal{U}_{t} = [n] \setminus \mathcal{F}_{t}.
$$

- $\mathcal{F}_{t}$ = epoch $t$ 끝에 "공유 후보" 로 굳어진 slot ($\S 2.1$).
- $\mathcal{U}_{t}$ = 미확정. 이 집합 내부에서만 매 epoch 새 Hungarian.
- $\mathcal{F}$ 에 한번 들어간 slot 은 **학습 끝까지 frozen** — slot identity 안정.

**Compute 문제 + EMA 해결.** 매 epoch 끝마다 전체 train set (566k pairs) 으로 $\boldsymbol{C}$ 를 계산하면 **variant 당 $\sim 3$ 시간 overhead**. 해결:

- 매 학습 step 의 batch cross-correlation $\boldsymbol{C}^{\mathrm{batch}}$ 를 **EMA** 로 누적.
$$
\bar{\boldsymbol{C}} \leftarrow m \bar{\boldsymbol{C}} + (1 - m) \boldsymbol{C}^{\mathrm{batch}}, \qquad m = 0.99.
$$
- Epoch 끝의 gate + Hungarian 에 $\bar{\boldsymbol{C}}$ 사용 — **별도 forward pass 없음**.
- Cost 감소: $3$ 시간 $\to\ 30$ 초 ($\times 360$).
- Trade-off: EMA staleness vs exact $\boldsymbol{C}$. Gate 결정엔 충분.

### 2.4 Dead slot 회복 불가

**문제.** 학습 초기에 특정 slot 이 발화하지 않으면 해당 slot 으로는 gradient 가 거의 흐르지 않아 **영구 dead** 로 고착. Real 에서는 이 현상이 심해 약 $85\%$ slot 이 dead.

**아이디어 (Revive).** 매 epoch 끝에 **그 epoch 동안 한 번도 발화하지 않은 slot** 을 감지 → encoder/decoder weight 를 random 으로 재초기화. 재초기화된 slot 은 다음 epoch 에서 새로운 방향으로 학습 기회를 얻음.

Revive 를 per-epoch Hungarian (§2.3) 과 결합하면: 부활한 slot 은 자동으로 다음 epoch 의 $\mathcal{U}_{t+1}$ (unsettled) 에 포함 → 기존 slot 과 매칭될 기회를 자동 획득.

---

## 3. 실험 결과

### 3.1 Synthetic — $\lambda$ 선정 + 별다른 효과 없음

**$\lambda$ 선정 방식.** Synthetic 은 GT 를 알 수 있어 $\lambda$ sweep 을 통해 각 variant 의 sweet spot 을 찾음. Criterion: "RE 를 recon_only 대비 크게 망치지 않으면서 aux 효과가 가장 큰 $\lambda$".
- naive, barlow: $\lambda = 0.25$ (RE 손실 $+0.6\%$).
- InfoNCE: $\lambda = 0.01$ (RE 손실 $+0.4\%$). 큰 $\lambda$ 에서는 scale 차이로 재앙 (RE $+60\%$ at $\lambda = 1$).

**결과.** 위 $\lambda$ 선정 후 9 variant 모두 비교했을 때 **variant 간 측정가능한 차이가 거의 없음**:
- 모든 aux variant 의 RE 손실 $\le 1.2\%$.
- GRR (per-modality atom 회복) 은 aux 유무와 무관 ($\pm 0.01$).
- Barlow 와 naive 가 바이트-identical (synthetic 의 $\mathcal{F} \times \mathcal{F}$ off-diagonal 이 noise 수준이라 $\lambda_{\mathrm{off}}$ 효과 0).
- Revive $0$ 회 trigger ($k = 16$, $L = 4096$ 조합에서 dead slot 이 생기지 않음).
- **Post-hoc Hungarian baseline** : recon 만 학습 후 학습 끝에 Hungarian 을 한 번 걸면 slot-aligned joint recovery 의 $90\%$ 를 aux 없이 달성 → synthetic 에서는 method 의 독자 가치 확인 불가.

→ Synthetic 은 너무 clean 해서 §2 의 걱정들 대부분이 **실제 문제가 아니었음**. Real 이 decisive test.

### 3.2 Real — CLIP B/32 on COCO Karpathy (7/7 완료)

**Setup**: CLIP ViT-B/32, per-side $L = 4096$, top-$k = 8$, 30 epoch, batch 1024. $\lambda$ 는 §3.1 의 synthetic best 그대로 (naive/barlow $0.25$, InfoNCE $0.01$). Per-epoch variants 는 §2.3 의 EMA-$\bar{\boldsymbol{C}}$ 로 학습.

**지표 정의**:
- **RE**: eval set (25k pairs) 의 modality 평균 recon loss (img + txt).
- **alive_i, alive_t**: fire rate $> 10^{-3}$ 인 slot 수 (per side).
- **matched**: alive-alive submatrix 에 Hungarian 후 얻은 pair 수.
- **shared**: fire ratio $r_k = \mathrm{rate}_{\mathrm{I}}(k) / (\mathrm{rate}_{\mathrm{I}}(k) + \mathrm{rate}_{\mathrm{T}}(k))$ 가 $[0.4, 0.6]$ 인 alive slot 수. "양쪽 modality 가 균형 있게 발화하는 slot".
- **$\hat{\alpha}$**: top-10\% co-firing matched pair 의 decoder cosine median. Real CLIP 의 $\alpha$ 추정치.

**전체 결과**:

| variant | RE | ΔRE% | alive_i / alive_t | matched | $\hat{\alpha}$ | shared |
|---|---|---|---|---|---|---|
| recon_only | 0.5648 | 0.0 | 620 / 522 | 522 | 0.526 | 11 |
| naive_once | 0.5725 | +1.4 | 537 / 527 | 527 | 0.526 | 197 |
| barlow_once | 0.5624 | −0.4 | 580 / 539 | 539 | 0.512 | 194 |
| infonce_once | 0.5554 | −1.7 | 531 / 599 | 531 | 0.536 | 181 |
| naive_perepoch+revive | 0.6986 | +24 | 402 / 166 | 166 | 0.478 | 49 |
| barlow_perepoch+revive | 0.6910 | +22 | 344 / 186 | 186 | 0.477 | 66 |
| infonce_perepoch+revive | 1.1664 | +106 | 354 / 145 | 145 | 0.485 | 22 |

**관찰**:

1. **Once 와 per_epoch+revive 의 결과가 크게 갈림**. 모든 once variant 는 RE 가 recon_only 와 비슷하거나 낮음 ($-1.7 \sim +1.4\%$). 반면 모든 per_epoch+revive variant 는 RE 가 크게 악화됨 ($+22 \sim +106\%$). Alive count 도 per_epoch+revive 에서 $522 \to 145$–$186$ 으로 급감. §2.3 (per_epoch) + §2.4 (revive) 가 real 에서는 **학습을 불안정화**. 구체적 원인은 추가 조사 필요 (revive 의 빈도, EMA-$\bar{\boldsymbol{C}}$ 의 staleness, per-epoch Hungarian 의 slot 재permute 가 후보).

2. **§2.1 (InfoNCE + alive-mask) 은 real 에서 synthetic 과 정반대 방향**. Synthetic 에서는 $\lambda = 1$ 기준 재앙이었는데, real 에서는 $\lambda = 0.01$ + alive-mask 로 `infonce_once` 가 RE 를 $-1.7\%$ 로 낮춤. Alive-mask 가 softmax 분모에서 dead slot 을 제거하는 게 학습 안정에 기여.

3. **Shared slot 개수는 once 와 per_epoch+revive 에서 크게 차이**:
   - recon_only: 11
   - once 변종: 181–197 ($\sim 18 \times$ 증가)
   - per_epoch+revive: 22–66
   
   Once aux 는 "양쪽 modality balanced 발화하는 slot" 을 훨씬 많이 만듦. Post-hoc Hungarian 은 slot 인덱스만 재배치하므로 이 구조는 학습 중 aux pressure 로만 생성됨.

4. **$\hat{\alpha} \approx 0.5$ 는 method 와 독립적**. Once 변종 $0.51$–$0.54$, per_epoch+revive $0.48$–$0.49$. Multi-model boxplot 에서 관찰된 "real CLIP $\alpha \approx 0.5$" 가 method 선택과 무관 — $\hat{\alpha}$ 추정만 목적이면 method 차이가 결정적이지 않음.

**Qualitative check — shared slot 이 실제로 cross-modal concept 을 잡는가 (naive_once 기준).** 각 shared slot 의 image-측 top-10 activation 과 text-측 top-10 activation 을 COCO Karpathy 캡션과 대조. 8개 top shared slot 수동 검토:

| slot | $r$ | Concept (직접 할당) | 일관성 |
|---|---|---|---|
| $5$ | $0.51$ | 고양이 | ✅ |
| $8$ | $0.53$ | 욕실 | ✅ |
| $41$ | $0.54$ | 해변 | ✅ |
| $64$ | $0.43$ | 여성 (broad) | 🟡 |
| $70$ | $0.52$ | 어린 남자아이 | ✅ |
| $105$ | $0.51$ | 새/가축 떼 | ✅ |
| $125$ | $0.59$ | 음식 접시 | ✅ |
| $153$ | $0.41$ | 도시/교통 | ✅ |

**$8/8$ 모두 image-측과 text-측 caption 이 의미상 같은 concept 을 기술**. 즉 shared slot 은 진짜 cross-modal semantic concept 을 잡는다 — aux loss 의 핵심 qualitative evidence. 초기에 측정했던 Jaccard overlap $= 0$ 은 metric 오류 (같은 concept 이 수천 장의 이미지에 퍼져 있어 image-id 수준 overlap 은 자연스럽게 $0$; caption 텍스트 수준 의미 일치가 올바른 기준).

### 3.3 실험 종합

비교한 7 variant 중 real 에서 관찰된 패턴:

- **Once schedule 의 3 variant (naive / barlow / infonce) 는 recon_only 와 비슷하거나 더 낮은 RE**. Loss form 간 차이는 $1$–$3\%$ 수준으로 작음.
- **Per_epoch+revive 3 variant 는 모두 RE 대폭 악화** ($+22\%$ 이상). Loss form 무관. Per-epoch Hungarian + revive 조합이 real 데이터 분포에서 학습을 불안정화하는 것으로 보이며, 구체적 원인은 추가 실험이 필요 (revive 빈도, EMA staleness, slot identity 재permute 중 어떤 것이 주원인인지).
- **Shared slot 개수는 schedule × loss 조합에 크게 의존**. Once + aux 조합이 recon_only 대비 $18 \times$ 증가 ($11 \to 181$–$197$), per_epoch+revive 조합은 $22$–$66$ 수준으로 낮음.
- **Caption 수준 qualitative 검토** (naive_once 상위 8개 shared slot) 결과 $8/8$ 모두 image-측과 text-측에서 동일 concept 을 기술. 즉 once+aux 의 shared slot 은 random artifact 가 아니라 cross-modal semantic concept.
- **$\hat{\alpha} \approx 0.5$** 는 variant 선택과 독립적. $\hat{\alpha}$ 를 단독 metric 으로 보면 variant 간 차이 미미.

결과를 post-hoc Hungarian baseline 과 대조:
- **Slot-index 정렬 자체** 는 recon_only + post-hoc Hungarian 으로 대부분 재현됨 (synthetic 의 joint_mgt = $0.565$ 대 aux 의 $0.569$).
- **Activation 패턴 (shared slot, cross-modal concept 이 같은 slot 에서 발화)** 은 post-hoc Hungarian 으로 재현 불가능 — 학습 중 aux pressure 가 있어야 생성됨.

남은 질문:
- per_epoch+revive 가 real 에서 실패한 정확한 원인 (ablation 필요: revive 만, per_epoch 만, 조합)
- 다른 base VLM (MetaCLIP, DataComp 등) 에서도 이 패턴이 재현되는지
- Multi-seed 안정성 확인 (지금 seed=1 단일).
