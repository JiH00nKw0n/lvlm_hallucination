# CLAUDE.md — Multimodal SAE Alignment

> Resume할 때 먼저 읽을 것. **이론 + 합성 실험 + 다중 VLM 실증 + 논문 작성**이 한 줄기. 서버 워크플로우와 gotcha는 §6–7에.
> 참고: `src/decoding/` 아래 VCD/VISTA/OPERA 등은 **별개 프로젝트** — 여기서 다루지 않음.

---

## 1. Research Question

VLM의 paired image–text 임베딩에서, *같은 shared concept이라도 두 modality의 feature direction이 정렬되지 않을 수 있다* (**Cross-Modal Feature Heterogeneity**). 우리는 이를 ① 정의하고 ② 실제 VLM에서 보편적으로 나타남을 실증하고 ③ 이로 인해 단일 SAE가 학습 중 features를 merge한다는 이론적 결과(Theorem 2)를 보이고 ④ 이를 회피하면서 cross-modal 대응을 establish하는 *modality-specific SAEs + post-hoc alignment*를 제안.

세 줄기:
1. **Theory** — Theorem 1 (m≥2n: modality split), Theorem 2 (m<2n: geometric merging by closeness), Propositions 3/4 (Iso-Energy / group-sparse aux loss의 phase transition + collapse).
2. **Synthetic** — GT $(\boldsymbol\Phi, \boldsymbol\Psi)$ known인 generative model에서 위 이론적 예측을 직접 검증 (Fig 1: $\cos(\phi_i, \psi_i)$ sweep / Fig 2: $\lambda$ sweep).
3. **Real-VLM** — 5 base VLM (CLIP/MetaCLIP/OpenCLIP/MobileCLIP/SigLIP2)에서 $\cos(\phi_i, \psi_i)$의 proxy로 cross-modal feature heterogeneity가 보편적임을 실증 (Fig multi_density).

---

## 2. Theory (paper §3 + §4)

**Generative model.** $\vx = \vPhi \vz$, $\vy = \vPsi \vz$ — $\vPhi, \vPsi \in \mathbb{R}^{d \times n}$의 column $\phi_i, \psi_i$가 modality별 feature direction. Sparsity $\Pr(z_i=0)=s\to 1$. **Cross-modal feature heterogeneity**: shared concept $i \in [n_S]$에 대해 $\phi_i \neq \psi_i$ 일 수 있음.

**Theorem 1 (`thm:m-large`, m≥2n).** Optimal $\hat\vV = [\vPhi\;\vPsi\;\vzero]\vP$ — SAE column이 modality-specific feature를 그대로 분리해 담음 (modality split의 origin).

**Theorem 2 (`thm:m-small`, m<2n).** Capacity-constrained regime에서 partition $(\sA_1,\ldots,\sA_m)$의 각 group을 sum of $\vM_i$의 top eigenvector로 represent. *Geometrically close features가 merge됨* — 같은 concept의 $\phi_i, \psi_i$가 cos가 작으면 다른 column으로 분리되고, 무관한 features가 가까우면 merge될 수도 있음.

**Propositions 3/4 (`thm:prev:align` / `thm:prev:group`).** Iso-Energy alignment loss / group-sparse loss의 global minimizer 분석. $\lambda < \lambda^\star(\rho)$이면 modality split 유지 ($\hat\vV=[\phi\;\psi]\vP$, recon loss 0); $\lambda > \lambda^\star(\rho)$이면 *cross-modal collapse onto $(\phi+\psi)/\|\phi+\psi\|$* — recon error $(1-\rho)\E[z^2]$. Group-sparse는 $\lambda > \lambda^\dagger$에서 dead-neuron regime 추가.

**Practical implication.**
- **Shared SAE**: heterogeneity 하에서 capacity 제약이 걸리면 features를 merge → reconstruction loss 발생.
- **Iso-Energy / Group-Sparse aux loss**: phase transition 위에서 cross-modal collapse + reconstruction trade-off.
- **Modality-Specific SAEs**: 두 dictionary 분리로 interference 회피, 단 slot correspondence 없음.
- **Ours (paper §4.3, sec:ours)**: Modality-specific SAEs로 학습 후 *post-hoc Hungarian alignment*로 slot correspondence를 establish — 학습 중 추가 loss/hyperparameter 없음.

---

## 3. Synthetic Setup (paper §5.1 / `Validation on Synthetic Data`)

### 3.1 Data

`src/datasets/synthetic_paired_builder.py`의 `SyntheticPairedBuilder`. **현재 표준 ($\cos(\phi_i, \psi_i)$ sweep용)**: $d=256$, $n_S=1024$, $n_I=n_T=512$, $s=0.99$, $\text{Exp}(1)$ 계수, $\sigma_\text{obs}=0.05$, $\varepsilon_\text{max}=0.10$ (Fig 1) / $0.30$ (Fig 2). Independent shared mode — mask만 공유, 계수 독립.

### 3.2 Methods (`run_synthetic_v2.py`)

| Method | label in figs | Single SAE? | Loss |
|---|---|---|---|
| `single_recon` | Shared SAE | Yes | recon |
| `two_recon` | Modality-Specific SAEs | No (두 decoder, 분리) | per-modality recon |
| `iso_align` | Iso-Energy Alignment | Yes | recon $-\beta_\text{IA}\cos(z_I,z_T)$ |
| `group_sparse` | Group-Sparse | Yes | recon $+\lambda_\text{GS}\sum_j\sqrt{z_I^{j2}+z_T^{j2}}$ |
| (post-hoc, offline) | Post-hoc Alignment (Ours) | No (two_recon ckpt + offline Hungarian) | n/a (학습 시 0 hyperparam) |

**Ours는 별도 method가 아님** — `two_recon` 학습 후 `lambda_sweep.yaml`의 plot pipeline에서 offline Hungarian 매칭으로 ESim/FSim 계산. paper §4.3와 일치.

### 3.3 Metrics (Fig 1: 3-panel / Fig 2: 4-panel)

| Metric | 의미 |
|---|---|
| **CR** (Collapse Rate) | GT shared atom 중 best-img/best-txt column이 cos>0.95인 비율 — Theorem 2의 merge 정도 |
| **RE** | $\E[\|x-\tilde x\|^2 + \|y-\tilde y\|^2]/2$ — recon quality |
| **GRE** (untied, 2026-04-24~) | GT atom을 학습된 SAE pipeline (W_enc → ReLU → top-1 → W_dec)에 통과시킨 recon MSE |
| **ESim** | paired eval $(x_n, y_n)$의 $\cos(z_I, z_T)$ — sample sparse code 유사도 |
| **FSim** | GT atom 주입 시 $\cos(z_I, z_T)$ — pure GT probe |

### 3.4 핵심 파일

- `run_synthetic_v2.py` — 모든 method 학습+eval 단일 entry. YAML config로 sweep.
- `configs/synthetic/alpha_1R_2R_L8192_5seeds_coarse.yaml` — Fig 1 ($\cos(\phi_i,\psi_i)$ sweep, 1R vs 2R, 5 seeds, L=8192, k=16).
- `configs/synthetic/lambda_sweep.yaml` — Fig 2 ($\alpha=0.5$ 고정, IA/GS × 5 λ, 5 seeds, k=16).
- **`scripts/plot_alpha_sweep.py`** — Fig 1 plot. x축 `cos(φ_i, ψ_i)`. figsize=(5.5, 1.015) (0.7× 세로).
- **`scripts/plot_lambda_sweep.py`** — Fig 2 plot. RE log scale; GRE log scale + y_min=0.2; ESim/FSim linear. Post-hoc Alignment를 horizontal로 표시. Hungarian on training latents → cache `params/.esim_posthoc_cache.json`.
- `scripts/palette.py` — 공통 색.
- `src/datasets/synthetic_paired_builder.py` — 데이터 generator. (`synthetic_theory_feature.py`는 legacy)
- `src/metrics/synthetic_eval.py` — `compute_gre_top1`, `compute_merged_fraction`.

저장 스키마 (`outputs/theorem2_v2_*/runs/run_<ts>/params/*.npz`): `w_enc_{img,txt}, b_enc_{img,txt}, w_dec_{img,txt}, b_dec_{img,txt}, phi_S, psi_S, phi_I, psi_T, alpha_target, seed, latent_size_img, latent_size_txt, same_model_flag`. Offline 메트릭 재계산용. **`output.save_decoders: true` 필요**.

### 3.5 Run Directory Index

| Dir | Purpose | Config | Methods | Notes |
|---|---|---|---|---|
| `theorem2_v2_1R2R_5seeds_coarse` | Fig 1 | `alpha_1R_2R_L8192_5seeds_coarse.yaml` | 1R, 2R | 5 seeds × 6 α values × 2 methods |
| `theorem2_v2_lambda_sweep` | Fig 2 | `lambda_sweep.yaml` | 1R/2R + IA×5 λ + GS×5 λ | α=0.5, 5 seeds |

---

## 4. Synthetic Results (headline)

**Fig 1** (α sweep, x축 $\cos(\phi_i,\psi_i)$ ∈ {0, 0.2, …, 1.0}, 3-panel):
- **CR** — Shared SAE: monotone increase to 1; Modality-Specific SAEs: ~0 across all α. Theorem 2 (merge by geometric closeness) 직접 실증.
- **RE** — 작지만 단조 gap. Modality-Specific SAEs 전 구간 우수.
- **GRE** — α↑일수록 cross-over: α≈1에서 Shared가 우세, α≲0.7에서는 Modality-Specific 우세.

**Fig 2** (λ sweep, α=0.5, 4-panel RE/GRE/ESim/FSim):
- **Post-hoc Alignment (Ours)**: 모든 λ에서 Modality-Specific SAEs와 RE/GRE 동일 + ESim/FSim 단연 우세.
- **Iso-Energy**: $\beta$ 5 decade 이동해도 메트릭 변화 미미 (inert).
- **Group-Sparse**: $\lambda \times 1 \to \times 4$에서 RE 절벽 (collapse regime).

---

## 5. Real-VLM Empirical Verification (paper §3.3)

### 5.1 Multi-VLM density figure (Fig multi_density)

5 base VLM에서 *cross-modal feature heterogeneity가 보편적*임을 실증:

- **Models**: CLIP B/32 (`clip_b32`), MetaCLIP B/32 (`metaclip_b32`), OpenCLIP/DataComp B/32 (`datacomp_b32`), MobileCLIP2-B (`mobileclip2b`), SigLIP2 Base (`siglip2_base`). paper text는 OpenCLIP으로 표기.
- **Pipeline**: COCO paired embedding → modality-specific SAEs $(\vV, \vW)$ 학습 (`m=8192`, TopK $K=8$, AdamW lr 5e-4, wd 1e-5, batch 1024, cosine warmup 5%, 30 epochs) → training set에서 $\Corr(\tilde\vz_I, \tilde\vz_T)$ 계산 → Hungarian-alive matching → 매칭 pair의 decoder cosine 측정 + correlation bin별 density.
- **Headline**: 5개 모델 *전부* correlation $\ge 0.6$ pair에서도 cosine distance가 0이 아니라 $\sim 0.5$ 근처 sharpen — *shared concept도 modality 간 다른 direction으로 인코딩*.
- Cache: `cache/{clip_b32,metaclip_b32,datacomp_b32,mobileclip2b,siglip2_base}_coco/`. Plot: `outputs/multi_model_density.{svg,pdf,png}`, caption tex `outputs/multi_model_density_caption.tex`.

### 5.2 Diagnostic A/B (legacy real_alpha_followup_*, no longer headline)

`real_alpha_followup_{1,2,3}` 디렉토리 — 초기 CLIP B/32 single-model diagnostic. 핵심 발견:
- Diagnostic A: $\Delta_\text{loss}$가 $L=4096\to 8192$에서 sign flip — Theorem 2 cost는 $L\ge 8192$에서 관측 가능.
- Diagnostic B: Hungarian-alive 매칭 (dead 85%), $r(C, \rho)=0.685$. C bin별 ρ median이 0.13→0.5로 monotonic 상승, $C\ge 0.6$에서 ρ≈0.5에 sharpen.
- Bisector verification: collapsed single-2R pair 14개 중 11/14에서 single이 (img+txt) bisector 근처 (cos>0.8).

이 결과들은 §5.1의 multi-VLM 검증으로 일반화됨.

### 5.3 ImageNet-1K + k Sweep + CC3M-Trained Pipeline (`real_exp_*`, downstream용)

별도 downstream evaluation 트랙:
- `real_exp_v1` (k=8 matrix, 6 method) + `real_exp_k{4,16,32}` (k sweep) → k=16 sweet spot.
- `real_exp_cc3m` (CC3M trained, k=32, 5 method) → **Post-hoc Alignment가 retrieval/zs 1위**: COCO I→T R@1 10.80, ImageNet zs top-1 27.23%. 자세한 결과 `outputs/real_exp_cc3m/RESULTS.md`.

### 5.4 Real-side 코드 맵

- `scripts/real_alpha/_bootstrap.py` — `src.*` import fix (transformers 5.5.3 우회).
- `scripts/real_alpha/extract_{clip_coco,imagenet,clip_cc3m}_cache.py` — embedding cache.
- `scripts/real_alpha/preprocess_cc3m_cache.py` — dict→stacked tensor 변환 (24GB OOM 회피).
- `scripts/real_alpha/train_real_sae.py` — `--variant {one_sae, two_sae, aux_sae, vl_sae}`, K env override.
- `scripts/real_alpha/run_real_v2.py` — YAML-driven 단일 driver (synthetic v2 mirror). `configs/real/{coco,cc3m}.yaml`.
- `scripts/real_alpha/build_real_table.py` — table.tex / table.md 생성.
- `scripts/real_alpha/run_diagnostic_B.py` — Hungarian + decoder cosine.
- `scripts/real_alpha/plot_multi_model_density*.py`, `plot_multi_model_boxplot.py` — multi-VLM plot.
- `experiment.sh`, `Dockerfile.experiment` — 10-model (5 Base + 5 Large) Docker pipeline.

### 5.5 Scope / Limits

- Linear atom 가정.
- Dead latent ~85% at k=8 → alive-restricted Hungarian.
- 현재 single seed가 다수 (실 paper에선 5 seed로 reorganization 필요).
- $\hat\alpha$ 정량화는 calibration table 미완 (synthetic α → ρ_median lookup TODO).

---

## 6. Paper Writing Artifacts (project root)

- **`paper.tex`** — `\subsection{Empirical Verification across Vision-Language Models}` 본문 (multi-VLM density 결과). intro/setup/results 3 paragraph.
- **`related_work.tex`** — Related Work section: 3 paragraph (Mech Interp + LRH / SAE for VLMs / Internal Structure of Joint Embedding Space).
- **`references.bib`** — placeholder bibtex. `% TODO` 마킹된 항목은 저자/venue 미확인.
- **인용 키 매핑 정리**: 본문에서 사용 중인 키들 — `liang2022mind`, `papadimitriou2025interpreting`, `kaushik2026learning`, `dhimoila2026crossmodal`, `zaigrajew2025interpreting`, `zhang2024connect`, `eslami2025mitigate`, `mistretta2025cross`, `bricken2023`, `cunningham2023`, `templeton2024scaling`, `gao2025scaling`, `rajamanoharan2024jumprelu`, `bussmann2024batchtopk`, `olah2017feature`, `elhage2021mathematical`, `park2024lrh`, `elhage2022toy`, `nanda2023emergent`, `hewitt2019structural`, `hewitt2019designing`, `subramani2022extracting`, `arditi2024refusal`, `rimsky2024steering`, `huben2023sparse`, `lieberum2024gemma`, `markssparse`, `rao2024discover`, `pach2026sparse`, `costa2026from`, `chuang2026meta`, `cherti2023reproducible`, `faghri2025mobileclip`, `tschannen2025siglip`, `radford21clip`, `kuhn1955hungarian`, `lin2014microsoft`, `makhzani2013k` 등.

---

## 7. Server Workflow

### 7.1 머신
- SSH alias `elice-40g` (10g 별도). 작업 dir: `/mnt/working/lvlm_hallucination` (로컬과 **별개 checkout**).
- GPU 10GB class, $L\le 16384$ + batch 1024 OK. transformers 5.5.3 (CLIP getter 우회 필요).
- `cache/` ~26GB: COCO 5VLM (~2GB) + ImageNet (3.1GB) + CC3M (13GB). 개별 dir만 rsync.

### 7.2 Canonical Loop (synthetic sweep)
1. 로컬에서 `configs/synthetic/<name>.yaml` 작성 + 새 `output.root`.
2. `rsync` config + script → 서버.
3. `ssh elice-40g "cd ... && nohup bash scripts/run_v2_<name>.sh > .log/v2_<name>.log 2>&1 & disown"`.
4. `ScheduleWakeup`로 ETA. Polling 금지.
5. 완료 후 `rsync -av elice-40g:.../outputs/<root>/ outputs/<root>/`.
6. 플롯은 **로컬에서**.

---

## 8. Open TODO & Gotchas

### 8.1 To Run
- ~~Multi-VLM 실증~~ ✅ (5 base 모델 §5.1).
- ~~CC3M 학습 + downstream~~ ✅ (`real_exp_cc3m`).
- **Synthetic α calibration table** — 합성 α ∈ {0.2,0.5,0.7,0.9,1.0}에서 동일 Hungarian-alive protocol → (α, ρ_median) lookup. $\hat\alpha$ 정량화의 missing piece.
- **Seed sweep** — real 모든 매트릭에 5 seed 붙이기.
- **Mean-centering control** — modality-specific train mean 빼고 재실험.
- **Dead-latent mitigation** — AuxK loss 또는 L 축소.

### 8.2 Figure / Notation Gotchas
- **Fig 1/2 figsize** — `(5.5, 1.015)` (0.7× 세로). plot script 직접 수정 시 유의.
- **Fig 1 x축** — `r"$\cos(\phi_i, \psi_i)$"` (α 표기 X). 본문도 `$\cos(\phi_i, \psi_i)$` for $i \in [n_S]$.
- **Fig 2 GRE y_min** — 0.2 (log scale). `set_yticks([0.2, 0.5, 1.0])`.
- **Method labels** — `"Shared SAE"`, `"Modality-Specific SAEs"`, `"Iso-Energy Alignment"`, `"Group-Sparse"`, `"Post-hoc Alignment"`. *옛 표기 (Separated SAE / Iso-Energy Align / Group-Sparse / Post-hoc Matching)는 사용 X*.
- **Theorem labels** — `thm:m-large` (Theorem 1), `thm:m-small` (Theorem 2). Propositions: `thm:prev:align`, `thm:prev:group`. (구 CLAUDE.md의 "Theorem 1 = Separate decoders preserve angle"는 *paper Theorem이 아니라 Diagnostic B의 motivation*이었음 — 혼동 주의.)

### 8.3 Server / Code
- **Two repos** — local + `/mnt/working/lvlm_hallucination`. 자동 sync 안됨.
- **Never `scp` whole repo** — `params/*.npz` 큼. dir별 rsync.
- **`output.save_decoders: true`** — synthetic 재계산 위해 필수.
- **CC3M OOM** — `preprocess_cc3m_cache.py`로 stacked tensor 만든 뒤 mmap 로드.
- **Cache key format** — COCO `f"{image_id}_{cap_idx}"`, CC3M `f"{image_id}"`. `CachedClipPairsDataset._text_key_for()` fallback.
- **`_bootstrap.py` 필수** — `scripts/real_alpha/*.py`에서 `src.*` import 시.
- **Matplotlib mathtext** — `\boldsymbol`, `\mathrm` (context별), `\!\,\;` 미지원. plain `\mathbf`, `\frac`, `_{S}` 사용.
- **`same_model_flag` caveat** — 1R/IA/GS는 `w_dec_img == w_dec_txt` byte-for-byte. decoder cosine 메트릭 해석 시 유의.
