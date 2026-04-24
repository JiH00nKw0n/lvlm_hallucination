# CLAUDE.md — Multimodal SAE Alignment

> 이 프로젝트를 resume할 때 먼저 읽을 것. **이론 (Theorem 1/2) + 합성 실험 + real-VLM $\alpha$-diagnostic**이 한 줄기이고, 서버 워크플로우와 gotcha는 §6–7에 몰아 뒀음.
> 참고: `src/decoding/` 아래 VCD/VISTA/OPERA 등 VLM hallucination decoding 코드는 **별개 프로젝트**. 여기서는 다루지 않음.

---

## 1. Research Question

Paired image–text embedding에서 공유 atom의 modality 간 정렬이 불완전 ($\alpha < 1$)할 때, **single-decoder SAE는 구조적으로 두 atom을 bisector로 merge**한다 (Theorem 2). Modality-Masking SAE (두 개의 분리된 decoder)는 merge를 피하지만 slot correspondence가 없다 (Theorem 1). **Ours = two-decoder + paired-slot aux loss**로 둘 다 얻는지 검증.

Two legs:
1. **Synthetic** — GT atom을 아는 상태에서 Theorem 1/2를 직접 확인 (✅ 완료: Fig 1–2, bridge matrix).
2. **Real-data** — $\alpha$가 관측 불가능한 실제 VLM (CLIP, SigLIP, …)에서 **indirect diagnostic**으로 $\hat\alpha$ 추정. 현재: COCO Diagnostic A+B (CLIP B/32) 완료, Multi-VLM Diagnostic B (5 Base 모델) 완료, ImageNet-1K 실험 진행 중.

---

## 2. Theory

**Generative model.** $\boldsymbol{x} = \boldsymbol{\Phi}_S \boldsymbol{z}_S + \boldsymbol{\Phi}_I \boldsymbol{z}_I + \boldsymbol{\epsilon}_I$, $\boldsymbol{y} = \boldsymbol{\Psi}_S \boldsymbol{z}_S + \boldsymbol{\Psi}_T \boldsymbol{z}_T + \boldsymbol{\epsilon}_T$. Partial alignment $\alpha_i := \cos([\boldsymbol{\Phi}_S]_{:,i}, [\boldsymbol{\Psi}_S]_{:,i}) \in [0,1]$ (실험에선 $\alpha_i = \alpha$ 상수).

**Theorem 1 (Separate decoders).** Two disjoint decoders $\boldsymbol{V}, \boldsymbol{W}$에서는 각 column이 대응 GT atom을 복원하고, **matched pair의 각도 $\alpha$가 training 후에도 보존됨**: $\cos(\boldsymbol{V}^\star[:,k], \boldsymbol{W}^\star[:,\pi(k)]) = \alpha$. Diagnostic B가 이를 사용.

**Theorem 2 (Shared decoder bisector merge).** Single decoder $\boldsymbol{D}$에서는 shared concept $i$의 최적 column이 $\boldsymbol{D}^\star[:,k] \propto [\boldsymbol{\Phi}_S]_{:,i} + [\boldsymbol{\Psi}_S]_{:,i}$ — 두 modality atom을 bisector로 병합. $\alpha=1$이면 무손실, $\alpha \in (0,1)$이면 structurally sub-optimal.

**Practical implication.**
- Conventional (1R, single decoder): 매 shared concept마다 bisector 비용 지불.
- Modality-Masking (2R, two decoders): merge 없음 but slot correspondence 없음.
- **Ours**: two decoders + 첫 $m_S$ slot에 paired-slot aux loss → 두 modality atom을 복원하면서 slot index 정렬.

---

## 3. Synthetic Setup

### 3.1 Data (followup12 이후 표준)

`src/datasets/synthetic_theory_feature.py`의 `SyntheticTheoryFeatureBuilder`. 주요 값: $d=256$, $n_S=1024$, $n_I=n_T=512$, $\varepsilon_{\max}=0.10$, Bernoulli sparsity $s=0.99$, $\text{Exp}(1)$ 계수, $\sigma_\text{obs}=0.05$.

**두 가지 함정** (이미 한 번씩 당함):
- $k=16$은 **SAE top-$k$**이지 data sparsity가 아님. Data는 Bernoulli $p=0.01$ → modality당 평균 ~15.4 active coords.
- Shared coefficients는 **independent mode** — mask만 공유, $[z_S^\text{img}]_i$와 $[z_S^\text{txt}]_i$는 독립 $\text{Exp}(1)$ 샘플. 따라서 $\mathbb{E}[\cos(x,y)] \approx 0.33\alpha$ (0.67$\alpha$가 아님).

### 3.2 Methods (`run_synthetic_v2.py`)

| Method | Single SAE? | `same_model_flag` | Loss |
|---|---|---|---|
| Conventional (1R, `single_recon`) | Yes | 1 | recon |
| Iso-Energy (IA, `iso_align`) | Yes | 1 | recon $-\beta_\text{IA}\cos(z_I, z_T)$ |
| Group-Sparse (GS, `group_sparse`) | Yes | 1 | recon $+\lambda_\text{GS}\sum_j \sqrt{z_I^{j2}+z_T^{j2}}$ |
| Modality-Masking (2R, `two_recon`) | No (두 decoder) | 0 | per-modality recon |
| **Ours** | No (두 decoder + aux) | 0 | recon $+\lambda_\text{aux}\cdot$ 첫 $m_S$ slot paired alignment |

**중요**: 1R/IA/GS는 decoder 파라미터를 literally 공유 ($\boldsymbol{V} = \boldsymbol{W}$) → decoder-cosine diagonal이 trivially 1. 이들에 대한 alignment 메트릭 해석 시 항상 유의.

Entry points: `run_synthetic_v2.py` (single trainer for all methods) driven by YAML configs in `configs/synthetic/`. 레거시 `synthetic_theorem2_method.py` 및 followup shell runner들은 2026-04-24 정리 시 제거됨.

### 3.3 Metrics

| Metric | Definition | 의미 |
|---|---|---|
| **RE** | $\mathbb{E}[\|x-\tilde x\|^2 + \|y-\tilde y\|^2]$ | recon quality |
| **GRR** @ $\tau$ | GT shared atom 중 $\tau$-cone 안에 decoder column이 있는 비율 | dictionary가 GT를 담는가 |
| **CR** (Collapse Rate) | GT shared atom 중 best-img/best-txt column이 $>0.95$로 평행한 비율 | decoder가 두 modality를 merge하는가 |
| **ELSim** | paired eval $(x_n,y_n)$의 $\cos(z_I,z_T)$ | real sample에서 sparse code 유사도 |
| **FLSim** | GT atom 주입 시 $\cos(z_I,z_T)$ | pure GT probe에서 sparse code 일치도 |
| **Cross-slot corr (top-$m_S$)** | 첫 $m_S$ slot Pearson diagonal 평균 | Ours에 유리한 설계 지표 |
| **Bridge matrix** | $\boldsymbol{B} = \mathbb{E}[z_I z_T^\top] \odot \tilde V\tilde W^\top$ | bridge energy 구조 — offline |
| **Dead neuron frac** | eval 전체에서 0-activation slot 비율 | offline |

**CR vs bridge trace**: CR은 decoder geometry, bridge trace는 top-$k$ inference — 다른 질문에 답함 (§4.3 참조).

### 3.4 파일/디렉토리

핵심 파일 (2026-04-24 정리 후):
- `src/datasets/synthetic_theory_feature.py` — 데이터 생성기. "데이터 어떻게 만드냐" 질문엔 기억 말고 이 파일을 읽을 것.
- `run_synthetic_v2.py` — 모든 method 학습 + eval. YAML config로 sweep 정의. 출력 root는 config의 `output.root` 필드.
- `configs/synthetic/alpha_1R_2R_L8192_5seeds_coarse.yaml` — Fig 1 α sweep (1R vs 2R, α ∈ {0.0, 0.2, …, 1.0}, 5 seeds, L=8192).
- `configs/synthetic/lambda_sweep.yaml` — Fig 2 λ sweep (α=0.5, L=8192, 5 seeds, single/two + IA/GS × 5 λ).
- `scripts/run_v2_alpha_1R_2R_coarse.sh`, `scripts/run_v2_lambda_sweep.sh` — 서버 런처.
- `scripts/plot_fig1_v3_from_npz.py` — Fig 1 plot (α sweep output 읽음).
- `scripts/plot_fig2_v3.py` — Fig 2 plot (λ sweep output 읽음).
- `scripts/palette.py` — 공통 색 팔레트.

저장 스키마 (`outputs/theorem2_v2_<name>/runs/run_<timestamp>/params/*.npz`): `w_enc_{img,txt}, b_enc_{img,txt}, w_dec_{img,txt}, b_dec_{img,txt}, phi_S, psi_S, phi_I, psi_T, alpha_target, seed, latent_size_img, latent_size_txt, same_model_flag`. 모든 offline 메트릭 재계산에 사용 (재학습 금지). `output.save_decoders: true` 필요.

Metrics JSON: `runs/*/result.json` → `sweep_results[0]["aggregate"]`.

### 3.5 Run Directory Index

2026-04-24 정리로 synthetic은 v2 pipeline 2개 output만 유지. 새 sweep은 새 config + 새 `output.root` 지정. 기존 dir은 **절대 덮어쓰지 말 것**.

| Dir | Purpose | $\alpha$ | $\sigma$ | Methods | Notes |
|---|---|---|---|---|---|
| `theorem2_v2_1R2R_5seeds_coarse` | α sweep, Fig 1 | 0.0, 0.2, …, 1.0 | 0.05 | 1R, 2R | 5 seeds, L=8192, `alpha_1R_2R_L8192_5seeds_coarse.yaml` |
| `theorem2_v2_lambda_sweep` | $\lambda$ sweep, Fig 2 | 0.5 | 0.05 | 1R/2R + IA/GS × 5 λ | 5 seeds, L=8192, `lambda_sweep.yaml` |

(레거시 `theorem2_followup_*` outputs는 2026-04-24 정리 시 제거됨.)

Real-data runs (CLIP ViT-B/32, COCO Karpathy, Diagnostic A+B):

| Dir | $L$ (one / two/side) | one loss | two loss | $\Delta_\text{loss}$ | Notes |
|---|---|---:|---:|---:|---|
| `real_alpha_followup_1` | 8192 / 4096 | 0.2820 | 0.2811 | +0.0009 | primary; Diagnostic B 보고서 |
| `real_alpha_followup_2` | 4096 / 2048 | 0.2847 | 0.2921 | −0.0074 | two worse (capacity) |
| `real_alpha_followup_3` | 16384 / 8192 | 0.2704 | 0.2650 | +0.0054 | 최대 gap |

공통: `openai/clip-vit-base-patch32`, $k=8$, 30 epochs, batch 1024, AdamW cosine+warmup 5%, lr 5e-4, wd 1e-5, grad clip 1.0, `normalize_decoder=True`, loss $=(\mathcal L_I+\mathcal L_T)/2$, seed 0, eval Karpathy test (25k pairs). Cache: `cache/clip_b32_coco/` (1.6 GB).

---

## 4. Synthetic Results (headline)

**Fig 1** (`theorem2_v2_1R2R_5seeds_coarse`, α ∈ {0, 0.2, …, 1.0}):
- **CR** — 가장 깔끔. 1R의 CR은 $\alpha$에 따라 monotonic하게 1까지 상승, 2R은 $\alpha$ 전 구간에서 ~0. Theorem 2의 직접적인 empirical shadow.
- **GRR** — 높은 $\alpha$에서 cross-over: $\alpha \approx 1$ 근처에서 1R이 2R을 앞서고, $\alpha \lesssim 0.7$에서는 2R 우세.
- **RE** — 작지만 단조적인 gap. 2R이 전 구간에서 약간 우수 (Diagnostic A 예측과 일치).

**Fig 2** (followup15, $\alpha=0.5$, $\lambda$ sweep):
- **Ours는 모든 $\lambda$에서 2R과 RE/CR 동등** + GRR은 약간 상회 ("best-of-both").
- **IA는 inert** — $\beta_\text{IA}$ 5 decade 이동해도 메트릭 변화 거의 없음.
- **GS는 $\lambda \times 1 \to \times 4$에서 절벽** — RE ~5배 급증, CR 0. Paper default 바로 위 collapse regime.

**Bridge matrix** (followup15, `report_bridge_matrix.md`): diagonal-concentration $\rho_\text{diag} = \text{tr}(B)/\sum_{ij} B_{ij}$ — Ours ~0.50, single-SAE baselines ~0.35, 2R ~0.001. Only Ours가 matched slot에 bridge energy 집중. IA @ $\lambda\times16$은 raw bridge trace만 inflate (magnitude scaling) — $\rho_\text{diag}$는 여전히 낮음.

**보조 수치** ($\alpha=0.5$): ELSim — Ours ~0.19, baseline ≤0.13, 2R ~0.003. FLSim — Ours 0.86–0.90, 1R ~0.43, 2R ~0. Dead neuron — Ours/2R 3–7%, 1R ~11%, GS collapse regime에서 >50%.

---

## 5. Real-Data Diagnostic

Plan: `outputs/theorem2_followup_15/plan_real_alpha_diagnostic.md`. 상세 분석: `outputs/real_alpha_followup_1/diagnostic_B_report.md`. Notion: https://www.notion.so/Real-Data-343abf49604280ab8771e7f26fb38f04

**Two Diagnostics.**
- **A (loss sign test)**: $\Delta_\text{loss} := \mathcal L_\text{rec}^\text{one} - \mathcal L_\text{rec}^\text{two}$. Positive면 2R이 우수 (= Theorem 2 cost 보임).
- **B (matched decoder cosine)**: 2R 학습 후 paired dense latent로 Pearson $C \in [-1,1]^{L/2 \times L/2}$ 계산, Hungarian-match ($-C$), 각 pair $(i,\pi(i))$에 대해 $\rho_k = \cos(V_I[:,i], V_T[:,\pi(i)])$.

### 5.1 핵심 결과 (`real_alpha_followup_1`, $L=8192$)

**Diagnostic A**: $\Delta$가 $L=4096 \to 8192$ 사이에서 부호 전환. $L \ge 8192$에서 Theorem 2 cost 관측 가능 (gap이 $L$과 함께 확대).

**Diagnostic B**: 85% dead latents — alive-restricted Hungarian (558 pairs). $r(C, \rho) = 0.685$ ($p \approx 10^{-78}$). $C$ bin별 $\rho$ median: [0,0.2) 0.126 → [0.2,0.4) 0.345 → [0.4,0.6) 0.431 → [0.6,0.8) 0.528 → [0.8,1.0) 0.488. 강하게 co-fire하는 쌍 ($C \ge 0.6$)의 median $\rho \approx 0.5$ — **calibration 없이는 $\hat\alpha$로 직결 불가**, raw observable.

**Merge evidence (single SAE)**:
- Per-sample top-$k$ overlap: $L$ 클수록 감소하지만 전 $L$에서 random chance 4–9× (merge 실존).
- $\cos \ge 0.7$ pair는 single SAE에서 거의 전부 self-matched (collapsed). 2R는 구조적으로 이 영역에 거의 없음.
- Collapse rate at $L=4096$: alive pairs with $C\ge 0.2$ 236개 중 self-matched (cos=1) 16개 → **6.8%**. 합성 $\alpha=0.5$의 CR ~10%와 같은 ball-park → shared concept $\alpha$가 0.4–0.6 범위 시사.
- 이 collapsed 16개를 2R pair와 top-50 sample Jaccard로 cross-match한 결과, 대응되는 2R pair cosine mean 0.546, median 0.589.

**Bisector verification (Theorem 2 in real)**: collapsed single-2R pair 14개 매칭 중 **11/14에서 single decoder가 (img + txt) bisector 근처** (cos > 0.8). Asymmetric merge — single이 image 방향으로 10/14 케이스에서 치우침.

**Iso-energy**: single SAE의 bimodal latents는 3.8%만, collapsed 24개는 $\mu \approx 0.40$에 집중 (iso-energy diagonal). 2R matched pairs는 대부분 (86%) 균형 잡힌 iso-energy diagonal 위. 상세 수치 → `outputs/real_alpha_followup_2/iso_energy{,_two}.npz`.

**Per-atom recon (Theorem 2 cost)**: 2R decoder column을 GT proxy로 취급. Collapsed 14 concept에서 single SAE recon error가 2R 대비 **image atom +47%, text atom +84%** 증가 (text 쪽이 큰 것은 bisector가 image 쪽으로 치우친 것과 일관). Script: `scripts/real_alpha/per_atom_recon_error.py`.

**Monosemanticity (MS)**: paired comparison (same concept)에서는 collapsed single ≈ 2R (유의미한 차이 없음). Aggregate 차이는 **selection effect**. MS는 collapse cost의 올바른 지표가 아님 — per-atom recon으로 측정할 것.

### 5.2 Multi-VLM Diagnostic B (5 Base 모델, §5.11 → 합쳐짐)

CLIP B/32, MetaCLIP B/32, DataComp B/32, MobileCLIP2-B, SigLIP2 Base 모두 동일 파이프라인 (COCO 임베딩 → 2R $L=8192$ → Hungarian-alive). 5개 모델 **전부** $C \uparrow \Rightarrow \rho \uparrow$, $C \ge 0.6$에서 $\rho$가 ~0.5 근처 집중 — 실제 VLM의 shared feature가 부분 정렬임을 시사. Cache: `cache/{clip_b32,metaclip_b32,datacomp_b32,mobileclip2b,siglip2_base}_coco/`. Plot: `outputs/multi_model_boxplot.pdf`.

### 5.3 ImageNet-1K Controlled Experiment

Class-label pairing + 80 OpenAI template — COCO free-form caption보다 구조적. Embedding cache (2026-04-19): train 1,281,167 images + val 50,000 (offset 2000000+) + 80,000 text (1000 cls × 80), `cache/clip_b32_imagenet/` ~3.1 GB. `CachedImageNetPairsDataset`이 `__getitem__`마다 random template 샘플링, `max_per_class` 파라미터로 balanced subsample. ImageNet k=8 matrix (6 method: shared/separated/iso_align/group_sparse/ours/vl_sae)는 `outputs/real_exp_v1/<method>/imagenet/` + monosemanticity 분석 완료 (§RESULTS.md).

### 5.4 k Sweep (COCO, k ∈ {4, 8, 16, 32})

6 method × 4 k 값으로 top-K sparsity sweep. Results: `outputs/real_exp_k{4,16,32}/` (k=8은 `real_exp_v1`). Headline: recon monotonic ↓ with k (Separated/Ours 0.213→0.0701 at k=4→32), retrieval/cross linprobe는 대부분 method에서 k=16 peak 후 k=32에서 약간 regression (dominant slot 개수 증가로 masking 손실 커짐). VL-SAE만 k 증가에 계속 scaling. 종합: **k=16 balanced sweet spot** (compute 대비 recon/downstream trade-off 최적). Full tables → `outputs/real_exp_k32/RESULTS_full_k_sweep.md`.

### 5.5 CC3M Embedding Cache (2026-04-24)

Full pipeline 캐시 완료(ish). Source: `pixparse/cc3m-wds` (HF streaming). Model: CLIP ViT-B/32. Output `cache/clip_b32_cc3m/`: 13 GB (image_embeddings.pt 6.6 GB, text_embeddings.pt 6.6 GB, splits.json 48 MB). **Partial**: ~98.2% (n=2,818,048 / ~2.87M) — final flush에서 torch.save deadlock으로 마지막 1.8%는 손실. Resume 지원되지만 추가 실행 없이 98%로 사용. Script: `scripts/real_alpha/extract_clip_cc3m_cache.py` (DataLoader num_workers=2 권장, 4는 I/O 경합). Throughput 실측 60–73 samples/sec → 약 13시간 소요 (HF streaming + JPEG decode가 bottleneck, GPU 아님).

### 5.6 Real-side 코드 맵 (핵심만)

- `scripts/real_alpha/_bootstrap.py` — server `src.*` import fix (transformers 5.5.3 `flex_attention` 브레이크 우회). 모든 real-alpha 스크립트에서 `sys.path.insert(0,...) ; import _bootstrap`.
- `scripts/real_alpha/extract_{clip_coco,imagenet,clip_cc3m}_cache.py` — 임베딩 추출 (idempotent; cc3m은 HF streaming).
- `scripts/real_alpha/train_real_sae.py` — `--variant {one_sae, two_sae, aux_sae, vl_sae}`, HF Trainer. K env override (`K=4/8/16/32`)로 sweep.
- `scripts/real_alpha/run_real_exp_matrix.sh` / `run_cross_valprobe.sh` — downstream matrix + cross eval. `OUT=outputs/real_exp_k{K}`로 분리 저장.
- `scripts/real_alpha/eval_imagenet_{valprobe,zeroshot,zeroshot_masked_dense}.py` — 3가지 zs 변형 (raw / dominant-masked dense / masked+top-1).
- `scripts/real_alpha/run_diagnostic_B.py` — Hungarian + decoder cosine + cofiring threshold sweep.
- `scripts/real_alpha/plot_diagnostic_{A_pub,B_violin}.py` — publication plots.
- `scripts/real_alpha/{iso_energy_check,iso_energy_two,per_atom_recon_error,monosemanticity_experiment}.py` — 분석.
- `scripts/real_alpha/{render_collapse_v2,inspect_top_activations,cache_single_sae_C}.py` — diagnostic HTML/캐시.
- `scripts/real_alpha/plot_multi_model_boxplot.py`, `run_multi_model_pipeline.sh` — multi-VLM.
- `experiment.sh`, `Dockerfile.experiment` — 10-model (5 Base + 5 Large) Docker 파이프라인.
- `src/datasets/cached_{clip,imagenet}_pairs.py` — Dataset wrapper.
- `src/models/{configuration,modeling}_sae.py` — `TwoSidedTopKSAE{Config,}` 추가.
- `src/runners/trainer.py` — `OneSidedSAETrainer`, `TwoSidedSAETrainer`.

### 5.7 Scope / Limits

- Linear atom 가정 (prior work 공통).
- SAE 최적화 local optima → seed sweep 필요.
- Dead latent 85% (top-$k=8$, no AuxK): alive-restricted가 완화이지만 근본 수정 아님.
- L-capacity: $L=4096$에서는 per-side halving으로 2R 손해 ($\Delta<0$). $L\ge 8192$에서 비로소 Theorem 2 signal 보임.
- Hungarian은 Theorem 1 permutation의 proxy일 뿐.
- 현재 single seed + CLIP B/32 only + calibration 미완. 정량적 $\hat\alpha$ 주장하려면 seed sweep + synthetic lookup table + multi-VLM 필요.

---

## 6. Server Workflow

### 6.1 머신

- SSH alias `elice-40g` (10g도 있음). Config `~/.ssh/config`.
- 서버 작업 디렉토리: `/mnt/working/lvlm_hallucination` — 로컬과 **별개 checkout**. 변경 사항 수동 sync 필요.
- GPU: 10 GB class. $L \le 16384$, batch 1024까지 수용. `nvidia-smi`는 "Insufficient Permissions" 떠도 `torch.cuda`는 정상.
- Python: `.venv/` + `source .venv/bin/activate`. **transformers 5.5.3** — `CLIPModel.get_image_features`가 `BaseModelOutputWithPooling` 반환. `model.vision_model(...)` + `model.visual_projection(...)` 직접 호출해야 함.
- `cache/` 총 ~26 GB: clip/metaclip/datacomp/mobileclip2b/siglip2 COCO (~2 GB) + clip imagenet (3.1 GB) + **clip cc3m (13 GB)**. 개별 dir만 선택적 rsync.
- 서버 디스크 256G, ~100G 사용 중 (2026-04-24). HF cache 주 소비자. 주기적으로 `pip cache purge`.

### 6.2 Canonical Experiment Loop (합성 sweep 표준)

1. 로컬에서 `configs/synthetic/<name>.yaml` 작성 (새 실험은 새 `output.root` 지정). 런처는 `scripts/run_v2_<name>.sh` 템플릿 참고: `python run_synthetic_v2.py --config configs/synthetic/<name>.yaml`.
2. `rsync -av configs/synthetic/<name>.yaml scripts/run_v2_<name>.sh elice-40g:/mnt/working/lvlm_hallucination/<path>/`
3. 백그라운드 시작:
   ```
   ssh elice-40g "cd /mnt/working/lvlm_hallucination && nohup bash scripts/run_v2_<name>.sh > .log/v2_<name>.log 2>&1 & disown; echo pid=\$!"
   ```
4. 시작 확인: `ssh elice-40g "tail -3 .log/v2_<name>.log; pgrep -af run_synthetic_v2"`.
5. `ScheduleWakeup`으로 예상 완료 시간 + buffer 설정. Polling 금지.
6. 완료 후 `rsync -av elice-40g:/mnt/working/lvlm_hallucination/outputs/<root>/ outputs/<root>/` (result.json + params/*.npz 전부; `output.save_decoders: true` 설정 시 .npz 저장됨).
7. 플롯은 **로컬에서** 수행. 서버에서 렌더링 금지.

진행 추정: $L=8192$, 10 ep, batch 256, 50k train 기준 method × α당 ~3분.

---

## 7. Open TODO & Gotchas

### 7.1 To Run

- ~~Real CLIP paired-encoder sweep~~ — ✅ `real_alpha_followup_{1,2,3}`.
- ~~Multi-VLM cross-check~~ — ✅ 5 Base 완료.
- ~~Repo cleanup~~ — ✅ 2026-04-24. 레거시 synthetic scripts, hallucination decoding, test scripts, followup runner shells, 구 plot 전부 제거. Synthetic은 `run_synthetic_v2.py` + 2 config로 일원화.
- **Synthetic calibration** — 동일 Hungarian-alive protocol을 합성 $\alpha \in \{0.2, 0.5, 0.7, 0.9, 1.0\}$에 적용해 $(\alpha, \rho_\text{median})$ lookup 구축. **이게 $\hat\alpha$ 정량화의 핵심 missing piece**.
- **Mean-centering control** — per-modality train mean 뺀 후 ℓ2-norm, 재학습, Diagnostic B 재측정. CLIP modality gap (Liang 2022) vs real shared structure 분리 테스트.
- **Dead-latent mitigation** — AuxK loss ($2^{-5}$) 또는 $L$ 축소로 alive 비율 ~14% → ≥50%. 신규 alive atom이 $\rho>0$ 영역에 가는지 확인.
- ~~ImageNet-1K 2R 학습 + Diagnostic B~~ — ✅ k=8 matrix 완료 (`real_exp_v1`, §5.3).
- ~~CC3M 캐싱~~ — ✅ 98.2% (`cache/clip_b32_cc3m/`, §5.5). CC3M 기반 real 학습/Diagnostic 후속 실험 남음.
- **Seed sweep** — real 전부 + followup 15/16 모두 single seed. 에러 바 붙이려면 3 seed 필수.

### 7.2 Figure Gaps

- Fig 2 companion table (bridge trace/sum/$\rho_\text{diag}$ across $\lambda$) 아직 figure 형태 아님.
- v2 α coarse sweep은 1R/2R만 — Ours/IA/GS까지 포함한 wide α sweep은 새 config로 재실험하면 Fig 1 강화.
- Notion: `Exp-syn` 합성 write-up, `Real-` Diagnostic A+B 표. Figure는 drag-drop 수동. `report_bridge_matrix.md`는 아직 로컬 only.

### 7.3 Gotchas

- **Two separate repos** — 로컬 `~/Desktop/Projects/lvlm_hallucination`과 서버 `/mnt/working/lvlm_hallucination`은 sync 안 됨. 편집 시 scp/rsync 명시적으로.
- **Never `scp` whole repo** — `params/` 덤프 큼. scripts는 타겟 scp, outputs는 dir별 rsync.
- **v2 config `output.save_decoders: true`** — 없으면 `.npz` 저장 안 돼서 offline 메트릭 재계산 불가. 항상 켤 것.
- **nohup 좀비 sweep** — ssh 끊어도 process 살아있음. `ssh elice-40g "pgrep -af run_synthetic_v2"` 확인.
- **Matplotlib mathtext**는 `\boldsymbol, \displaystyle, \dfrac, \tfrac, \mathrm` (context 따라), `\bigl/\bigr`, `\!/\,/\;` 미지원. `\mathbf`, `\frac`, plain subscript (`_{S}`) 사용.
- **Notion upload tool** (`.claude/scripts/upload_md_to_notion.py`)은 `\boldsymbol{...}`, `\mathbb{R}`만 허용 — `\v`, `\sR` 같은 커스텀 매크로는 raw로 렌더.
- **`same_model_flag` caveat** — 1R/IA/GS는 `w_dec_img == w_dec_txt` byte-for-byte → $\boldsymbol A$ diagonal 자동 1. "baseline fails at slot-level alignment"는 co-activation 이야기지 decoder cosine 이야기 아님.
- **v2 α coarse는 1R/2R만** — "Ours at $\alpha=1.0$" 읽으려 하면 KeyError. Ours/IA/GS 포함 sweep은 별도 config로.
- **CJK font** — `scripts/plot_diagnostic_B_intuition.py`에 한글. AppleGothic/NanumGothic 없으면 box로 렌더.
- **`dead_neuron_fraction`은 offline 계산** — `result.json`에 없음. bridge report protocol로 재계산.
- **`_bootstrap.py` 필수** — `scripts/real_alpha/*.py`에서 `src.*` import 시 transformers 5.5.3의 `flex_attention` 브레이크 우회. `sys.path.insert(0, ...) ; import _bootstrap`.
- **Real-data dead latent ~85%** at $k=8$, $L \in \{4096,8192,16384\}$, 30 ep, no AuxK. Diagnostic B 분석은 반드시 alive-restricted Hungarian (fire rate >0.001). Firing rate cache는 `diagnostic_B_firing_rates.npz`.
- **L-capacity on $\Delta_\text{loss}$** — $L=4096$에서는 per-side 2048뿐이라 2R이 손해 ($\Delta<0$). Theorem 2 signal ($\Delta>0$)은 $L\ge 8192$부터. 부호 해석 전 $L$ 확인.
