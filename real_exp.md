# Real-Data Downstream Experiment Plan (CLIP ViT-B/32)

260422 슬라이드의 Real data 파트 구현용 실험 계획. 논문 본문 Table 1 (ImageNet)
+ Table 2 (COCO)를 채우는 것이 목적이고, 여기서 논하는 pipeline은 이후 다른
VLM 체크포인트로도 동일하게 확장한다.

## 1. Task × Dataset × Method Matrix

### Table 1 — ImageNet-1K (k=1, single-feature regime)

Dataset: `CachedImageNetPairsDataset(split=train, max_per_class=1000)` → 1M pairs.
Each `__getitem__` returns `{image_embeds, text_embeds}` where the text side is
one of 80 OpenAI templates (random per step).

| Method | Recon ↓ | Linear probe ↑ (image-side latent) | Zero-shot ↑ (text prototype match) |
|---|---|---|---|
| Shared SAE         |  |  |  |
| Separated SAE      |  |  |  |
| Iso-Energy Align   |  |  |  |
| Group-Sparse       |  |  |  |
| **Ours** (Hungarian) |  |  |  |

### Table 2 — COCO Karpathy (k=8, multi-feature regime)

Dataset: `CachedClipPairsDataset` train / test splits.

| Method | Recon ↓ | I→T R@1/5/10 ↑ | T→I R@1/5/10 ↑ |
|---|---|---|---|
| Shared SAE         |  |  |  |
| Separated SAE      |  |  |  |
| Iso-Energy Align   |  |  |  |
| Group-Sparse       |  |  |  |
| **Ours** (Hungarian) |  |  |  |

## 2. What Gets Trained

8 checkpoints in total (4 per dataset). **Ours 는 학습하지 않음** — Separated
체크포인트를 그대로 재사용하고 평가 시점에 Hungarian permutation만 적용.

| Variant | Model | Loss |
|---|---|---|
| `one_sae` | `TopKSAE(L=8192, k)` | `(recon_I + recon_T) / 2` |
| `two_sae` | `TwoSidedTopKSAE(L=8192, k)` (4096/side) | per-side recon 평균 |
| `aux_sae` + `iso_align` | `TopKSAE(L=8192, k)` | `recon − β · iso_penalty(z_I, z_T)` |
| `aux_sae` + `group_sparse` | `TopKSAE(L=8192, k)` | `recon + λ · ‖(z_I, z_T)‖_{2,1}` |

공통 hp (CLAUDE.md §5.1 관례와 동일): batch 1024, 30 epochs, AdamW, lr 5e-4
(cosine, warmup 5%), wd 1e-5, grad clip 1.0, `normalize_decoder=True`, seed 0.

Aux loss 기본값 (합성 `followup15` 수준): `iso β = 1e-4`, `group_sparse λ = 0.05`.

## 3. How Ours Works (post-hoc)

1. Load Separated SAE checkpoint.
2. Stream paired train embeddings through both sides with `return_dense_latents=True`;
   build Pearson correlation matrix `C ∈ ℝ^{L/2 × L/2}` via
   `synthetic_theorem2_method._compute_latent_correlation`.
3. Hungarian: `row_ind, col_ind = scipy.optimize.linear_sum_assignment(-C)`.
4. Save `perm = col_ind` as the text-slot → image-slot mapping.
5. At eval time, image latent stays as-is; text latent is re-indexed: `z_T_aligned = z_T[:, perm]`.

(Theorem 1 guarantees column-level recovery but not slot correspondence; Hungarian
is the proxy that restores slot alignment.)

## 4. Evaluation

공통: `l2_normalize=True`로 로드된 cached embeddings. 모든 eval은 SAE의
reconstructed latent 공간에서만 수행 (raw CLIP 공간 아님).

1. **Recon**: eval split에서 `0.5 * (‖x − x̂‖² + ‖y − ŷ‖²)` sample-mean.
2. **Linear probe** (ImageNet train/val): 이미지 SAE latent `z_I ∈ ℝ^L`을 train 전체에 뽑은 뒤 scikit-learn `LogisticRegression`(multinomial, lbfgs, C=1.0, max_iter=1000)을 fit. val top-1 accuracy 보고.
3. **Zero-shot** (ImageNet val): class prototype은 CLIP 공간에서 80 템플릿 embedding **평균 → SAE 인코딩**. val image latent와 각 prototype간 cosine max로 top-1 예측.
4. **Retrieval** (COCO Karpathy test): 5k image / 25k caption. I→T 는 각 이미지 latent와 모든 caption latent 간 cosine 랭킹, T→I는 역방향. R@{1,5,10}.

## 5. File layout

| 역할 | 경로 |
|---|---|
| Trainer 추가분 (aux loss) | `src/runners/trainer.py::OneSidedAuxSAETrainer` |
| 학습 엔트리 확장 | `scripts/real_alpha/train_real_sae.py` (`--variant aux_sae`, `--aux-loss`, `--aux-lambda`, `--dataset {coco,imagenet}`, `--max-per-class`) |
| 공통 eval 유틸 | `scripts/real_alpha/eval_utils.py` |
| Eval scripts | `scripts/real_alpha/eval_{recon_downstream,imagenet_linprobe,imagenet_zeroshot,coco_retrieval}.py` |
| 서버 드라이버 | `scripts/real_alpha/run_real_exp_matrix.sh` |
| 테이블 집계 | `scripts/real_alpha/collate_real_exp_table.py` |
| 결과 덤프 | `outputs/real_exp_v1/<method>/<dataset>/{results.json, perm.npz}` |

## 6. Server run recipe (elice-40g)

캐시는 `cache/clip_b32_{coco,imagenet}`에 이미 있음 (각각 1.7G / 3.1G, 확인 완료).

```
# local
scp src/runners/trainer.py \
    scripts/real_alpha/{train_real_sae.py,eval_utils.py,eval_*.py,run_real_exp_matrix.sh,collate_real_exp_table.py} \
    elice-40g:/mnt/working/lvlm_hallucination/<same-paths>

# server
ssh elice-40g
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
nohup bash scripts/real_alpha/run_real_exp_matrix.sh > .log/real_exp_v1.log 2>&1 &
```

예상 소요: 학습 8개 × ~20분 + eval (특히 linear probe) ≈ 3–4시간.
완료 후 로컬로 rsync 해서 `collate_real_exp_table.py`로 LaTeX 생성.

## 7. Out-of-scope (이번 iteration 아님)

- Multi-VLM (CLIP 외): COCO caches는 5종 있지만 먼저 CLIP B/32로 table 마감.
- Seed sweep: single seed로 수치 확보 후 camera-ready.
- Steering 실험, Aux fine-tune 버전의 Ours, ImageNet retrieval, COCO linear probe.
