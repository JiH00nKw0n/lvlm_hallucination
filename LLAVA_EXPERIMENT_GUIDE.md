# LLaVA-1.5 Experiment — Docker Build & Run Guide

Runs the **multi-density figure + Table 1** on **LLaVA-1.5-7B HACL-style paired
EOS embeddings** (Jiang et al., CVPR 2024): the image is `[<image>(576 visual
tokens), EOS]` and the caption is `[text, EOS]`, both pushed through the *same*
Vicuna-7B; we take the **final-layer hidden state at the EOS position** (dim
4096) as each side's embedding. A modality-specific TopK SAE (latent 65536,
k=256) is trained on COCO, and we produce the decoder-cosine density figure plus
the COCO-retrieval / ImageNet-zero-shot table.

Everything is packaged as a Docker image. You only need Docker + an NVIDIA GPU +
a Hugging Face token.

---

## 0. Prerequisites

| | requirement |
|---|---|
| GPU | NVIDIA, **≥ 24 GB** (40 GB comfortable). LLaVA-7B loads in bf16 (~14 GB); the SAE trains in ~9 GB. |
| Disk | ~40 GB free (LLaVA weights ~14 GB + caches ~15 GB + checkpoints ~10 GB). |
| Docker | with the NVIDIA Container Toolkit (`--gpus all` works). |
| HF token | required — **ImageNet-1k is gated** (llava-1.5-7b-hf and COCO are public). |

---

## 1. Get the code

```bash
git clone https://github.com/JiH00nKw0n/lvlm_hallucination.git
cd lvlm_hallucination
```

## 2. Hugging Face token (.env)

The zero-shot eval streams **ILSVRC/imagenet-1k**, which is gated. Accept its
terms once (while logged in) at
<https://huggingface.co/datasets/ILSVRC/imagenet-1k>, then create a token at
<https://huggingface.co/settings/tokens>.

Put it in a `.env` file (a template is provided):

```bash
cp .env.example .env
# edit .env and set HF_TOKEN=hf_xxxxxxxx
```

`.env`:
```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Load it into your shell so `docker run -e HF_TOKEN=$HF_TOKEN` can forward it:

```bash
export $(grep -v '^#' .env | xargs)
```

> The token is passed at **run time** via `-e HF_TOKEN=...` and is never baked
> into the image (`.env` is in `.dockerignore`).

## 3. Build the image

```bash
docker build -f Dockerfile.llava -t lvlm-llava .
```

(~5–10 min: installs torch 2.7 + transformers 4.57.1 and copies the code.)

## 4. Smoke run first (recommended, ~15 min)

Validates the whole pipeline end-to-end on 64 images at the real SAE size:

```bash
docker run --rm --gpus all \
  -e HF_TOKEN=$HF_TOKEN -e SMOKE=1 \
  -v $PWD/hf_cache:/workspace/.hf \
  -v $PWD/cache:/workspace/lvlm_hallucination/cache \
  -v $PWD/outputs:/workspace/lvlm_hallucination/outputs \
  lvlm-llava
```

Success prints `LLAVA_EXPERIMENT_DONE` and writes
`outputs/real_exp_llava_smoke/table.md` + `.../llava_density.{pdf,png,svg}`.
An early sanity line `paired-cosine sanity ... 0.5xxx (should be clearly > 0)`
confirms the LLaVA EOS extraction is meaningful.

> The smoke validates the **plumbing only** — its table numbers are throwaway
> (64 images / 1 epoch → e.g. 100% recall, `nan` recon). The full run produces
> real values. (Validated end-to-end on an A100 40 GB MIG on 2026-07-24.)

## 5. Full run

Same command **without `-e SMOKE=1`**:

```bash
docker run --rm --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -v $PWD/hf_cache:/workspace/.hf \
  -v $PWD/cache:/workspace/lvlm_hallucination/cache \
  -v $PWD/outputs:/workspace/lvlm_hallucination/outputs \
  lvlm-llava
```

- The `-v` mounts persist the model cache (`hf_cache`), embedding caches
  (`cache/`) and results (`outputs/`) on the host, so a re-run **resumes**
  (every stage skips if its artifact already exists).
- Wall time on one 40 GB A100 ≈ **~24 h** (dominated by LLaVA extraction of
  ~113k COCO images + 50k ImageNet val; SAE training + eval ≈ ~8 h).

### Outputs (host `outputs/real_exp_llava_coco/`)

| file | what |
|---|---|
| `table.md`, `table.tex` | Shared / Iso-Energy / Group-Sparse / Modality-Specific / **Post-hoc (Ours)** on COCO recon+retrieval and ImageNet zero-shot (single seed). |
| `llava_density.{pdf,png,svg}` | matched decoder-cosine density by co-activation bin. |
| `<method>/…` | per-method checkpoints + eval JSONs. |

---

## Stages (what the entrypoint does)

`scripts/run_llava_experiment.sh`:
1. **Extract** COCO (train+test) and ImageNet-val EOS embeddings →
   `cache/llava_coco`, `cache/llava_imagenet` (memmap, resumable, ETA-logged).
2. **Table 1** — `run_real_v2.py` trains Shared/Separated/Iso/Group, builds the
   Post-hoc perm from the Separated SAE, evaluates, writes the table.
3. **Density** — `run_diagnostic_B.py` on the Separated checkpoint →
   `plot_multi_model_density.py`.

## Configuration knobs

Edit `configs/real/coco_llava.yaml`:

| key | default | note |
|---|---|---|
| `training.latent_size` | 65536 | 16 × hidden (4096); split 32768/side for Separated. |
| `training.k` | 256 | 0.78 % sparsity (matches the CLIP baseline ratio). |
| `training.batch_size` | 1024 | cached-embedding training — memory is not batch-bound. |
| `training.num_epochs` | 30 | |
| `data.cache_dir` | `cache/llava_coco` | |

Env knobs: `SMOKE=1` (tiny run), `SAE_SAVE_TOTAL_LIMIT=N` (checkpoints kept per
method; defaults to 1 on the full run to save disk).

## Native (no Docker) alternative

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-server.txt
.venv/bin/pip install sentencepiece protobuf hf_transfer accelerate
export HF_TOKEN=hf_xxxx
SMOKE=1 bash scripts/run_llava_experiment.sh      # then without SMOKE for full
```

## Troubleshooting

- **HF 401 / gated** → `HF_TOKEN` unset or ImageNet-1k terms not accepted.
- **CUDA OOM during extraction** → lower `--image-batch` (extractor arg; default 8).
- **CUDA OOM during SAE training** → the latent-65536 SAE needs ~9 GB; on a
  small GPU reduce `training.latent_size` (e.g. 32768) in the yaml.
- **Interrupted run** → just re-run the same `docker run`; caches + the
  per-stage skip logic resume where it stopped.

---

## Reporting the density figure as numbers

Some venues forbid figures in a rebuttal. `scripts/real_alpha/density_bin_stats.py`
prints the same distribution the density plot draws — for each co-activation
correlation bin, the mean / median / std / min / max of the matched decoder
**cosine distance**.

It reads only what Stage 3 already wrote, so it needs **no GPU and no
re-extraction**:

```
outputs/real_exp_llava_coco/separated/ckpt/final/diagnostic_B_C_train.npy
outputs/real_exp_llava_coco/separated/ckpt/final/model.safetensors
```

Run it after (or instead of) the plotting step:

```bash
python scripts/real_alpha/density_bin_stats.py \
    --run-dir outputs/real_exp_llava_coco/separated/ckpt/final \
    --name "LLaVA-1.5-7B" \
    --out outputs/real_exp_llava_coco/density_bin_stats
```

Inside the Docker image:

```bash
docker run --rm --gpus all \
  -v $PWD/outputs:/workspace/lvlm_hallucination/outputs \
  --entrypoint python lvlm-llava \
  scripts/real_alpha/density_bin_stats.py \
    --run-dir outputs/real_exp_llava_coco/separated/ckpt/final \
    --name "LLaVA-1.5-7B" \
    --out outputs/real_exp_llava_coco/density_bin_stats
```

If `outputs/real_exp_llava_coco/density_models.json` exists (Stage 3 writes it),
`--models outputs/real_exp_llava_coco/density_models.json` does the same thing
and also handles several models at once.

### Outputs

| file | what |
|---|---|
| `density_bin_stats.md` | ready-to-paste tables, one section per model |
| `density_bin_stats.json` | the same numbers plus quartiles and the per-bin correlation summary |

`density_bin_stats.md` looks like this (numbers from CLIP ViT-B/32, for shape only):

| correlation bin | pairs | share | mean | median | std | min | max |
|---|---|---|---|---|---|---|---|
| [0.0, 0.2) | 289 | 51.8% | 0.8433 | 0.8743 | 0.1586 | 0.2708 | 1.1760 |
| [0.2, 0.4) | 132 | 23.7% | 0.6657 | 0.6551 | 0.1760 | 0.2372 | 1.0443 |
| [0.4, 0.6) | 70 | 12.5% | 0.5728 | 0.5687 | 0.1460 | 0.2929 | 0.9185 |
| [0.6, 0.8) | 45 | 8.1% | 0.4939 | 0.4722 | 0.0950 | 0.3174 | 0.8822 |
| [0.8, 1.0] | 22 | 3.9% | 0.5017 | 0.5123 | 0.0711 | 0.4066 | 0.6974 |

A second table repeats the same pairs as cosine instead of cosine distance,
since the two conventions get mixed up easily.

### Notes

- The script imports `load_model_data` from `plot_multi_model_density.py`
  instead of recomputing anything, so the table and the figure use the same
  alive rule, the same Hungarian matching and the same bin edges. They cannot
  drift apart.
- Bin edges are `[0, 0.2, 0.4, 0.6, 0.8, 1.0]`, and the top bin is closed on the
  right so a pair at exactly 1.0 is not dropped.
- Matched pairs whose correlation is negative fall below every bin. The script
  counts them and says so; they are still included in the *all matched pairs*
  row.
- The header line states which alive rule was used. Stage 3 does not write
  `diagnostic_B_firing_rates.npz`, so a fresh LLaVA run falls back to the
  nonzero-variance proxy — the same fallback the figure uses.
