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

---

## Quickstart: the run is already finished, I just need the numbers

Nothing needs to be retrained or re-extracted. Three steps.

**1. Update the code.**

```bash
cd lvlm_hallucination
git pull origin master
```

**2. Check the two inputs are there.** These are left behind by Stage 3 of the
run you already completed:

```bash
ls outputs/real_exp_llava_coco/separated/ckpt/final/diagnostic_B_C_train.npy \
   outputs/real_exp_llava_coco/separated/ckpt/final/model.safetensors
```

If `diagnostic_B_C_train.npy` is missing, Stage 3 never ran. Produce it with the
command below — it needs the COCO embedding cache and a GPU, and takes a few
minutes because the correlation matrix is the only thing it computes:

```bash
python scripts/real_alpha/run_diagnostic_B.py \
    --run-dir outputs/real_exp_llava_coco/separated/ckpt/final \
    --cache-dir cache/llava_coco
```

**3. Write the tables.**

```bash
python scripts/real_alpha/density_bin_stats.py \
    --run-dir outputs/real_exp_llava_coco/separated/ckpt/final \
    --name "LLaVA-1.5-7B" \
    --out outputs/real_exp_llava_coco/density_bin_stats
```

Docker equivalent, if the native environment is not set up (CPU is enough — pass
`--gpus all` only if you also need step 2):

```bash
docker run --rm \
  -v $PWD/outputs:/workspace/lvlm_hallucination/outputs \
  --entrypoint python lvlm-llava \
  scripts/real_alpha/density_bin_stats.py \
    --run-dir outputs/real_exp_llava_coco/separated/ckpt/final \
    --name "LLaVA-1.5-7B" \
    --out outputs/real_exp_llava_coco/density_bin_stats
```

**What to send back:** `outputs/real_exp_llava_coco/density_bin_stats.md`. The
console also prints the per-bin summary, so a copy of the terminal output works
as a fallback. `density_bin_stats.json` carries the same numbers plus quartiles
if anything needs checking later.

---

## Additional experiments: capacity × sparsity grid

The main run fixes one point on two axes at once — `k = 256` and
`latent_size = 65536`. This grid varies both so the conclusions can be read
against a range instead of a single setting.

`latent_size` is the **total**. The modality-specific methods (`separated`,
`ours`) keep the usual half-and-half split, so an arm at `latent_size = 8192`
gives each side 4096 coordinates. The shared methods use all of it, which is
what makes the two families capacity-matched.

| config | k | latent_size (total) | per side | active per side |
|---|---:|---:|---:|---:|
| `coco_llava_k32_L8192.yaml` | 32 | 8192 | 4096 | 0.781 % |
| `coco_llava_k32_L16384.yaml` | 32 | 16384 | 8192 | 0.391 % |
| `coco_llava_k32_L32768.yaml` | 32 | 32768 | 16384 | 0.195 % |
| `coco_llava_k16_L8192.yaml` | 16 | 8192 | 4096 | 0.391 % |
| `coco_llava_k16_L16384.yaml` | 16 | 16384 | 8192 | 0.195 % |
| `coco_llava_k16_L32768.yaml` | 16 | 32768 | 16384 | 0.098 % |
| `coco_llava.yaml` (main run) | 256 | 65536 | 32768 | 0.781 % |

**`k32_L8192` is the anchor arm.** 4096 coordinates per side at k = 32 is exactly
the CLIP setting from the paper's main experiments, so it makes the LLaVA and
CLIP numbers directly comparable. `k32_L8192` and the k = 256 main run also share
an active fraction of 0.781 %, which separates the effect of raw capacity from
the effect of sparsity.

Nothing else changes: epochs, batch size (1024), learning rate, schedule, data
splits and the five methods all come from `configs/real/coco_llava.yaml`.

### Running an arm

One script per arm, each standalone, so they can be launched in parallel:

```
scripts/run_llava_k32_L8192.sh
scripts/run_llava_k32_L16384.sh
scripts/run_llava_k32_L32768.sh
scripts/run_llava_k16_L8192.sh
scripts/run_llava_k16_L16384.sh
scripts/run_llava_k16_L32768.sh
```

```bash
bash scripts/run_llava_k32_L8192.sh
```

The arms share nothing at run time except the read-only embedding caches, and
each writes to its own `outputs/real_exp_llava_coco_k{K}_L{L}/`. Extraction is
**not** repeated — they reuse `cache/llava_coco` and `cache/llava_imagenet` from
the main run and start at SAE training. Each arm is idempotent: it returns
immediately if its `table.md` already exists.

Suggested order, one k group at a time:

```bash
# k = 32 first, all three capacities in parallel
for L in 8192 16384 32768; do
  nohup bash scripts/run_llava_k32_L${L}.sh > .log/arm_k32_L${L}.out 2>&1 &
done
wait

# then k = 16
for L in 8192 16384 32768; do
  nohup bash scripts/run_llava_k16_L${L}.sh > .log/arm_k16_L${L}.out 2>&1 &
done
```

To pin arms to different GPUs, set `CUDA_VISIBLE_DEVICES` per launch:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_llava_k32_L8192.sh &
CUDA_VISIBLE_DEVICES=1 bash scripts/run_llava_k32_L16384.sh &
```

### Docker — no rebuild needed

`Dockerfile.llava` bakes the code in with `COPY . .`, so a previously built image
does not contain these new scripts. Rather than rebuilding, mount `scripts/` and
`configs/` over the baked copies:

```bash
docker run --rm --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -v $PWD/hf_cache:/workspace/.hf \
  -v $PWD/cache:/workspace/lvlm_hallucination/cache \
  -v $PWD/outputs:/workspace/lvlm_hallucination/outputs \
  -v $PWD/scripts:/workspace/lvlm_hallucination/scripts \
  -v $PWD/configs:/workspace/lvlm_hallucination/configs \
  --entrypoint bash lvlm-llava \
  scripts/run_llava_k32_L8192.sh
```

The two extra `-v` lines are the only difference from the main run's command.
Launch one container per arm to run them in parallel.

### Fitting several arms on one GPU

SAE parameter count is `2 × hidden × latent_size` = `8192 × L`, the same for the
shared and the two-sided methods since capacity is matched. With fp32 weights,
gradients and AdamW moments that is roughly:

| latent_size | parameters | ≈ GPU memory per arm |
|---:|---:|---:|
| 8192 | 67 M | ~1.1 GB |
| 16384 | 134 M | ~2.2 GB |
| 32768 | 268 M | ~4.3 GB |

One k group (three arms) therefore needs about 8 GB plus a CUDA context per
process, so three arms fit comfortably on a 24 GB card and all six fit on a
40 GB one. These are estimates from parameter counts — watch `nvidia-smi` on the
first launch rather than trusting them.

### Cost

Extraction is skipped, so an arm is SAE training (5 methods) plus evaluation.
The k = 256 / 65536 main run takes ~8 h for this part; these arms are smaller, so
expect less. Run in parallel, one k group should finish well inside a day.

Disk grows by roughly 1–5 GB per arm at `SAE_SAVE_TOTAL_LIMIT=1`, which the
scripts set by default.

### Outputs

| path | what |
|---|---|
| `outputs/real_exp_llava_coco_k{K}_L{L}/table.md`, `table.tex` | the five-method table for that arm |
| `outputs/real_exp_llava_coco_k{K}_L{L}/<method>/…` | per-method checkpoints + eval JSONs |
| `outputs/real_exp_llava_coco_k{K}_L{L}/density_bin_stats.{md,json}` | only with `DENSITY=1` |
| `.log/llava_k{K}_L{L}.log` | full training/eval log for that arm |

Each script prints its own `table.md` when it finishes.

### What to watch

The sparsest arm, `k16_L32768`, activates 16 of 16384 coordinates per side. A
large dead-latent fraction there is expected rather than a bug, but it changes
how the row should be read: alive-restricted matching has fewer coordinates to
work with, so the permutation is solving a smaller problem.

These configs therefore add `dead_latents` to the COCO evaluation tasks, so every
method writes its own count to
`outputs/real_exp_llava_coco_k{K}_L{L}/<method>/coco/dead_latents.json`:

```json
{"alive_image_count": ..., "latent_size_image": ...,
 "alive_text_count": ...,  "latent_size_text": ...}
```

**A latent counts as alive if it fires at least once** (`alive_min_fires = 1`) on
the sample the evaluator sees, which is the first 50,000 pairs
(`eval_dead_latents.py --max-samples`, default 50000). This is the same rule
`build_perm` uses to decide which coordinates the Hungarian assignment may match,
so the two numbers refer to the same thing. It is *not* the stricter
firing-rate > 0.001 rule used by the density figure — a rarely firing latent
counts as alive here and does not there.

Please send back `table.md` for each arm (and `density_bin_stats.md` if
`DENSITY=1` was used).
