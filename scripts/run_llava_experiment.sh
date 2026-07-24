#!/usr/bin/env bash
# End-to-end LLaVA-1.5 experiment (HACL-style EOS embeddings):
#   Stage 1  extract COCO + ImageNet EOS embeddings (memmap, resumable)
#   Stage 2  Table-1 pipeline: train Shared/Separated/Iso/Group + Ours perm + eval
#   Stage 3  multi-density figure from the Separated (two-SAE) checkpoint
#
# Delivered as a Docker image (Dockerfile.llava) but also runs natively.
#   docker run --rm --gpus all -e HF_TOKEN=$HF_TOKEN [-e SMOKE=1] lvlm-llava
#   SMOKE=1 bash scripts/run_llava_experiment.sh                     # native
#
# Every stage is idempotent (skips when its output already exists), so a
# crashed run resumes by re-invoking. SMOKE=1 runs a tiny (64-image) end-to-end
# validation at the REAL SAE size.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f env.sh ] && source env.sh || true

PY=python; [ -x .venv/bin/python ] && PY=.venv/bin/python
SMOKE="${SMOKE:-0}"
mkdir -p .log cache outputs

if [ "$SMOKE" = "1" ]; then
  TAG=smoke
  COCO_CACHE=cache/llava_coco_smoke;  IN_CACHE=cache/llava_imagenet_smoke
  CFG=configs/real/coco_llava_smoke.yaml;  ROOT=outputs/real_exp_llava_smoke
  COCO_LIMIT=64; IN_LIMIT=100; NCLS=10; NTPL=80
else
  TAG=full
  COCO_CACHE=cache/llava_coco;  IN_CACHE=cache/llava_imagenet
  CFG=configs/real/coco_llava.yaml;  ROOT=outputs/real_exp_llava_coco
  COCO_LIMIT=0; IN_LIMIT=0; NCLS=1000; NTPL=80
  export SAE_SAVE_TOTAL_LIMIT="${SAE_SAVE_TOTAL_LIMIT:-1}"   # big SAE -> save disk
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN unset — ImageNet-1k is gated and will 401. "\
       "Pass -e HF_TOKEN=... (llava-1.5-7b-hf & COCO are public)." >&2
fi

banner(){ echo; echo "======== $* ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ========"; }

banner "STAGE 1a: extract COCO EOS embeddings -> $COCO_CACHE"
if [ -f "$COCO_CACHE/meta.json" ]; then echo "skip: $COCO_CACHE/meta.json exists"; else
  $PY scripts/real_alpha/extract_llava_cache.py --dataset coco \
      --cache-dir "$COCO_CACHE" --splits train test --limit "$COCO_LIMIT"
fi

banner "STAGE 1b: extract ImageNet val EOS embeddings -> $IN_CACHE"
if [ -f "$IN_CACHE/meta.json" ]; then echo "skip: $IN_CACHE/meta.json exists"; else
  $PY scripts/real_alpha/extract_llava_cache.py --dataset imagenet \
      --cache-dir "$IN_CACHE" --limit "$IN_LIMIT" --n-classes "$NCLS" --n-templates "$NTPL"
fi

banner "STAGE 2: Table-1 pipeline (train 4 SAEs + Ours perm + eval + table) -> $ROOT"
$PY scripts/real_alpha/run_real_v2.py --config "$CFG"

banner "STAGE 3: multi-density figure from Separated two-SAE ckpt"
SEP="$ROOT/separated/ckpt/final"
if [ ! -d "$SEP" ]; then echo "ERROR: $SEP missing (Stage 2 did not train 'separated')" >&2; exit 1; fi
$PY scripts/real_alpha/run_diagnostic_B.py --run-dir "$SEP" --cache-dir "$COCO_CACHE"
echo "[{\"name\": \"LLaVA-1.5-7B\", \"run_dir\": \"$SEP\"}]" > "$ROOT/density_models.json"
$PY scripts/real_alpha/plot_multi_model_density.py \
    --models "$ROOT/density_models.json" --out "$ROOT/llava_density.pdf"

banner "DONE ($TAG)"
echo "  Table   : $ROOT/  (table.md / table.tex)"
echo "  Density : $ROOT/llava_density.{pdf,png,svg}"
echo "LLAVA_EXPERIMENT_DONE"
