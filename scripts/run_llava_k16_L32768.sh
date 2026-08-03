#!/usr/bin/env bash
#
# One arm of the LLaVA capacity x sparsity grid: k=16, latent_size=32768.
#
# latent_size is the TOTAL, so the modality-specific (separated / ours) methods
# keep the usual half-and-half split at 16384/side, giving 0.098% active
# coordinates per side. Everything else matches configs/real/coco_llava.yaml.
#
# Standalone on purpose: the six arms share nothing at run time except the
# read-only embedding caches, so they can be launched in parallel.
#
#   bash scripts/run_llava_k16_L32768.sh
#   CUDA_VISIBLE_DEVICES=1 bash scripts/run_llava_k16_L32768.sh        # pin a GPU
#   DENSITY=1 bash scripts/run_llava_k16_L32768.sh                     # + per-bin density stats
#
# Idempotent: returns immediately if this arm's table.md already exists.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f env.sh ] && source env.sh || true

PY=python; [ -x .venv/bin/python ] && PY=.venv/bin/python
K=16
L=32768
CFG="configs/real/coco_llava_k${K}_L${L}.yaml"
ROOT="outputs/real_exp_llava_coco_k${K}_L${L}"
COCO_CACHE=cache/llava_coco
IN_CACHE=cache/llava_imagenet
DENSITY="${DENSITY:-0}"
mkdir -p .log

# One checkpoint per method: six arms at once would otherwise fill the disk.
export SAE_SAVE_TOTAL_LIMIT="${SAE_SAVE_TOTAL_LIMIT:-1}"

banner(){ echo; echo "======== $* ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ========"; }

if [ ! -f "$COCO_CACHE/meta.json" ]; then
  echo "ERROR: $COCO_CACHE/meta.json missing. This arm reuses the caches built by" >&2
  echo "       scripts/run_llava_experiment.sh -- run that first." >&2
  exit 1
fi
[ -f "$IN_CACHE/meta.json" ] || \
  echo "WARNING: $IN_CACHE/meta.json missing; ImageNet zero-shot will fail, COCO still runs." >&2
[ -f "$CFG" ] || { echo "ERROR: $CFG not found" >&2; exit 1; }

banner "k=$K latent=$L (16384/side) -> $ROOT"
if [ -f "$ROOT/table.md" ]; then
  echo "skip: $ROOT/table.md exists"
else
  $PY scripts/real_alpha/run_real_v2.py --config "$CFG" 2>&1 \
      | tee -a ".log/llava_k${K}_L${L}.log"
fi

if [ "$DENSITY" = "1" ]; then
  SEP="$ROOT/separated/ckpt/final"
  if [ ! -d "$SEP" ]; then
    echo "WARNING: $SEP missing, skipping density" >&2
  else
    banner "k=$K latent=$L: density + per-bin statistics"
    $PY scripts/real_alpha/run_diagnostic_B.py --run-dir "$SEP" --cache-dir "$COCO_CACHE"
    $PY scripts/real_alpha/density_bin_stats.py \
        --run-dir "$SEP" --name "LLaVA-1.5-7B (k=$K, latent=$L)" \
        --out "$ROOT/density_bin_stats"
  fi
fi

banner "ARM DONE  k=$K latent=$L"
cat "$ROOT/table.md" 2>/dev/null || echo "(no table)"
echo "LLAVA_ARM_DONE k=$K L=$L"
