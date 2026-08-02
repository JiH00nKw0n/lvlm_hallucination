#!/usr/bin/env bash
#
# Sparsity sweep on the LLaVA-1.5 EOS embeddings: rerun the Table-1 comparison
# at k = 8, 16, 32 against the k = 256 default.
#
# Only k changes. latent_size, epochs, batch, lr, schedule, data and the five
# methods are the ones in configs/real/coco_llava.yaml, so a difference between
# arms is attributable to sparsity and not to capacity or optimization.
#
# The embedding caches are the expensive part and this script does NOT rebuild
# them -- it reuses cache/llava_coco and cache/llava_imagenet from the main run.
# Run scripts/run_llava_experiment.sh first if they are missing.
#
#   bash scripts/run_llava_k_sweep.sh                # k = 8, 16, 32
#   KS="16" bash scripts/run_llava_k_sweep.sh        # one arm
#   DENSITY=1 bash scripts/run_llava_k_sweep.sh      # + per-bin density stats
#
# Each arm is idempotent: an arm whose table.md already exists is skipped, so a
# crashed sweep resumes by re-invoking.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f env.sh ] && source env.sh || true

PY=python; [ -x .venv/bin/python ] && PY=.venv/bin/python
KS="${KS:-8 16 32}"
DENSITY="${DENSITY:-0}"
COCO_CACHE=cache/llava_coco
IN_CACHE=cache/llava_imagenet
mkdir -p .log

banner(){ echo; echo "======== $* ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ========"; }

if [ ! -f "$COCO_CACHE/meta.json" ]; then
  echo "ERROR: $COCO_CACHE/meta.json missing. The sweep reuses the caches built by" >&2
  echo "       scripts/run_llava_experiment.sh -- run that first." >&2
  exit 1
fi
if [ ! -f "$IN_CACHE/meta.json" ]; then
  echo "WARNING: $IN_CACHE/meta.json missing; ImageNet zero-shot will fail." >&2
  echo "         COCO recon + retrieval will still run." >&2
fi

# The k=256 SAE keeps 1 checkpoint per method to save disk; do the same here,
# since the sweep triples the number of checkpoints on disk.
export SAE_SAVE_TOTAL_LIMIT="${SAE_SAVE_TOTAL_LIMIT:-1}"

for K in $KS; do
  CFG="configs/real/coco_llava_k${K}.yaml"
  ROOT="outputs/real_exp_llava_coco_k${K}"
  [ -f "$CFG" ] || { echo "ERROR: $CFG not found" >&2; exit 1; }

  banner "k=${K}: Table-1 pipeline -> $ROOT"
  if [ -f "$ROOT/table.md" ]; then
    echo "skip: $ROOT/table.md exists"
  else
    $PY scripts/real_alpha/run_real_v2.py --config "$CFG" 2>&1 | tee -a ".log/llava_k${K}.log"
  fi

  if [ "$DENSITY" = "1" ]; then
    SEP="$ROOT/separated/ckpt/final"
    if [ ! -d "$SEP" ]; then
      echo "WARNING: $SEP missing, skipping density for k=${K}" >&2
    else
      banner "k=${K}: density + per-bin statistics"
      $PY scripts/real_alpha/run_diagnostic_B.py --run-dir "$SEP" --cache-dir "$COCO_CACHE"
      $PY scripts/real_alpha/density_bin_stats.py \
          --run-dir "$SEP" --name "LLaVA-1.5-7B (k=${K})" \
          --out "$ROOT/density_bin_stats"
    fi
  fi
done

banner "SWEEP DONE"
for K in $KS; do
  echo; echo "---------------- k=${K} ----------------"
  cat "outputs/real_exp_llava_coco_k${K}/table.md" 2>/dev/null || echo "(no table)"
done
echo
echo "---------------- k=256 (main run, for reference) ----------------"
cat "outputs/real_exp_llava_coco/table.md" 2>/dev/null || echo "(main run not present)"
echo
echo "LLAVA_K_SWEEP_DONE"
