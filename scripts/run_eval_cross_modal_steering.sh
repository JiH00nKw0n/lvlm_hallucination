#!/usr/bin/env bash
# Cross-modal SAE steering on COCO test split for CC3M-trained variants.
#
# Usage on elice-40g:
#   nohup bash scripts/run_eval_cross_modal_steering.sh > .log/cms.log 2>&1 & disown

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

CACHE_DIR=${CACHE_DIR:-cache/clip_b32_coco}
CAPTIONS_JSON=${CAPTIONS_JSON:-${CACHE_DIR}/captions.json}
ROOT=${ROOT:-outputs/real_exp_cc3m}
OUT_ROOT=${OUT_ROOT:-$ROOT/cross_modal_steering}
LOG_DIR=${LOG_DIR:-.log}
mkdir -p "$LOG_DIR"

ALPHAS=${ALPHAS:-0,0.1,0.25,0.5,1.0,2.0}
N_BASE=${N_BASE:-100}

now() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(now)] $*"; }

# Variants to run. Steering only makes sense for methods with cross-modal
# correspondence; we report iso_align, group_sparse, ours.
declare -a VARIANTS=(iso_align group_sparse ours)
declare -A METHOD=(
  [shared]=shared [separated]=separated
  [iso_align]=aux [group_sparse]=aux [ours]=ours
)
declare -A CKPT=(
  [shared]="$ROOT/shared/ckpt/final"
  [separated]="$ROOT/separated/ckpt/final"
  [iso_align]="$ROOT/iso_align/ckpt/final"
  [group_sparse]="$ROOT/group_sparse/ckpt/final"
  [ours]="$ROOT/separated/ckpt/final"
)
# Single training-cache perm (built once by run_real_v2.py from data.cache_dir).
# Falls back to legacy per-dataset path for backwards compat with old runs.
PERM_OURS="$ROOT/ours/perm.npz"
if [[ ! -f "$PERM_OURS" ]]; then
  PERM_OURS="$ROOT/ours/coco/perm.npz"  # legacy fallback
fi

global_start=$(date +%s)
log "BEGIN cross-modal steering on COCO test"

for variant in "${VARIANTS[@]}"; do
  method="${METHOD[$variant]}"
  ckpt="${CKPT[$variant]}"
  out="$OUT_ROOT/$variant"

  log "BEGIN variant=$variant method=$method"
  if [[ ! -d "$ckpt" ]]; then
    log "  SKIP — ckpt missing: $ckpt"; continue
  fi
  if [[ -f "$out/summary.json" ]]; then
    log "  SKIP — already done: $out/summary.json"; continue
  fi

  variant_t0=$(date +%s)
  perm_arg=()
  if [[ "$method" == "ours" ]]; then
    perm_arg=(--perm "$PERM_OURS")
  fi

  python scripts/real_alpha/eval_cross_modal_steering.py \
    --variant "$variant" --method "$method" \
    --ckpt "$ckpt" \
    --cache-dir "$CACHE_DIR" \
    --captions-json "$CAPTIONS_JSON" \
    --alphas "$ALPHAS" \
    --n-base-images "$N_BASE" \
    --out "$out" \
    "${perm_arg[@]}" \
    --device cuda 2>&1 | sed "s|^|  [${variant}] |"

  dt=$(( $(date +%s) - variant_t0 ))
  log "DONE  variant=$variant dt=${dt}s"
done

global_dt=$(( $(date +%s) - global_start ))
log "ALL DONE total=${global_dt}s"
