#!/usr/bin/env bash
# Multimodal Monosemanticity Score (MMS, Kaushik et al. 2026).
# External similarity encoder = MetaCLIP B/32 (different from training's CLIP B/32).
#
# Usage on elice-40g:
#   # COCO test (default):
#   nohup bash scripts/run_eval_mms_coco.sh > .log/mms_coco.log 2>&1 & disown
#   # CC3M validation:
#   DATASET=cc3m SPLIT=validation \
#     TRAIN_CACHE=cache/clip_b32_cc3m_val EXT_CACHE=cache/metaclip_b32_cc3m_val \
#     OUT_ROOT=outputs/real_exp_cc3m/mms_cc3m_val \
#     bash scripts/run_eval_mms_coco.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

DATASET=${DATASET:-coco}
SPLIT=${SPLIT:-test}
TRAIN_CACHE=${TRAIN_CACHE:-cache/clip_b32_coco}
EXT_CACHE=${EXT_CACHE:-cache/metaclip_b32_coco}
ROOT=${ROOT:-outputs/real_exp_cc3m}
OUT_ROOT=${OUT_ROOT:-$ROOT/mms_${DATASET}_${SPLIT}}
LOG_DIR=${LOG_DIR:-.log}
mkdir -p "$LOG_DIR"

now() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(now)] $*"; }

declare -a VARIANTS=(shared separated iso_align group_sparse ours)
declare -A METHOD=(
  [shared]=shared [separated]=separated [iso_align]=aux [group_sparse]=aux [ours]=ours
)
declare -A CKPT=(
  [shared]="$ROOT/shared/ckpt/final"
  [separated]="$ROOT/separated/ckpt/final"
  [iso_align]="$ROOT/iso_align/ckpt/final"
  [group_sparse]="$ROOT/group_sparse/ckpt/final"
  [ours]="$ROOT/separated/ckpt/final"
)
# Single training-cache perm (built once by run_real_v2.py).
# Falls back to legacy per-dataset path for backwards compat with old runs.
PERM_OURS="$ROOT/ours/perm.npz"
if [[ ! -f "$PERM_OURS" ]]; then
  PERM_OURS="$ROOT/ours/coco/perm.npz"  # legacy fallback
fi

global_start=$(date +%s)
log "BEGIN MMS eval dataset=$DATASET split=$SPLIT ext=$EXT_CACHE out=$OUT_ROOT"

for variant in "${VARIANTS[@]}"; do
  method="${METHOD[$variant]}"
  ckpt="${CKPT[$variant]}"
  out="$OUT_ROOT/$variant"

  log "BEGIN variant=$variant method=$method"
  if [[ ! -d "$ckpt" ]]; then
    log "  SKIP — ckpt missing: $ckpt"; continue
  fi

  variant_t0=$(date +%s)
  perm_arg=()
  if [[ "$method" == "ours" ]]; then
    perm_arg=(--perm "$PERM_OURS")
  fi

  python scripts/real_alpha/eval_mms.py \
    --variant "$variant" --method "$method" \
    --ckpt "$ckpt" \
    --train-cache "$TRAIN_CACHE" \
    --ext-cache "$EXT_CACHE" \
    --dataset "$DATASET" \
    --split "$SPLIT" \
    --out "$out" \
    "${perm_arg[@]}" \
    --device cuda 2>&1 | sed "s|^|  [${variant}] |"

  dt=$(( $(date +%s) - variant_t0 ))
  log "DONE  variant=$variant dt=${dt}s"
done

global_dt=$(( $(date +%s) - global_start ))
log "ALL DONE total=${global_dt}s"
