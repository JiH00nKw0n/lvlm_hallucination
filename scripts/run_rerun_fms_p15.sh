#!/usr/bin/env bash
# Re-run val FMS to record acc_p1..acc_p5 (so FMS@1, @2, @3, @4, @5 all derivable).
# Backs up the current ones to *_p15bak.{csv,json}.
#
# Usage on elice-40g:
#   nohup bash scripts/run_rerun_fms_p15.sh > .log/fms_p15.log 2>&1 & disown

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

CLIP_CACHE=${CLIP_CACHE:-cache/clip_b32_imagenet}
DINO_CACHE=${DINO_CACHE:-cache/dinov2_b14_imagenet}
OUT_ROOT=${OUT_ROOT:-outputs/real_exp_cc3m/monosemanticity}
LOG_DIR=${LOG_DIR:-.log}
mkdir -p "$LOG_DIR"

now() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(now)] $*"; }

declare -a VARIANTS=(shared separated iso_align group_sparse)
declare -A METHOD=( [shared]=shared [separated]=separated [iso_align]=aux [group_sparse]=aux )
declare -A CKPT=(
  [shared]="outputs/real_exp_cc3m/shared/ckpt/final"
  [separated]="outputs/real_exp_cc3m/separated/ckpt/final"
  [iso_align]="outputs/real_exp_cc3m/iso_align/ckpt/final"
  [group_sparse]="outputs/real_exp_cc3m/group_sparse/ckpt/final"
)

global_start=$(date +%s)
log "BEGIN val FMS rerun (record acc_p1..acc_p5)"

for variant in "${VARIANTS[@]}"; do
  method="${METHOD[$variant]}"
  ckpt="${CKPT[$variant]}"
  out="$OUT_ROOT/$variant"

  log "BEGIN variant=$variant"
  if [[ ! -d "$ckpt" ]]; then
    log "  SKIP — ckpt missing"; continue
  fi

  variant_t0=$(date +%s)
  python scripts/real_alpha/eval_monosemanticity.py \
    --variant "$variant" --method "$method" \
    --ckpt "$ckpt" \
    --clip-cache "$CLIP_CACHE" \
    --dino-cache "$DINO_CACHE" \
    --out "$out" \
    --metrics fms \
    --device cuda 2>&1 | sed "s|^|  [${variant}] |"
  variant_dt=$(( $(date +%s) - variant_t0 ))
  log "DONE  variant=$variant dt=${variant_dt}s"
done

global_dt=$(( $(date +%s) - global_start ))
log "ALL DONE total=${global_dt}s"
