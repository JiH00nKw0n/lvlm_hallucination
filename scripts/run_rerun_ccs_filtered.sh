#!/usr/bin/env bash
# Re-run CCS only with an upper-bound on fire-count: latents firing on more than
# `max_fire_frac` of all N images are dropped from the CCS computation. This
# answers the question "do 'always-on' latents inflate CCS-Top1?"
#
# Backs up the existing CCS outputs to *_unfiltered.bak before overwriting.
#
# Usage on elice-40g:
#   MAX_FIRE_FRAC=0.5 nohup bash scripts/run_rerun_ccs_filtered.sh > .log/ccs_filt.log 2>&1 & disown

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
MAX_FIRE_FRAC=${MAX_FIRE_FRAC:-0.5}
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
log "BEGIN CCS rerun max_fire_frac=$MAX_FIRE_FRAC"

for variant in "${VARIANTS[@]}"; do
  method="${METHOD[$variant]}"
  ckpt="${CKPT[$variant]}"
  out="$OUT_ROOT/$variant"

  log "BEGIN variant=$variant method=$method"
  if [[ ! -d "$ckpt" ]]; then
    log "  SKIP variant=$variant — ckpt dir missing: $ckpt"
    continue
  fi

  # Back up original CCS outputs
  if [[ -f "$out/ccs_per_class.csv" && ! -f "$out/ccs_per_class_unfiltered.bak.csv" ]]; then
    cp -p "$out/ccs_per_class.csv"   "$out/ccs_per_class_unfiltered.bak.csv"
  fi
  if [[ -f "$out/ccs_summary.json" && ! -f "$out/ccs_summary_unfiltered.bak.json" ]]; then
    cp -p "$out/ccs_summary.json"    "$out/ccs_summary_unfiltered.bak.json"
  fi

  variant_t0=$(date +%s)
  python scripts/real_alpha/eval_monosemanticity.py \
    --variant "$variant" --method "$method" \
    --ckpt "$ckpt" \
    --clip-cache "$CLIP_CACHE" \
    --dino-cache "$DINO_CACHE" \
    --out "$out" \
    --metrics ccs \
    --ccs-max-fire-frac "$MAX_FIRE_FRAC" \
    --device cuda 2>&1 | sed "s/^/  [$variant] /"
  variant_dt=$(( $(date +%s) - variant_t0 ))
  log "DONE  variant=$variant dt=${variant_dt}s"
done

global_dt=$(( $(date +%s) - global_start ))
log "ALL DONE total=${global_dt}s max_fire_frac=$MAX_FIRE_FRAC"
