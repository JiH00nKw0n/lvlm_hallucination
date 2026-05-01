#!/usr/bin/env bash
# Re-run CCS with both 0.9 (truly always-on) and 0.5 (broadband generic) filters,
# AND compute latent uniqueness across the 1000 class top-1 picks.
#
# Per variant, runs the eval twice and writes:
#   ccs_summary.json                ← max_fire_frac=0.5 (matches our prior run)
#   ccs_per_class.csv               ← max_fire_frac=0.5
#   ccs_summary_t0p9.json           ← max_fire_frac=0.9 (new, conservative)
#   ccs_per_class_t0p9.csv          ← max_fire_frac=0.9
#
# Usage on elice-40g:
#   nohup bash scripts/run_rerun_ccs_uniqueness.sh > .log/ccs_uniq.log 2>&1 & disown

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

for variant in "${VARIANTS[@]}"; do
  method="${METHOD[$variant]}"
  ckpt="${CKPT[$variant]}"
  out="$OUT_ROOT/$variant"

  log "BEGIN variant=$variant method=$method"
  if [[ ! -d "$ckpt" ]]; then
    log "  SKIP variant=$variant — ckpt dir missing: $ckpt"
    continue
  fi

  variant_t0=$(date +%s)

  # Run 1: max_fire_frac=0.5 (overwrites primary ccs_*)
  python scripts/real_alpha/eval_monosemanticity.py \
    --variant "$variant" --method "$method" \
    --ckpt "$ckpt" \
    --clip-cache "$CLIP_CACHE" \
    --dino-cache "$DINO_CACHE" \
    --out "$out" \
    --metrics ccs \
    --ccs-max-fire-frac 0.5 \
    --device cuda 2>&1 | sed "s|^|  [${variant} t0.5] |"

  # Run 2: max_fire_frac=0.9. Run into a temporary subdir, then move files.
  tmpdir="$out/_t0p9"
  mkdir -p "$tmpdir"
  python scripts/real_alpha/eval_monosemanticity.py \
    --variant "$variant" --method "$method" \
    --ckpt "$ckpt" \
    --clip-cache "$CLIP_CACHE" \
    --dino-cache "$DINO_CACHE" \
    --out "$tmpdir" \
    --metrics ccs \
    --ccs-max-fire-frac 0.9 \
    --device cuda 2>&1 | sed "s|^|  [${variant} t0.9] |"
  mv "$tmpdir/ccs_per_class.csv" "$out/ccs_per_class_t0p9.csv"
  mv "$tmpdir/ccs_summary.json"  "$out/ccs_summary_t0p9.json"
  rm -rf "$tmpdir"

  variant_dt=$(( $(date +%s) - variant_t0 ))
  log "DONE  variant=$variant dt=${variant_dt}s"
done

global_dt=$(( $(date +%s) - global_start ))
log "ALL DONE total=${global_dt}s"
