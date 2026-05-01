#!/usr/bin/env bash
# Re-run FMS with paper-faithful protocol (iterative root-cut + depth-by-depth global,
# in-sample, no CV). Backs up the old CV-based FMS outputs to *_cvprotocol.bak.
#
# Usage on elice-40g:
#   nohup bash scripts/run_rerun_fms.sh > .log/fms_rerun.log 2>&1 & disown

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

fmt_dur() { local s=$1; printf '%02d:%02d:%02d' $((s/3600)) $(( (s%3600)/60 )) $((s%60)); }
now()     { date '+%Y-%m-%d %H:%M:%S'; }
log()     { echo "[$(now)] $*"; }

global_start=$(date +%s)

declare -a VARIANTS=(shared separated iso_align group_sparse)
declare -A METHOD=( [shared]=shared [separated]=separated [iso_align]=aux [group_sparse]=aux )
declare -A CKPT=(
  [shared]="outputs/real_exp_cc3m/shared/ckpt/final"
  [separated]="outputs/real_exp_cc3m/separated/ckpt/final"
  [iso_align]="outputs/real_exp_cc3m/iso_align/ckpt/final"
  [group_sparse]="outputs/real_exp_cc3m/group_sparse/ckpt/final"
)

declare -a VARIANT_DURATIONS=()
n_variants=${#VARIANTS[@]}

for idx in "${!VARIANTS[@]}"; do
  variant="${VARIANTS[$idx]}"
  method="${METHOD[$variant]}"
  ckpt="${CKPT[$variant]}"
  out="$OUT_ROOT/$variant"

  elapsed_global=$(( $(date +%s) - global_start ))
  if [[ ${#VARIANT_DURATIONS[@]} -gt 0 ]]; then
    sum=0; for d in "${VARIANT_DURATIONS[@]}"; do sum=$((sum+d)); done
    avg=$(( sum / ${#VARIANT_DURATIONS[@]} ))
    remaining=$(( avg * (n_variants - idx) ))
    log "# global elapsed=$(fmt_dur $elapsed_global) avg/variant=$(fmt_dur $avg) projected_remaining=$(fmt_dur $remaining)"
  else
    log "# global elapsed=$(fmt_dur $elapsed_global) (first variant)"
  fi

  log "BEGIN variant=$variant ($((idx+1))/$n_variants) method=$method"
  if [[ ! -d "$ckpt" ]]; then
    log "  SKIP variant=$variant — ckpt dir missing: $ckpt"
    continue
  fi

  # Back up old CV-protocol FMS outputs before overwriting
  if [[ -f "$out/fms_per_class.csv" ]]; then
    cp -p "$out/fms_per_class.csv" "$out/fms_per_class_cvprotocol.bak.csv" || true
  fi
  if [[ -f "$out/fms_summary.json" ]]; then
    cp -p "$out/fms_summary.json" "$out/fms_summary_cvprotocol.bak.json" || true
  fi

  variant_t0=$(date +%s)
  python scripts/real_alpha/eval_monosemanticity.py \
    --variant "$variant" --method "$method" \
    --ckpt "$ckpt" \
    --clip-cache "$CLIP_CACHE" \
    --dino-cache "$DINO_CACHE" \
    --out "$out" \
    --metrics fms \
    --device cuda 2>&1 | sed "s/^/  [$variant] /"

  variant_dt=$(( $(date +%s) - variant_t0 ))
  VARIANT_DURATIONS+=("$variant_dt")
  log "DONE  variant=$variant dt=$(fmt_dur $variant_dt)"
done

global_dt=$(( $(date +%s) - global_start ))
log "ALL DONE total=$(fmt_dur $global_dt) variants_run=${#VARIANT_DURATIONS[@]}"

log "Rebuilding summary table…"
python scripts/real_alpha/build_monosemanticity_table.py \
  --root "$OUT_ROOT" \
  --variants ${VARIANTS[*]} \
  --out "$OUT_ROOT/SUMMARY.md" || log "summary build failed (continuing)"
log "Wrote $OUT_ROOT/SUMMARY.md"
