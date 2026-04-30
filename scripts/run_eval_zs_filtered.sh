#!/usr/bin/env bash
# Re-run ImageNet zero-shot with always-on latent filtering for all 5 variants.
# Drops latents with image-side fire_rate > MAX_FIRE_RATE (default 0.5).
#
# Usage on elice-40g:
#   ROOT=outputs/real_exp_cc3m_s0 \
#     bash scripts/run_eval_zs_filtered.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

ROOT=${ROOT:?"set ROOT=outputs/real_exp_cc3m_s<seed>"}
CACHE_DIR=${CACHE_DIR:-cache/clip_b32_imagenet}
MAX_FIRE_RATE=${MAX_FIRE_RATE:-0.5}
LOG_DIR=${LOG_DIR:-.log}
mkdir -p "$LOG_DIR"

declare -a VARIANTS=(shared separated iso_align group_sparse ours)
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

PERM_OURS="$ROOT/ours/perm.npz"
[[ ! -f "$PERM_OURS" ]] && PERM_OURS="$ROOT/ours/coco/perm.npz"

now() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(now)] $*"; }

log "BEGIN filtered zs root=$ROOT max_fire_rate=$MAX_FIRE_RATE"
for variant in "${VARIANTS[@]}"; do
  method="${METHOD[$variant]}"
  ckpt="${CKPT[$variant]}"
  out="$ROOT/$variant/imagenet/zeroshot_filtered_fr${MAX_FIRE_RATE}.json"

  log "BEGIN variant=$variant method=$method"
  if [[ ! -d "$ckpt" ]]; then log "  SKIP — ckpt missing: $ckpt"; continue; fi
  if [[ -f "$out" ]];      then log "  SKIP — exists: $out"; continue; fi

  perm_arg=()
  [[ "$method" == "ours" ]] && perm_arg=(--perm "$PERM_OURS")

  python scripts/real_alpha/eval_imagenet_zeroshot_filtered.py \
    --variant "$variant" --method "$method" --ckpt "$ckpt" \
    --cache-dir "$CACHE_DIR" --output "$out" \
    --max-fire-rate "$MAX_FIRE_RATE" \
    "${perm_arg[@]}" --device cuda 2>&1 | sed "s|^|  [${variant}] |"
done
log "ALL DONE"
