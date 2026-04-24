#!/usr/bin/env bash
# Sweep valprobe + trainprobe at dominant_threshold=0.5 across all 6 methods.
#
# Assumes run_real_exp_matrix.sh already trained the 5 method ckpts and Ours
# reuses Separated. Writes output to:
#   outputs/real_exp_v1/<method>/imagenet/valprobe_t50.json
#   outputs/real_exp_v1/<method>/imagenet/trainprobe_t50.json
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
OUT="${OUT:-$ROOT/outputs/real_exp_v1}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

CACHE_IN="$ROOT/cache/clip_b32_imagenet"
T="${THRESHOLD:-0.5}"

methods=("shared:shared" "separated:separated" "iso_align:aux" "group_sparse:aux" "vl_sae:vl_sae")

# ---- valprobe threshold=0.5 (val fit+eval) ----
for info in "${methods[@]}"; do
  m=${info%:*}; em=${info#*:}
  out="$OUT/$m/imagenet/valprobe_t50.json"
  if [ -f "$out" ]; then echo "[skip] $m valprobe_t50"; continue; fi
  echo "[$(date '+%F %T')] valprobe_t50 $m"
  python scripts/real_alpha/eval_imagenet_valprobe.py \
      --ckpt "$OUT/$m/imagenet/final" \
      --method "$em" --cache-dir "$CACHE_IN" \
      --output "$out" --dominant-threshold "$T"
done
# Ours
out="$OUT/ours/imagenet/valprobe_t50.json"
if [ ! -f "$out" ]; then
  echo "[$(date '+%F %T')] valprobe_t50 ours"
  python scripts/real_alpha/eval_imagenet_valprobe.py \
      --ckpt "$OUT/separated/imagenet/final" \
      --method ours --cache-dir "$CACHE_IN" \
      --perm "$OUT/ours/imagenet/perm.npz" \
      --output "$out" --dominant-threshold "$T"
fi

# ---- trainprobe threshold=0.5 (train fit, val eval) ----
for info in "${methods[@]}"; do
  m=${info%:*}; em=${info#*:}
  out="$OUT/$m/imagenet/trainprobe_t50.json"
  if [ -f "$out" ]; then echo "[skip] $m trainprobe_t50"; continue; fi
  echo "[$(date '+%F %T')] trainprobe_t50 $m"
  python scripts/real_alpha/eval_imagenet_trainprobe.py \
      --ckpt "$OUT/$m/imagenet/final" \
      --method "$em" --cache-dir "$CACHE_IN" \
      --output "$out" --dominant-threshold "$T" --lp-epochs 10
done
# Ours
out="$OUT/ours/imagenet/trainprobe_t50.json"
if [ ! -f "$out" ]; then
  echo "[$(date '+%F %T')] trainprobe_t50 ours"
  python scripts/real_alpha/eval_imagenet_trainprobe.py \
      --ckpt "$OUT/separated/imagenet/final" \
      --method ours --cache-dir "$CACHE_IN" \
      --perm "$OUT/ours/imagenet/perm.npz" \
      --output "$out" --dominant-threshold "$T" --lp-epochs 10
fi

echo "[$(date '+%F %T')] sweep done"
