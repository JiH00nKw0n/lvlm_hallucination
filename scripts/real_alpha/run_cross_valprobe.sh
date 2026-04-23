#!/usr/bin/env bash
# Cross-dataset valprobe: SAE trained on COCO, evaluated on ImageNet val.
# For each method, loads the COCO-trained ckpt, encodes ImageNet val, applies
# dominant-slot masking (threshold=0.1), and runs val fit+eval linprobe.
#
# Output: outputs/real_exp_v1/<method>/coco_to_imagenet/valprobe_crossval.json
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
OUT="${OUT:-$ROOT/outputs/real_exp_v1}"
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

CACHE_IN="$ROOT/cache/clip_b32_imagenet"
T="${THRESHOLD:-0.1}"
MAX_PER_CLASS="${MAX_PER_CLASS:-1000}"

methods=("shared:shared" "separated:separated" "iso_align:aux" "group_sparse:aux" "vl_sae:vl_sae" "shared_enc:shared_enc")

# Build Ours perm on ImageNet domain if missing (fresh OUT root).
perm_path="$OUT/ours/coco_to_imagenet/perm.npz"
if [ ! -f "$perm_path" ]; then
  mkdir -p "$OUT/ours/coco_to_imagenet"
  echo "[$(date '+%F %T')] perm coco→imagenet (ImageNet pairs, COCO-trained separated)"
  python scripts/real_alpha/build_hungarian_perm.py \
      --ckpt "$OUT/separated/coco/final" \
      --dataset imagenet --cache-dir "$CACHE_IN" \
      --output "$perm_path" --max-per-class "$MAX_PER_CLASS"
fi

for info in "${methods[@]}"; do
  m=${info%:*}; em=${info#*:}
  out="$OUT/$m/coco_to_imagenet/valprobe_crossval.json"
  if [ -f "$out" ]; then echo "[skip] $m"; continue; fi
  mkdir -p "$OUT/$m/coco_to_imagenet"
  echo "[$(date '+%F %T')] cross_valprobe $m  (COCO ckpt → ImageNet val)"
  python scripts/real_alpha/eval_imagenet_valprobe.py \
      --ckpt "$OUT/$m/coco/final" \
      --method "$em" --cache-dir "$CACHE_IN" \
      --output "$out" --dominant-threshold "$T"
done

# Ours: use COCO-trained separated ckpt + ImageNet-domain perm (already built at
# outputs/real_exp_v1/ours/coco_to_imagenet/perm.npz).
out="$OUT/ours/coco_to_imagenet/valprobe_crossval.json"
if [ ! -f "$out" ]; then
  mkdir -p "$OUT/ours/coco_to_imagenet"
  echo "[$(date '+%F %T')] cross_valprobe ours (COCO separated + ImageNet perm)"
  python scripts/real_alpha/eval_imagenet_valprobe.py \
      --ckpt "$OUT/separated/coco/final" \
      --method ours --cache-dir "$CACHE_IN" \
      --perm "$OUT/ours/coco_to_imagenet/perm.npz" \
      --output "$out" --dominant-threshold "$T"
fi

echo "[$(date '+%F %T')] cross_valprobe sweep done"
