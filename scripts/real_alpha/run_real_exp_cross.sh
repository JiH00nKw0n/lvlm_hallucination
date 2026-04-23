#!/usr/bin/env bash
# Cross-dataset generalization: take SAE ckpts trained on SRC, evaluate on TGT.
# Default is SRC=coco → TGT=imagenet (no retraining).
#
# For each method in METHODS:
#   * Load COCO-trained checkpoint
#   * (Ours only) Rebuild Hungarian perm using TGT-domain paired pairs
#   * Run linprobe + zeroshot + dead_latents on TGT domain
#
# Outputs: outputs/real_exp_v1/<method>/<SRC>_to_<TGT>/{perm.npz,linprobe.json,zeroshot.json,dead_latents.json}
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
OUT="${OUT:-$ROOT/outputs/real_exp_v1}"
LOG="${LOG:-$ROOT/.log}"
mkdir -p "$OUT" "$LOG"

# Activate venv if present.
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

SRC="${SRC:-coco}"
TGT="${TGT:-imagenet}"
METHODS="${METHODS:-shared separated iso_align group_sparse vl_sae ours}"
MAX_PER_CLASS="${MAX_PER_CLASS:-1000}"
LP_EVAL_TOPK="${LP_EVAL_TOPK:-1}"

CACHE_COCO="$ROOT/cache/clip_b32_coco"
CACHE_IN="$ROOT/cache/clip_b32_imagenet"

if [ "$TGT" = "imagenet" ]; then
  TGT_CACHE="$CACHE_IN"
  PERM_EXTRA="--max-per-class $MAX_PER_CLASS"
else
  TGT_CACHE="$CACHE_COCO"
  PERM_EXTRA=""
fi

method_for_eval() {
  case "$1" in
    shared|separated) echo "$1" ;;
    iso_align|group_sparse) echo "aux" ;;
    ours) echo "ours" ;;
    vl_sae) echo "vl_sae" ;;
  esac
}

resolve_ckpt() {
  # Ours reuses Separated ckpt (on SRC domain).
  local method="$1"
  if [ "$method" = "ours" ]; then
    echo "$OUT/separated/$SRC/final"
  else
    echo "$OUT/$method/$SRC/final"
  fi
}

echo "[$(date '+%F %T')] cross eval: $SRC → $TGT, methods=$METHODS"

# Build Ours perm on TGT domain (using SRC-trained separated SAE).
if echo "$METHODS" | grep -qw "ours"; then
  out="$OUT/ours/${SRC}_to_${TGT}"
  mkdir -p "$out"
  if [ -f "$out/perm.npz" ]; then
    echo "[skip] perm ${SRC}_to_${TGT} (exists)"
  else
    echo "[perm] ${SRC}_to_${TGT}"
    python scripts/real_alpha/build_hungarian_perm.py \
        --ckpt "$OUT/separated/$SRC/final" \
        --dataset "$TGT" --cache-dir "$TGT_CACHE" \
        --output "$out/perm.npz" $PERM_EXTRA
  fi
fi

eval_one_cross() {
  local method="$1"
  local eval_method
  eval_method="$(method_for_eval "$method")"
  local ckpt
  ckpt="$(resolve_ckpt "$method")"
  local out="$OUT/$method/${SRC}_to_${TGT}"
  mkdir -p "$out"

  # Dead latents on TGT pairs (reflects how the SRC-trained SAE fires on TGT).
  if [ ! -f "$out/dead_latents.json" ]; then
    echo "[cross dead] $method / ${SRC}_to_${TGT}"
    if [ "$TGT" = "imagenet" ]; then dead_extra="--max-per-class $MAX_PER_CLASS"; else dead_extra=""; fi
    python scripts/real_alpha/eval_dead_latents.py \
        --ckpt "$ckpt" --method "$eval_method" --dataset "$TGT" \
        --cache-dir "$TGT_CACHE" --output "$out/dead_latents.json" $dead_extra
  fi

  # ImageNet downstream tasks (only valid when TGT=imagenet).
  if [ "$TGT" = "imagenet" ]; then
    if [ ! -f "$out/linprobe.json" ]; then
      echo "[cross linprobe] $method / ${SRC}_to_${TGT} (eval_topk=$LP_EVAL_TOPK)"
      python scripts/real_alpha/eval_imagenet_linprobe.py \
          --ckpt "$ckpt" --method "$eval_method" \
          --cache-dir "$TGT_CACHE" --output "$out/linprobe.json" \
          --max-per-class "$MAX_PER_CLASS" \
          --eval-topk "$LP_EVAL_TOPK"
    fi
    if [ ! -f "$out/zeroshot.json" ]; then
      echo "[cross zeroshot] $method / ${SRC}_to_${TGT}"
      local perm_flag=""
      if [ "$method" = "ours" ]; then perm_flag="--perm $OUT/ours/${SRC}_to_${TGT}/perm.npz"; fi
      python scripts/real_alpha/eval_imagenet_zeroshot.py \
          --ckpt "$ckpt" --method "$eval_method" \
          --cache-dir "$TGT_CACHE" --output "$out/zeroshot.json" $perm_flag
    fi
  fi

  # COCO retrieval (only valid when TGT=coco).
  if [ "$TGT" = "coco" ]; then
    if [ ! -f "$out/retrieval.json" ]; then
      echo "[cross retrieval] $method / ${SRC}_to_${TGT}"
      local perm_flag=""
      if [ "$method" = "ours" ]; then perm_flag="--perm $OUT/ours/${SRC}_to_${TGT}/perm.npz"; fi
      python scripts/real_alpha/eval_coco_retrieval.py \
          --ckpt "$ckpt" --method "$eval_method" \
          --cache-dir "$TGT_CACHE" --output "$out/retrieval.json" $perm_flag
    fi
  fi
}

for method in $METHODS; do
  eval_one_cross "$method"
done

echo "[$(date '+%F %T')] cross eval done"
