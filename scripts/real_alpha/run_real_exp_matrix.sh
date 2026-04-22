#!/usr/bin/env bash
# Real-data downstream experiment: 5 methods × (ImageNet k=1, COCO k=8).
# Runs on server (elice-40g) with pre-cached CLIP ViT-B/32 embeddings.
#
# Outputs to outputs/real_exp_v1/<method>/<dataset>/:
#   final/              — HF-saved SAE checkpoint
#   perm.npz            — (Ours only) Hungarian text→image slot perm
#   recon.json          — reconstruction error on eval split
#   linprobe.json       — (ImageNet) top-1 linear-probe accuracy
#   zeroshot.json       — (ImageNet) top-1 zero-shot accuracy
#   retrieval.json      — (COCO)     I↔T R@{1,5,10}
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
OUT="${OUT:-$ROOT/outputs/real_exp_v1}"
LOG="${LOG:-$ROOT/.log}"
mkdir -p "$OUT" "$LOG"

# Activate venv if present (idempotent).
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
fi

CACHE_COCO="$ROOT/cache/clip_b32_coco"
CACHE_IN="$ROOT/cache/clip_b32_imagenet"

EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-1024}"
LR="${LR:-5e-4}"
LATENT="${LATENT:-8192}"
SEED="${SEED:-0}"
MAX_PER_CLASS="${MAX_PER_CLASS:-1000}"

# iso/group default hyperparameters (from synthetic followup15 baselines).
ISO_LAMBDA="${ISO_LAMBDA:-1e-4}"
GS_LAMBDA="${GS_LAMBDA:-0.05}"

COMMON_TRAIN="--epochs $EPOCHS --batch-size $BATCH --lr $LR --latent $LATENT --seed $SEED"

echo "[$(date '+%F %T')] start real_exp_v1 matrix"

# ---------------------------------------------------------------------------
# Section: train 4 checkpoints per dataset (ours reuses separated)
# ---------------------------------------------------------------------------
run_train() {
  local dataset="$1" k="$2" method="$3" extra="$4"
  local cache="$5"
  local out="$OUT/$method/$dataset"
  if [ -d "$out/final" ]; then
    echo "[skip] $method/$dataset (final exists)"
    return
  fi
  mkdir -p "$out"
  echo "[train] $method / $dataset (k=$k)"
  python scripts/real_alpha/train_real_sae.py \
      --dataset "$dataset" --cache-dir "$cache" \
      --output-dir "$out" \
      --k "$k" $COMMON_TRAIN $extra
}

for dataset in imagenet coco; do
  if [ "$dataset" = "imagenet" ]; then
    K=1; CACHE="$CACHE_IN"
    IN_EXTRA="--max-per-class $MAX_PER_CLASS"
  else
    K=8; CACHE="$CACHE_COCO"
    IN_EXTRA=""
  fi

  run_train "$dataset" "$K" "shared"     "--variant one_sae $IN_EXTRA" "$CACHE"
  run_train "$dataset" "$K" "separated"  "--variant two_sae $IN_EXTRA" "$CACHE"
  run_train "$dataset" "$K" "iso_align"  "--variant aux_sae --aux-loss iso_align    --aux-lambda $ISO_LAMBDA $IN_EXTRA" "$CACHE"
  run_train "$dataset" "$K" "group_sparse" "--variant aux_sae --aux-loss group_sparse --aux-lambda $GS_LAMBDA $IN_EXTRA" "$CACHE"
done

# ---------------------------------------------------------------------------
# Section: build Hungarian perm for Ours
# ---------------------------------------------------------------------------
for dataset in imagenet coco; do
  if [ "$dataset" = "imagenet" ]; then
    CACHE="$CACHE_IN"
    PERM_EXTRA="--max-per-class $MAX_PER_CLASS"
  else
    CACHE="$CACHE_COCO"
    PERM_EXTRA=""
  fi
  out="$OUT/ours/$dataset"
  mkdir -p "$out"
  if [ -f "$out/perm.npz" ]; then
    echo "[skip] perm $dataset (exists)"
  else
    echo "[perm] $dataset"
    python scripts/real_alpha/build_hungarian_perm.py \
        --ckpt "$OUT/separated/$dataset/final" \
        --dataset "$dataset" --cache-dir "$CACHE" \
        --output "$out/perm.npz" $PERM_EXTRA
  fi
done

# ---------------------------------------------------------------------------
# Section: evaluate every method × every applicable metric
# ---------------------------------------------------------------------------
method_for_eval() {
  case "$1" in
    shared|separated) echo "$1" ;;
    iso_align|group_sparse) echo "aux" ;;
    ours) echo "ours" ;;
  esac
}

resolve_ckpt() {
  # Ours reuses Separated checkpoint.
  local method="$1" dataset="$2"
  if [ "$method" = "ours" ]; then
    echo "$OUT/separated/$dataset/final"
  else
    echo "$OUT/$method/$dataset/final"
  fi
}

eval_one() {
  local method="$1" dataset="$2"
  local eval_method
  eval_method="$(method_for_eval "$method")"
  local ckpt
  ckpt="$(resolve_ckpt "$method" "$dataset")"
  local out="$OUT/$method/$dataset"
  mkdir -p "$out"

  # Recon
  if [ ! -f "$out/recon.json" ]; then
    echo "[eval recon] $method / $dataset"
    if [ "$dataset" = "coco" ]; then split="test"; cache="$CACHE_COCO"; else split="val"; cache="$CACHE_IN"; fi
    python scripts/real_alpha/eval_recon_downstream.py \
        --ckpt "$ckpt" --method "$eval_method" --dataset "$dataset" \
        --cache-dir "$cache" --split "$split" --output "$out/recon.json"
  fi

  if [ "$dataset" = "imagenet" ]; then
    # Linear probe
    if [ ! -f "$out/linprobe.json" ]; then
      echo "[eval linprobe] $method / $dataset"
      python scripts/real_alpha/eval_imagenet_linprobe.py \
          --ckpt "$ckpt" --method "$eval_method" \
          --cache-dir "$CACHE_IN" --output "$out/linprobe.json" \
          --max-per-class "$MAX_PER_CLASS"
    fi
    # Zero-shot
    if [ ! -f "$out/zeroshot.json" ]; then
      echo "[eval zeroshot] $method / $dataset"
      local perm_flag=""
      if [ "$method" = "ours" ]; then perm_flag="--perm $OUT/ours/$dataset/perm.npz"; fi
      python scripts/real_alpha/eval_imagenet_zeroshot.py \
          --ckpt "$ckpt" --method "$eval_method" \
          --cache-dir "$CACHE_IN" --output "$out/zeroshot.json" $perm_flag
    fi
  else
    # COCO retrieval
    if [ ! -f "$out/retrieval.json" ]; then
      echo "[eval retrieval] $method / $dataset"
      local perm_flag=""
      if [ "$method" = "ours" ]; then perm_flag="--perm $OUT/ours/$dataset/perm.npz"; fi
      python scripts/real_alpha/eval_coco_retrieval.py \
          --ckpt "$ckpt" --method "$eval_method" \
          --cache-dir "$CACHE_COCO" --output "$out/retrieval.json" $perm_flag
    fi
  fi
}

for dataset in imagenet coco; do
  for method in shared separated iso_align group_sparse ours; do
    eval_one "$method" "$dataset"
  done
done

echo "[$(date '+%F %T')] finished real_exp_v1 matrix"
