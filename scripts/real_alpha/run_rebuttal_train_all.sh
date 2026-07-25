#!/usr/bin/env bash
# Train the six SAEs the rebuttal experiments need.
#
# Two operating points, three independent runs each. One run trains an image SAE
# and a text SAE together (TwoSidedTopKSAE with separate decoders), so six runs
# cover every comparison we need.
#
#   Setting F — the paper's Figure 2 point:  COCO,  L=8192 (4096/side), K=8,  30 epochs
#   Setting T — the paper's Table 1 point:   CC3M,  L=8192 (4096/side), k=32, 10 epochs
#
# Hyperparameters are copied verbatim from the runs that produced those results
# (outputs/real_alpha_followup_1/two_sae/config.json and
#  outputs/real_exp_cc3m_s0/separated/ckpt/config.json). Only the seed differs
# between runs, and train_rebuttal_sae.py seeds before the model is built so the
# initialization is genuinely seed-controlled.
#
#   bash scripts/real_alpha/run_rebuttal_train_all.sh            # all six
#   ONLY=F bash scripts/real_alpha/run_rebuttal_train_all.sh     # setting F only
#   SEEDS="1 2" bash scripts/real_alpha/run_rebuttal_train_all.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f env.sh ] && source env.sh || true
[ -f /mnt/elice/working/env.sh ] && source /mnt/elice/working/env.sh || true

PY=python; [ -x .venv/bin/python ] && PY=.venv/bin/python
SEEDS="${SEEDS:-1 2 3}"
ONLY="${ONLY:-FT}"
ROOT=outputs/rebuttal_models
export SAE_SAVE_TOTAL_LIMIT="${SAE_SAVE_TOTAL_LIMIT:-1}"   # disk is tight; keep one ckpt
mkdir -p "$ROOT" .log

TRAIN="$PY scripts/real_alpha/train_rebuttal_sae.py --variant two_sae \
  --hidden-size 512 --batch-size 1024 --lr 5e-4 --warmup-ratio 0.05 \
  --weight-decay 1e-5 --max-grad-norm 1.0 --latent 8192"

started=$(date +%s)
run_one() {   # $1 = tag, $2... = extra args
  local tag="$1"; shift
  local out="$ROOT/$tag"
  if [ -f "$out/final/model.safetensors" ]; then
    echo "[skip] $tag already trained"
    return
  fi
  echo
  echo "======== $tag  ($(date -u +%H:%M:%SZ)) ========"
  local t0=$(date +%s)
  # shellcheck disable=SC2086
  $TRAIN --output-dir "$out" "$@" 2>&1 | tee ".log/rebuttal_train_$tag.log"
  echo "[done] $tag in $((($(date +%s) - t0) / 60)) min"
}

for s in $SEEDS; do
  case "$ONLY" in *F*)
    run_one "coco_k8_r$s" --dataset coco --cache-dir cache/clip_b32_coco \
      --k 8 --epochs 30 --dataloader-num-workers 2 --seed "$s" ;;
  esac
done

for s in $SEEDS; do
  case "$ONLY" in *T*)
    run_one "cc3m_k32_r$s" --dataset cc3m --cache-dir cache/clip_b32_cc3m \
      --eval-samples 20000 --k 32 --epochs 10 --dataloader-num-workers 0 --seed "$s" ;;
  esac
done

echo
echo "======== all runs finished in $((($(date +%s) - started) / 60)) min ========"
echo "--- initial weight hashes (must all differ) ---"
for d in "$ROOT"/*/; do
  [ -f "$d/init_signature.json" ] && echo "$(basename "$d")  $(cat "$d/init_signature.json" | tr -d ' \n')"
done
echo "REBUTTAL_TRAIN_DONE"
