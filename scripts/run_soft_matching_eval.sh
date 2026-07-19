#!/usr/bin/env bash
# Rebuttal E6: hard-vs-soft alignment comparison on existing CC3M 3-seed ckpts.
#
# For each seed: build {hungarian, greedy, dec_cos, sinkhorn×eps} assignments
# from the training cache, then run COCO retrieval + ImageNet zero-shot with
# each. Results land under outputs/rebuttal_E6/s<seed>/<variant>/.
#
# Usage:  bash scripts/run_soft_matching_eval.sh [seeds...]   (default: 0 1 2)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

SEEDS=("${@:-0 1 2}")
if [[ $# -eq 0 ]]; then SEEDS=(0 1 2); fi
CC3M_CACHE=${CC3M_CACHE:-cache/clip_b32_cc3m}
COCO_CACHE=${COCO_CACHE:-cache/clip_b32_coco}
IMAGENET_CACHE=${IMAGENET_CACHE:-cache/clip_b32_imagenet}
EPS_LIST=${EPS_LIST:-"0.01 0.05 0.1"}

for SEED in "${SEEDS[@]}"; do
  CKPT="outputs/real_exp_cc3m_s${SEED}/separated/ckpt/final"
  OUT="outputs/rebuttal_E6/s${SEED}"
  echo "=== seed ${SEED}: building assignments → ${OUT}"
  python scripts/real_alpha/build_soft_assignment.py \
    --ckpt "$CKPT" --dataset cc3m --cache-dir "$CC3M_CACHE" \
    --eps $EPS_LIST --output-dir "$OUT"

  run_variant () {  # $1 = variant name, $2 = "--perm file" or "--soft-map file"
    local NAME=$1; shift
    echo "--- seed ${SEED} variant ${NAME}"
    mkdir -p "$OUT/$NAME"
    python scripts/real_alpha/eval_coco_retrieval.py \
      --ckpt "$CKPT" --method ours --cache-dir "$COCO_CACHE" --split test \
      "$@" --output "$OUT/$NAME/retrieval.json"
    if [[ -d "$IMAGENET_CACHE" ]]; then
      python scripts/real_alpha/eval_imagenet_zeroshot.py \
        --ckpt "$CKPT" --method ours --cache-dir "$IMAGENET_CACHE" \
        "$@" --output "$OUT/$NAME/zeroshot.json"
    else
      echo "    (skip zeroshot: $IMAGENET_CACHE missing)"
    fi
  }

  run_variant hungarian --perm "$OUT/perm_hungarian.npz"
  run_variant greedy    --perm "$OUT/perm_greedy.npz"
  run_variant dec_cos   --perm "$OUT/perm_dec_cos.npz"
  for EPS in $EPS_LIST; do
    run_variant "sinkhorn_eps${EPS}" --soft-map "$OUT/soft_T_eps${EPS}.npz"
  done
done
echo "E6 done."
