#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${DEVICE:=auto}"
: "${OUTPUT_ROOT:=outputs/synthetic_theorem2}"
: "${STAGE:=all}"   # smoke|main|ablation|all

ALL_METHODS="single_recon,two_recon,group_sparse,trace_align,ours"

COMMON_FULL="--k 16 --num-epochs 10 --num-train 50000 --num-eval 10000 \
  --lr 2e-4 --batch-size 256 \
  --n-shared 512 --n-image 256 --n-text 256 --representation-dim 768 \
  --latent-size-sweep 2048 --num-seeds 3 --seed-base 1 \
  --device $DEVICE --output-root $OUTPUT_ROOT"

if [[ "$STAGE" == "smoke" || "$STAGE" == "all" ]]; then
  python synthetic_theorem2_method.py --self-test

  python synthetic_theorem2_method.py \
    --alpha-sweep "1.0" \
    --latent-size-sweep "1024" \
    --methods "$ALL_METHODS" \
    --num-seeds 1 --num-epochs 2 --num-train 2000 --num-eval 500 \
    --n-shared 64 --n-image 32 --n-text 32 --representation-dim 128 \
    --lambda-aux-sweep "1.0" --m-s-sweep "64" --k-align-sweep "1" \
    --run-tag "smoke" --device cpu \
    --output-root "$OUTPUT_ROOT" "$@"
fi

if [[ "$STAGE" == "main" || "$STAGE" == "all" ]]; then
  python synthetic_theorem2_method.py \
    --alpha-sweep "0.3,0.6,0.9,1.0" \
    --methods "$ALL_METHODS" \
    --lambda-aux-sweep "1.0" --m-s-sweep "512" --k-align-sweep "6" \
    --group-sparse-lambda 0.05 --trace-beta 1e-4 \
    --run-tag "main_comparison" \
    $COMMON_FULL "$@"
fi

if [[ "$STAGE" == "ablation" || "$STAGE" == "all" ]]; then
  # Ablation 1: lambda sweep (m_S = 512, k_align = 6)
  python synthetic_theorem2_method.py \
    --alpha-sweep "1.0" --methods "ours" \
    --lambda-aux-sweep "0.0625,0.25,1,4,16,64,256" \
    --m-s-sweep "512" --k-align-sweep "6" \
    --run-tag "ablation_lambda" \
    $COMMON_FULL "$@"

  # Ablation 2: m_S fine scan around n_shared (lambda = 1, k_align = 6)
  python synthetic_theorem2_method.py \
    --alpha-sweep "1.0" --methods "ours" \
    --lambda-aux-sweep "1" \
    --m-s-sweep "384,448,512,576,640" \
    --k-align-sweep "6" \
    --run-tag "ablation_mS" \
    $COMMON_FULL "$@"

  # Ablation 3: k_align sweep (lambda = 1, m_S = 512)
  python synthetic_theorem2_method.py \
    --alpha-sweep "1.0" --methods "ours" \
    --lambda-aux-sweep "1" --m-s-sweep "512" \
    --k-align-sweep "2,4,6,8" \
    --run-tag "ablation_kalign" \
    $COMMON_FULL "$@"
fi
