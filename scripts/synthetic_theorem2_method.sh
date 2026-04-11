#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

: "${DEVICE:=auto}"
: "${OUTPUT_ROOT:=outputs/synthetic_theorem2}"
: "${STAGE:=all}"   # smoke|main|ablation|all

ALL_METHODS="single_recon,two_recon,group_sparse,trace_align,iso_align,ours"

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
  # v7 main comparison:
  #  - α ∈ {0.3, 0.5, 0.7, 0.9} (all mismatched; drop the α=1.0 identity case)
  #  - ours: GLOBAL normalization only (decided after L=8192 v6 Pareto)
  #  - lambda = 1.0, m_S = 512, k_align = 6
  #  - baselines: group_sparse λ=0.05 (paper), trace_align β=1e-4 (paper
  #    Eq. 1), iso_align β=0.03 (IsoEnergy repo code default).
  python synthetic_theorem2_method.py \
    --alpha-sweep "0.3,0.5,0.7,0.9" \
    --methods "$ALL_METHODS" \
    --lambda-aux-sweep "1.0" --m-s-sweep "512" --k-align-sweep "6" \
    --aux-norm-sweep "global" \
    --group-sparse-lambda 0.05 --trace-beta 1e-4 --iso-align-beta 0.03 \
    --run-tag "main_comparison" \
    $COMMON_FULL "$@"
fi

if [[ "$STAGE" == "ablation" || "$STAGE" == "all" ]]; then
  # v7 ablation 1: lambda sweep on ours (global only)
  # 7-point FACTOR-OF-4 grid centered on λ=1: {2^-6, 2^-4, ..., 2^6}.
  python synthetic_theorem2_method.py \
    --alpha-sweep "0.9" --methods "ours" \
    --lambda-aux-sweep "0.015625,0.0625,0.25,1,4,16,64" \
    --m-s-sweep "512" --k-align-sweep "6" \
    --aux-norm-sweep "global" \
    --run-tag "ablation_lambda" \
    $COMMON_FULL "$@"

  # v7 ablation 2: m_S fine scan around n_shared (λ=1, k_align=6)
  python synthetic_theorem2_method.py \
    --alpha-sweep "0.9" --methods "ours" \
    --lambda-aux-sweep "1" \
    --m-s-sweep "384,448,512,576,640" \
    --k-align-sweep "6" \
    --aux-norm-sweep "global" \
    --run-tag "ablation_mS" \
    $COMMON_FULL "$@"

  # v7 ablation 3: k_align sweep with k=10 diagnostic endpoint
  python synthetic_theorem2_method.py \
    --alpha-sweep "0.9" --methods "ours" \
    --lambda-aux-sweep "1" --m-s-sweep "512" \
    --k-align-sweep "2,4,6,8,10" \
    --aux-norm-sweep "global" \
    --run-tag "ablation_kalign" \
    $COMMON_FULL "$@"
fi
