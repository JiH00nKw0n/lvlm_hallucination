#!/usr/bin/env bash
# Basic experiment chain: main comparison only (no ablation).
# Usage: bash run_basic_chain.sh [K] [SHARED_COEFF_MODE] [NUM_SEEDS] [COEFF_DIST]
# Example: bash run_basic_chain.sh 16 independent 1
#          bash run_basic_chain.sh 8 independent 1 relu_gaussian
set -uo pipefail
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate

K=${1:-16}
COEFF_MODE=${2:-independent}
NUM_SEEDS=${3:-3}
COEFF_DIST=${4:-exponential}
TAG="k${K}_${COEFF_MODE}_${COEFF_DIST}"
LOG=.log/theorem2_followup_${TAG}.log
mkdir -p .log

echo "===== basic chain start $(date -u +%Y-%m-%dT%H:%M:%SZ) K=$K COEFF=$COEFF_MODE SEEDS=$NUM_SEEDS DIST=$COEFF_DIST =====" >> "$LOG"

ALL_METHODS="single_recon,two_recon,group_sparse,trace_align,iso_align,ours"

COMMON="--k $K --num-epochs 10 --num-train 50000 --num-eval 10000 \
  --lr 2e-4 --batch-size 256 \
  --n-shared 512 --n-image 256 --n-text 256 --representation-dim 768 \
  --num-seeds $NUM_SEEDS --seed-base 1 \
  --shared-coeff-mode $COEFF_MODE \
  --coeff-dist $COEFF_DIST \
  --device cuda --output-root outputs/theorem2_followup"

# Main comparison only: L=4096, α∈{0.5,0.7,0.9}, λ_aux=2.0
echo "----- main start $(date -u +%H:%M:%SZ) -----" >> "$LOG"
python synthetic_theorem2_method.py \
  --alpha-sweep "0.5,0.7,0.9" \
  --latent-size-sweep "4096" \
  --methods "$ALL_METHODS" \
  --lambda-aux-sweep "2.0" --m-s-sweep "512" --k-align-sweep "6" \
  --aux-norm-sweep "global" \
  --group-sparse-lambda 0.05 --trace-beta 1e-4 --iso-align-beta 0.03 \
  --run-tag "main_${TAG}" \
  $COMMON >> "$LOG" 2>&1
echo "----- main DONE $(date -u +%H:%M:%SZ) -----" >> "$LOG"

echo "===== basic chain DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"
