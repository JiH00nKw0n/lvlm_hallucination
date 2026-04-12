#!/usr/bin/env bash
# Basic experiment chain for theorem2_followup.
# Usage: bash run_basic_chain.sh [K] [SHARED_COEFF_MODE] [NUM_SEEDS]
# Example: bash run_basic_chain.sh 16 independent 1
#          bash run_basic_chain.sh 8 independent 3
set -uo pipefail
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate

K=${1:-16}
COEFF_MODE=${2:-independent}
NUM_SEEDS=${3:-3}
TAG="k${K}_${COEFF_MODE}"
LOG=.log/theorem2_followup_${TAG}.log
mkdir -p .log

echo "===== basic chain start $(date -u +%Y-%m-%dT%H:%M:%SZ) K=$K COEFF=$COEFF_MODE SEEDS=$NUM_SEEDS =====" >> "$LOG"

ALL_METHODS="single_recon,two_recon,group_sparse,trace_align,iso_align,ours"

COMMON="--k $K --num-epochs 10 --num-train 50000 --num-eval 10000 \
  --lr 2e-4 --batch-size 256 \
  --n-shared 512 --n-image 256 --n-text 256 --representation-dim 768 \
  --num-seeds $NUM_SEEDS --seed-base 1 \
  --shared-coeff-mode $COEFF_MODE \
  --device cuda --output-root outputs/theorem2_followup"

# Main comparison: L=4096, α∈{0.5,0.7,0.9}, λ_aux=2.0
echo "----- main start $(date -u +%H:%M:%SZ) -----" >> "$LOG"
python synthetic_theorem2_method.py \
  --alpha-sweep "0.5,0.7,0.9" \
  --latent-size-sweep "4096" \
  --methods "$ALL_METHODS" \
  --lambda-aux-sweep "2.0" --m-s-sweep "512" --k-align-sweep "6" \
  --aux-norm-sweep "global" \
  --group-sparse-lambda 0.05 --trace-beta 1e-4 --iso-align-beta 0.03 \
  --run-tag "main_${TAG}" \
  $COMMON "$@" >> "$LOG" 2>&1
echo "----- main DONE $(date -u +%H:%M:%SZ) -----" >> "$LOG"

# Ablation 1: lambda sweep (α=0.9 only)
echo "----- ablation_lambda start $(date -u +%H:%M:%SZ) -----" >> "$LOG"
python synthetic_theorem2_method.py \
  --alpha-sweep "0.9" \
  --latent-size-sweep "4096" \
  --methods "ours" \
  --lambda-aux-sweep "0.015625,0.0625,0.25,1,2,4,16,64" \
  --m-s-sweep "512" --k-align-sweep "6" \
  --aux-norm-sweep "global" \
  --run-tag "ablation_lambda_${TAG}" \
  $COMMON "$@" >> "$LOG" 2>&1
echo "----- ablation_lambda DONE $(date -u +%H:%M:%SZ) -----" >> "$LOG"

# Ablation 2: m_S sweep (α=0.9 only)
echo "----- ablation_mS start $(date -u +%H:%M:%SZ) -----" >> "$LOG"
python synthetic_theorem2_method.py \
  --alpha-sweep "0.9" \
  --latent-size-sweep "4096" \
  --methods "ours" \
  --lambda-aux-sweep "2.0" --m-s-sweep "384,448,512,576,640" --k-align-sweep "6" \
  --aux-norm-sweep "global" \
  --run-tag "ablation_mS_${TAG}" \
  $COMMON "$@" >> "$LOG" 2>&1
echo "----- ablation_mS DONE $(date -u +%H:%M:%SZ) -----" >> "$LOG"

# Ablation 3: k_align sweep (α=0.9 only)
echo "----- ablation_kalign start $(date -u +%H:%M:%SZ) -----" >> "$LOG"
python synthetic_theorem2_method.py \
  --alpha-sweep "0.9" \
  --latent-size-sweep "4096" \
  --methods "ours" \
  --lambda-aux-sweep "2.0" --m-s-sweep "512" --k-align-sweep "2,4,6,8,10" \
  --aux-norm-sweep "global" \
  --run-tag "ablation_kalign_${TAG}" \
  $COMMON "$@" >> "$LOG" 2>&1
echo "----- ablation_kalign DONE $(date -u +%H:%M:%SZ) -----" >> "$LOG"

echo "===== basic chain DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"
