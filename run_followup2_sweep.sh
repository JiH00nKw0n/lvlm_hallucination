#!/usr/bin/env bash
# theorem2_followup_2: 3-variable × 2-point sweep for relu_gaussian harder settings
# Variables: coefficient (μ, σ), max_interference, obs_noise_std
# Fixed: LR=5e-4, ep=10, k_align=6, relu_gaussian, independent coeff
set -uo pipefail
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log outputs/theorem2_followup_2

LOG=.log/followup2_sweep.log
echo "===== followup2 start $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"

ALL_METHODS="single_recon,two_recon,group_sparse,trace_align,iso_align,ours"

run_one () {
  local K=$1 MU=$2 SIG=$3 INT=$4 NOISE=$5 TAG=$6
  local RUN_TAG="followup2_k${K}_${TAG}"
  echo "----- ${RUN_TAG} start $(date -u +%H:%M:%SZ) [k=${K} mu=${MU} sig=${SIG} int=${INT} noise=${NOISE}] -----" >> "$LOG"
  python synthetic_theorem2_method.py \
    --alpha-sweep "0.5,0.7,0.9" \
    --latent-size-sweep "4096" \
    --methods "$ALL_METHODS" \
    --lambda-aux-sweep "2.0" --m-s-sweep "512" --k-align-sweep "6" \
    --aux-norm-sweep "global" \
    --group-sparse-lambda 0.05 --trace-beta 1e-4 --iso-align-beta 0.03 \
    --run-tag "$RUN_TAG" \
    --k "$K" --num-epochs 10 --num-train 50000 --num-eval 10000 \
    --lr 5e-4 --batch-size 256 \
    --n-shared 512 --n-image 256 --n-text 256 --representation-dim 768 \
    --num-seeds 1 --seed-base 1 \
    --shared-coeff-mode independent \
    --coeff-dist relu_gaussian \
    --coeff-mu "$MU" --coeff-sigma "$SIG" \
    --max-interference "$INT" \
    --obs-noise-std "$NOISE" \
    --device cuda --output-root outputs/theorem2_followup_2 >> "$LOG" 2>&1
  echo "----- ${RUN_TAG} DONE $(date -u +%H:%M:%SZ) -----" >> "$LOG"
}

# 8 combinations × 2 k values = 16 runs
# Combinations (coeff, interference, noise):
#   B   = baseline   (4.5, 0.5, 0.1, 0.0)
#   C   = coeff only (2.0, 1.0, 0.1, 0.0)
#   I   = interference only (4.5, 0.5, 0.2, 0.0)
#   N   = noise only (4.5, 0.5, 0.1, 0.5)
#   CI  = coeff + interference (2.0, 1.0, 0.2, 0.0)
#   CN  = coeff + noise (2.0, 1.0, 0.1, 0.5)
#   IN  = interference + noise (4.5, 0.5, 0.2, 0.5)
#   CIN = all three harder (2.0, 1.0, 0.2, 0.5)
for K in 16 8; do
  run_one $K 4.5 0.5 0.1 0.0 "B"
  run_one $K 2.0 1.0 0.1 0.0 "C"
  run_one $K 4.5 0.5 0.2 0.0 "I"
  run_one $K 4.5 0.5 0.1 0.5 "N"
  run_one $K 2.0 1.0 0.2 0.0 "CI"
  run_one $K 2.0 1.0 0.1 0.5 "CN"
  run_one $K 4.5 0.5 0.2 0.5 "IN"
  run_one $K 2.0 1.0 0.2 0.5 "CIN"
done

echo "===== followup2 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"
