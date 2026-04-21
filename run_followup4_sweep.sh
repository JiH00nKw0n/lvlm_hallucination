#!/usr/bin/env bash
# theorem2_followup_4: n_S=1024 (overcomplete shared dict at m=768) + exp(1) + noise=0.1
# Goal: harder shared-identifiability regime to differentiate methods on recon AND mgt/mcc
# Total: 2 runs (k ∈ {16, 8})
set -uo pipefail
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log outputs/theorem2_followup_4

LOG=.log/followup4_sweep.log
echo "===== followup4 start $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"

ALL_METHODS="single_recon,two_recon,group_sparse,trace_align,iso_align,ours"

run_one () {
  local K=$1
  local RUN_TAG="followup4_k${K}_ns1024_exp1_noise0p1"
  echo "----- ${RUN_TAG} start $(date -u +%H:%M:%SZ) [k=${K}] -----" >> "$LOG"
  python synthetic_theorem2_method.py \
    --alpha-sweep "0.5,0.7,0.9" \
    --latent-size-sweep "4096" \
    --methods "$ALL_METHODS" \
    --lambda-aux-sweep "2.0" --m-s-sweep "1024" --k-align-sweep "6" \
    --aux-norm-sweep "global" \
    --group-sparse-lambda 0.05 --trace-beta 1e-4 --iso-align-beta 0.03 \
    --run-tag "$RUN_TAG" \
    --k "$K" --num-epochs 10 --num-train 50000 --num-eval 10000 \
    --lr 5e-4 --batch-size 256 \
    --n-shared 1024 --n-image 256 --n-text 256 --representation-dim 768 \
    --num-seeds 1 --seed-base 1 \
    --shared-coeff-mode independent \
    --coeff-dist exponential \
    --cmin 0.0 --beta 1.0 \
    --max-interference 0.1 \
    --obs-noise-std 0.1 \
    --device cuda --output-root outputs/theorem2_followup_4 >> "$LOG" 2>&1
  echo "----- ${RUN_TAG} DONE $(date -u +%H:%M:%SZ) -----" >> "$LOG"
}

for K in 16 8; do
  run_one $K
done

echo "===== followup4 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"
