#!/usr/bin/env bash
# theorem2_followup_8: same config as followup7 (m=256, n_S=1024, n_I=n_T=512, L=8192,
# noise=0.1, k ∈ {16, 32}) but with new post-hoc XMA metric added (probe_*_posthoc keys).
set -uo pipefail
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log outputs/theorem2_followup_8

LOG=.log/followup8_sweep.log
echo "===== followup8 start $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"

ALL_METHODS="single_recon,two_recon,group_sparse,trace_align,iso_align,ours"

run_one () {
  local K=$1
  local RUN_TAG="followup8_k${K}_m256_ns1024_pi512_L8192_noise0p1"
  echo "----- ${RUN_TAG} start $(date -u +%H:%M:%SZ) [k=${K}] -----" >> "$LOG"
  python synthetic_theorem2_method.py \
    --alpha-sweep "0.5,0.7,0.9" \
    --latent-size-sweep "8192" \
    --methods "$ALL_METHODS" \
    --lambda-aux-sweep "2.0" --m-s-sweep "1024" --k-align-sweep "6" \
    --aux-norm-sweep "global" \
    --group-sparse-lambda 0.05 --trace-beta 1e-4 --iso-align-beta 0.03 \
    --run-tag "$RUN_TAG" \
    --k "$K" --num-epochs 10 --num-train 50000 --num-eval 10000 \
    --lr 5e-4 --batch-size 256 \
    --n-shared 1024 --n-image 512 --n-text 512 --representation-dim 256 \
    --num-seeds 1 --seed-base 1 \
    --shared-coeff-mode independent \
    --coeff-dist exponential \
    --cmin 0.0 --beta 1.0 \
    --max-interference 0.1 \
    --obs-noise-std 0.1 \
    --device cuda --output-root outputs/theorem2_followup_8 >> "$LOG" 2>&1
  echo "----- ${RUN_TAG} DONE $(date -u +%H:%M:%SZ) -----" >> "$LOG"
}

for K in 16 32; do
  run_one $K
done

echo "===== followup8 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"
