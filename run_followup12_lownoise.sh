#!/usr/bin/env bash
# theorem2_followup_12: Lower-noise (σ=0.05) main sweep at L=8192.
# 6 methods × α 0.2..0.8. GS uses paper default λ=0.05.
# Saves all SAE parameters for offline metric reuse.
#
# Setup follows followup8/9 basic but with noise reduced from 0.1 to 0.05
# to operate in convergent regime (SNR ≈ 3 instead of 0.78).

set -uo pipefail
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log outputs/theorem2_followup_12

LOG=.log/followup12_sweep.log
echo "===== followup12 start $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"

ALL_METHODS="single_recon,two_recon,group_sparse,trace_align,iso_align,ours"

run () {
  local TAG=$1; shift
  echo "----- ${TAG} start $(date -u +%H:%M:%SZ) -----" >> "$LOG"
  python synthetic_theorem2_method.py \
    --alpha-sweep "0.2,0.3,0.4,0.5,0.6,0.7,0.8" \
    --latent-size-sweep "8192" \
    --methods "$ALL_METHODS" \
    --lambda-aux-sweep "2.0" --m-s-sweep "1024" --k-align-sweep "6" \
    --aux-norm-sweep "global" \
    --group-sparse-lambda 0.05 --trace-beta 1e-4 --iso-align-beta 0.03 \
    --run-tag "$TAG" \
    --k 16 --num-epochs 10 --num-train 50000 --num-eval 10000 \
    --lr 5e-4 --batch-size 256 \
    --n-shared 1024 --n-image 512 --n-text 512 --representation-dim 256 \
    --num-seeds 1 --seed-base 1 \
    --shared-coeff-mode independent --coeff-dist exponential \
    --cmin 0.0 --beta 1.0 \
    --max-interference 0.1 \
    --obs-noise-std 0.05 \
    --save-decoders \
    --device cuda --output-root outputs/theorem2_followup_12 >> "$LOG" 2>&1
  echo "----- ${TAG} DONE $(date -u +%H:%M:%SZ) -----" >> "$LOG"
}

run "fu12_lownoise_6methods_alpha_sweep"

echo "===== followup12 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"
