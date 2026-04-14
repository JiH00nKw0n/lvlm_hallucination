#!/usr/bin/env bash
# theorem2_followup_16: wider α sweep (0.1..1.0) for Figure 1.
# Matches followup12 (lownoise σ=0.05) setup; only 1R/2R methods.

set -uo pipefail
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log outputs/theorem2_followup_16

LOG=.log/followup16_sweep.log
echo "===== followup16 start $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"

python synthetic_theorem2_method.py \
    --alpha-sweep "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
    --latent-size-sweep "8192" \
    --methods "single_recon,two_recon" \
    --group-sparse-lambda 0.05 --trace-beta 1e-4 --iso-align-beta 0.03 \
    --run-tag "fu16_wide_alpha_1R_2R" \
    --k 16 --num-epochs 10 --num-train 50000 --num-eval 10000 \
    --lr 5e-4 --batch-size 256 \
    --n-shared 1024 --n-image 512 --n-text 512 --representation-dim 256 \
    --num-seeds 1 --seed-base 1 \
    --shared-coeff-mode independent --coeff-dist exponential \
    --cmin 0.0 --beta 1.0 \
    --max-interference 0.1 \
    --obs-noise-std 0.05 \
    --save-decoders \
    --device cuda --output-root outputs/theorem2_followup_16 >> "$LOG" 2>&1

echo "===== followup16 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"
