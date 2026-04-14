#!/usr/bin/env bash
# theorem2_followup_10: 1R vs 2R at L=4096 with α sweep 0.1..0.9.
# Saves all trained decoder matrices so we can compute additional metrics
# offline without re-training.
#
# Basic setup (followup8/9): d=256, n_S=1024, n_I=n_T=512, Exp(1)+N(0,0.1^2),
# k=16, LR=5e-4, 10 epochs, 1 seed.

set -uo pipefail
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log outputs/theorem2_followup_10

LOG=.log/followup10_sweep.log
echo "===== followup10 start $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"

BASIC="\
--lr 5e-4 --num-epochs 10 --num-train 50000 --num-eval 10000 --batch-size 256 \
--n-shared 1024 --n-image 512 --n-text 512 --representation-dim 256 \
--num-seeds 1 --seed-base 1 \
--shared-coeff-mode independent --coeff-dist exponential --cmin 0.0 --beta 1.0 \
--max-interference 0.1 --obs-noise-std 0.1 \
--k 16 --latent-size-sweep 4096 \
--device cuda --output-root outputs/theorem2_followup_10 \
--save-decoders"

# Baseline lambdas not used (no GS/TA/IA/ours here), but the parser still wants them.
BASELINE_LAMS="--group-sparse-lambda 0.05 --trace-beta 1e-4 --iso-align-beta 0.03"

run () {
  local TAG=$1; shift
  echo "----- ${TAG} start $(date -u +%H:%M:%SZ) -----" >> "$LOG"
  python synthetic_theorem2_method.py $BASIC $BASELINE_LAMS "$@" --run-tag "$TAG" >> "$LOG" 2>&1
  echo "----- ${TAG} DONE $(date -u +%H:%M:%SZ) -----" >> "$LOG"
}

run "fu10_L4096_1R_2R_alpha_sweep" \
    --alpha-sweep "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9" \
    --methods "single_recon,two_recon"

echo "===== followup10 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"
