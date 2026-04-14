#!/usr/bin/env bash
# theorem2_followup_11: GS lambda sweep at L=8192.
# Three smaller lambdas: 0.001, 0.005, 0.01 (vs default 0.05).
# 1R baseline included for reference. α sweep 0.2..0.8.
#
# Setup matches followup8/9 basic: d=256, n_S=1024, n_I=n_T=512,
# Exp(1)+N(0,0.1^2), k=16, LR=5e-4, 10 epochs, 1 seed.

set -uo pipefail
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log outputs/theorem2_followup_11

LOG=.log/followup11_sweep.log
echo "===== followup11 start $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"

BASIC="\
--lr 5e-4 --num-epochs 10 --num-train 50000 --num-eval 10000 --batch-size 256 \
--n-shared 1024 --n-image 512 --n-text 512 --representation-dim 256 \
--num-seeds 1 --seed-base 1 \
--shared-coeff-mode independent --coeff-dist exponential --cmin 0.0 --beta 1.0 \
--max-interference 0.1 --obs-noise-std 0.1 \
--k 16 --latent-size-sweep 8192 \
--device cuda --output-root outputs/theorem2_followup_11 \
--save-decoders \
--alpha-sweep 0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
--trace-beta 1e-4 --iso-align-beta 0.03"

run () {
  local TAG=$1; shift
  echo "----- ${TAG} start $(date -u +%H:%M:%SZ) -----" >> "$LOG"
  python synthetic_theorem2_method.py $BASIC "$@" --run-tag "$TAG" >> "$LOG" 2>&1
  echo "----- ${TAG} DONE $(date -u +%H:%M:%SZ) -----" >> "$LOG"
}

# Sweep three smaller GS lambdas
for LAM in 0.001 0.005 0.01; do
  run "fu11_GS_lam${LAM}" \
      --methods "group_sparse" \
      --group-sparse-lambda $LAM
done

# Also include 1R baseline (one run, no GS) for direct comparison
run "fu11_1R_baseline" \
    --methods "single_recon" \
    --group-sparse-lambda 0.05  # unused for single_recon but parser needs it

echo "===== followup11 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"
