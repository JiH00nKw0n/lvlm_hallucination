#!/usr/bin/env bash
# theorem2_followup_13: β/λ sweep for prior alignment losses + ours.
# Each method has its paper default in the middle, with 2 smaller and 2
# larger values around it. All other settings match followup12 (lownoise).
#
# Setup: d=256, n_S=1024, n_I=n_T=512, σ=0.05, k=16, LR=5e-4, 10 epochs.
#
# Methods and ranges:
#   TA   (paper default beta = 1e-4): {1e-6, 1e-5, 1e-4, 1e-3, 1e-2}
#   IA   (paper default beta = 0.03): {0.003, 0.01, 0.03, 0.1, 0.3}
#   ours (paper default λ_aux = 2.0): {0.5, 1.0, 2.0, 4.0, 8.0}

set -uo pipefail
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log outputs/theorem2_followup_13

LOG=.log/followup13_sweep.log
echo "===== followup13 start $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"

BASIC="\
--lr 5e-4 --num-epochs 10 --num-train 50000 --num-eval 10000 --batch-size 256 \
--n-shared 1024 --n-image 512 --n-text 512 --representation-dim 256 \
--num-seeds 1 --seed-base 1 \
--shared-coeff-mode independent --coeff-dist exponential --cmin 0.0 --beta 1.0 \
--max-interference 0.1 --obs-noise-std 0.05 \
--k 16 --latent-size-sweep 8192 \
--device cuda --output-root outputs/theorem2_followup_13 \
--save-decoders \
--alpha-sweep 0.2,0.3,0.4,0.5,0.6,0.7,0.8 \
--group-sparse-lambda 0.05"

run () {
  local TAG=$1; shift
  echo "----- ${TAG} start $(date -u +%H:%M:%SZ) -----" >> "$LOG"
  python synthetic_theorem2_method.py $BASIC "$@" --run-tag "$TAG" >> "$LOG" 2>&1
  echo "----- ${TAG} DONE $(date -u +%H:%M:%SZ) -----" >> "$LOG"
}

# 1R baseline (reference)
run "fu13_1R_baseline" \
    --methods "single_recon" \
    --trace-beta 1e-4 --iso-align-beta 0.03

# TA β sweep — 5 values centered at paper default (1e-4)
for B in 1e-6 1e-5 1e-4 1e-3 1e-2; do
  run "fu13_TA_beta${B}" \
      --methods "trace_align" \
      --trace-beta $B --iso-align-beta 0.03
done

# IA β sweep — 5 values centered at paper default (0.03)
for B in 0.003 0.01 0.03 0.1 0.3; do
  run "fu13_IA_beta${B}" \
      --methods "iso_align" \
      --trace-beta 1e-4 --iso-align-beta $B
done

# ours λ_aux sweep — 5 values centered at paper default (2.0)
for L in 0.5 1.0 2.0 4.0 8.0; do
  run "fu13_ours_lam${L}" \
      --methods "ours" \
      --trace-beta 1e-4 --iso-align-beta 0.03 \
      --lambda-aux-sweep "$L" --m-s-sweep "1024" --k-align-sweep "6" \
      --aux-norm-sweep "global"
done

echo "===== followup13 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"
