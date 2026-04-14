#!/usr/bin/env bash
# theorem2_followup_15: IA/GS/ours β/λ sweep at 4× intervals (TA dropped — inert).
# α=0.5 only, σ=0.05.

set -uo pipefail
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log outputs/theorem2_followup_15

LOG=.log/followup15_sweep.log
echo "===== followup15 start $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"

BASIC="\
--lr 5e-4 --num-epochs 10 --num-train 50000 --num-eval 10000 --batch-size 256 \
--n-shared 1024 --n-image 512 --n-text 512 --representation-dim 256 \
--num-seeds 1 --seed-base 1 \
--shared-coeff-mode independent --coeff-dist exponential --cmin 0.0 --beta 1.0 \
--max-interference 0.1 --obs-noise-std 0.05 \
--k 16 --latent-size-sweep 8192 \
--device cuda --output-root outputs/theorem2_followup_15 \
--save-decoders \
--alpha-sweep 0.5"

run () {
  local TAG=$1; shift
  echo "----- ${TAG} start $(date -u +%H:%M:%SZ) -----" >> "$LOG"
  python synthetic_theorem2_method.py $BASIC "$@" --run-tag "$TAG" >> "$LOG" 2>&1
  echo "----- ${TAG} DONE $(date -u +%H:%M:%SZ) -----" >> "$LOG"
}

# 1R + 2R baselines
run "fu15_1R_2R_baseline" \
    --methods "single_recon,two_recon" \
    --group-sparse-lambda 0.05 --trace-beta 1e-4 --iso-align-beta 0.03

# IA β sweep — 4× intervals around 0.03
for B in 0.0019 0.0075 0.03 0.12 0.48; do
  run "fu15_IA_beta${B}" \
      --methods "iso_align" \
      --group-sparse-lambda 0.05 --trace-beta 1e-4 --iso-align-beta $B
done

# GS λ sweep — 4× intervals around 0.05
for L in 0.003125 0.0125 0.05 0.2 0.8; do
  run "fu15_GS_lam${L}" \
      --methods "group_sparse" \
      --group-sparse-lambda $L --trace-beta 1e-4 --iso-align-beta 0.03
done

# ours λ_aux sweep — 4× intervals around 2.0
for L in 0.125 0.5 2.0 8.0 32.0; do
  run "fu15_ours_lam${L}" \
      --methods "ours" \
      --group-sparse-lambda 0.05 --trace-beta 1e-4 --iso-align-beta 0.03 \
      --lambda-aux-sweep "$L" --m-s-sweep "1024" --k-align-sweep "6" \
      --aux-norm-sweep "global"
done

echo "===== followup15 DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"
