#!/usr/bin/env bash
# Full sweep: LR × epoch × k_align for all 4 basic settings.
# Runs main comparison only (no ablation), 1 seed each.
set -uo pipefail
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log outputs/theorem2_followup

LOG=.log/sweep_all.log
echo "===== sweep start $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"

ALL_METHODS="single_recon,two_recon,group_sparse,trace_align,iso_align,ours"

run_one () {
  local K=$1 COEFF_MODE=$2 COEFF_DIST=$3 LR=$4 EPOCHS=$5 KALIGN=$6
  local TAG="k${K}_${COEFF_MODE}_${COEFF_DIST}_lr${LR}_ep${EPOCHS}_ka${KALIGN}"
  echo "----- ${TAG} start $(date -u +%H:%M:%SZ) -----" >> "$LOG"
  python synthetic_theorem2_method.py \
    --alpha-sweep "0.5,0.7,0.9" \
    --latent-size-sweep "4096" \
    --methods "$ALL_METHODS" \
    --lambda-aux-sweep "2.0" --m-s-sweep "512" --k-align-sweep "$KALIGN" \
    --aux-norm-sweep "global" \
    --group-sparse-lambda 0.05 --trace-beta 1e-4 --iso-align-beta 0.03 \
    --run-tag "sweep_${TAG}" \
    --k "$K" --num-epochs "$EPOCHS" --num-train 50000 --num-eval 10000 \
    --lr "$LR" --batch-size 256 \
    --n-shared 512 --n-image 256 --n-text 256 --representation-dim 768 \
    --num-seeds 1 --seed-base 1 \
    --shared-coeff-mode "$COEFF_MODE" \
    --coeff-dist "$COEFF_DIST" \
    --device cuda --output-root outputs/theorem2_followup >> "$LOG" 2>&1
  echo "----- ${TAG} DONE $(date -u +%H:%M:%SZ) -----" >> "$LOG"
}

# ============================================================
# Part 1: LR sweep (epochs=10, k_align=6)
# ============================================================
for K in 16 8; do
  for DIST in exponential relu_gaussian; do
    for LR in 5e-4 1e-3 5e-3; do
      run_one $K independent $DIST $LR 10 6
    done
  done
done

# ============================================================
# Part 2: 20-epoch sweep with k_align ∈ {4,8,12,16}
# ============================================================
for K in 16 8; do
  for DIST in exponential relu_gaussian; do
    for KA in 4 8 12 16; do
      run_one $K independent $DIST 2e-4 20 $KA
    done
  done
done

# ============================================================
# Part 3: 30-epoch sweep with k_align ∈ {4,8,12,16,20,24}
# ============================================================
for K in 16 8; do
  for DIST in exponential relu_gaussian; do
    for KA in 4 8 12 16 20 24; do
      run_one $K independent $DIST 2e-4 30 $KA
    done
  done
done

echo "===== sweep DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"
