#!/usr/bin/env bash
# Full sweep: LR × epoch × k_align for all 4 basic settings.
# - LR/epoch changes → all 6 methods
# - k_align changes → ours only (other methods unaffected)
set -uo pipefail
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log outputs/theorem2_followup

LOG=.log/sweep_all.log
echo "===== sweep start $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"

ALL_METHODS="single_recon,two_recon,group_sparse,trace_align,iso_align,ours"

run_main () {
  local K=$1 CM=$2 CD=$3 LR=$4 EP=$5 KA=$6 METHODS=$7
  local TAG="k${K}_${CM}_${CD}_lr${LR}_ep${EP}_ka${KA}"
  echo "----- ${TAG} (${METHODS}) start $(date -u +%H:%M:%SZ) -----" >> "$LOG"
  python synthetic_theorem2_method.py \
    --alpha-sweep "0.5,0.7,0.9" \
    --latent-size-sweep "4096" \
    --methods "$METHODS" \
    --lambda-aux-sweep "2.0" --m-s-sweep "512" --k-align-sweep "$KA" \
    --aux-norm-sweep "global" \
    --group-sparse-lambda 0.05 --trace-beta 1e-4 --iso-align-beta 0.03 \
    --run-tag "sweep_${TAG}" \
    --k "$K" --num-epochs "$EP" --num-train 50000 --num-eval 10000 \
    --lr "$LR" --batch-size 256 \
    --n-shared 512 --n-image 256 --n-text 256 --representation-dim 768 \
    --num-seeds 1 --seed-base 1 \
    --shared-coeff-mode "$CM" --coeff-dist "$CD" \
    --device cuda --output-root outputs/theorem2_followup >> "$LOG" 2>&1
  echo "----- ${TAG} DONE $(date -u +%H:%M:%SZ) -----" >> "$LOG"
}

# Helper: run all 4 settings (k16/k8 × exp/rg)
run_4settings () {
  local LR=$1 EP=$2 KA=$3 METHODS=$4
  for K in 16 8; do
    for DIST in exponential relu_gaussian; do
      run_main $K independent $DIST $LR $EP $KA "$METHODS"
    done
  done
}

# ============================================================
# Part 1: LR sweep (epochs=10, k_align=6) — ALL methods
# ============================================================
echo "====== Part 1: LR sweep (ep=10) ======" >> "$LOG"
for LR in 5e-4 1e-3 5e-3; do
  run_4settings $LR 10 6 "$ALL_METHODS"
done

# ============================================================
# Part 2: 20-epoch — ALL methods at k_align=6, then ours-only k_align sweep
# ============================================================
echo "====== Part 2: 20 epochs ======" >> "$LOG"
# All methods with default k_align=6
run_4settings 2e-4 20 6 "$ALL_METHODS"
# ours-only k_align sweep (skip 6, already done above)
for KA in 4 8 12 16; do
  if [ "$KA" -eq 6 ]; then continue; fi
  run_4settings 2e-4 20 $KA "ours"
done

# ============================================================
# Part 3: 30-epoch — ALL methods at k_align=6, then ours-only k_align sweep
# ============================================================
echo "====== Part 3: 30 epochs ======" >> "$LOG"
# All methods with default k_align=6
run_4settings 2e-4 30 6 "$ALL_METHODS"
# ours-only k_align sweep (skip 6, already done above)
for KA in 4 8 12 16 20 24; do
  if [ "$KA" -eq 6 ]; then continue; fi
  run_4settings 2e-4 30 $KA "ours"
done

echo "===== sweep DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" >> "$LOG"
