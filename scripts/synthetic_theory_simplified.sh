#!/usr/bin/env bash
# Synthetic Theory Simplified: 2D Multimodal SAE experiments
#
# Exp A:  m sweep (1~5) × θ sweep (0~90°)
# Exp B1: group-sparse λ sweep × θ sweep (m=4)
# Exp B2: trace alignment λ sweep × θ sweep (m=4)
# All runs: no-bias only

set -euo pipefail

COMMON="--num-train 10000 --num-epochs 10 --lr 1e-3 --batch-size 64 \
    --seed 2026 --viz-every 5 --fps 10 \
    --theta-values 0,10,20,30,90 \
    --single-bias --no-bias \
    --device auto"

OUT="outputs/synthetic_theory_simplified"

echo "============================================================"
echo "  Experiment A: Capacity Sweep (no bias)"
echo "============================================================"
python synthetic_theory_simplified.py \
    --experiment A \
    --m-values "1,2,3,4,5" \
    --output-dir "$OUT" \
    $COMMON "$@"

echo "============================================================"
echo "  Experiment B1: Group-Sparse λ Sweep (no bias)"
echo "============================================================"
python synthetic_theory_simplified.py \
    --experiment B1 \
    --lambda-values "0,0.25,0.5,1,2,4,8,16,32,64,128,256,512" \
    --output-dir "$OUT" \
    $COMMON "$@"

echo "============================================================"
echo "  Experiment B2: Trace Alignment λ Sweep (no bias)"
echo "============================================================"
python synthetic_theory_simplified.py \
    --experiment B2 \
    --lambda-values "0,0.25,0.5,1,2,4,8,16,32,64,128,256,512" \
    --output-dir "$OUT" \
    $COMMON "$@"

echo "============================================================"
echo "  Generating unified report from all metrics..."
echo "============================================================"
python synthetic_theory_simplified.py \
    --experiment report \
    --output-dir "$OUT"

echo "============================================================"
echo "  All experiments complete! (155 runs)"
echo "  Output: $OUT/"
echo "  Report: $OUT/experiment_report.md"
echo "============================================================"
