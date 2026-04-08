#!/usr/bin/env bash
# Synthetic Theory Simplified: 2D Multimodal SAE experiments
#
# Exp A:  m sweep (1~5) × θ sweep (0,10,20,30,90)
# Exp B1: group-sparse λ sweep × θ={10,20} (m=4)
# Exp B2: trace alignment λ sweep × θ={10,20} (m=4)
# All runs: no-bias, decoder norm ON, 3 random seeds + optimal init → best

set -euo pipefail

COMMON_BASE="--num-train 30000 --num-eval 10000 --num-epochs 3 --lr 1e-3 --batch-size 64 \
    --seed 2026 --viz-every 5 --fps 10 \
    --single-bias --no-bias \
    --device auto"

OUT="outputs/synthetic_theory_simplified"

echo "============================================================"
echo "  Experiment A: Capacity Sweep"
echo "============================================================"
python synthetic_theory_simplified.py \
    --experiment A \
    --theta-values "0,10,20,30,90" \
    --m-values "1,2,3,4,5" \
    --output-dir "$OUT" \
    $COMMON_BASE "$@"

echo "============================================================"
echo "  Experiment B1: Group-Sparse λ Sweep (θ=10,20)"
echo "============================================================"
python synthetic_theory_simplified.py \
    --experiment B1 \
    --theta-values "10,20" \
    --lambda-values "0,0.5,2,8,32,128,512,2048,8192,32768" \
    --output-dir "$OUT" \
    $COMMON_BASE "$@"

echo "============================================================"
echo "  Experiment B2: Trace Alignment λ Sweep (θ=10,20)"
echo "============================================================"
python synthetic_theory_simplified.py \
    --experiment B2 \
    --theta-values "10,20" \
    --lambda-values "0,0.5,2,8,32,128,512,2048,8192,32768" \
    --output-dir "$OUT" \
    $COMMON_BASE "$@"

echo "============================================================"
echo "  Generating unified report from all metrics..."
echo "============================================================"
python synthetic_theory_simplified.py \
    --experiment report \
    --output-dir "$OUT"

echo "============================================================"
echo "  All experiments complete! (~53 configs)"
echo "  Output: $OUT/"
echo "  Report: $OUT/experiment_report.md"
echo "============================================================"
