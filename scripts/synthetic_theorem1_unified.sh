#!/bin/bash
# Unified Theorem 1 experiments with diagnostics
# See papers/synthetic_theorem_1_fw.md for framework description

set -euo pipefail

COMMON="--k 16 --num-epochs 10 --lr 2e-4 --batch-size 256 --n-shared 512 --n-image 256 --n-text 256"

echo "=== Focused diagnostic: alpha 0.7-1.0 (H1 verification) ==="
python synthetic_theorem1_unified.py \
    --alpha-sweep "0.7,0.8,0.9,1.0" \
    --latent-size-sweep "2048" \
    --num-seeds 3 \
    --run-tag "diag_focus" \
    $COMMON "$@"

echo ""
echo "=== Full sweep: alpha 0.1-1.0 (calibration curve) ==="
python synthetic_theorem1_unified.py \
    --alpha-sweep "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
    --latent-size-sweep "2048" \
    --num-seeds 3 \
    --run-tag "full_3seeds" \
    $COMMON "$@"
