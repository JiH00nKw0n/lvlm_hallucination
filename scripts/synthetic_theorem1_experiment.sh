#!/usr/bin/env bash
# Theorem 1 experiments: Single SAE impossibility/degradation
#
# Exp 1a: Mismatch (α) × capacity (latent_size) sweep
#   → Shows that C1∧C2∧C3 cannot hold when Φ_S ≠ Ψ_S
#
# Exp 1b: Private feature × capacity sweep, Φ_S = Ψ_S
#   → Shows practical degradation even without theoretical impossibility

set -euo pipefail

COMMON="--representation-dim 768 --k 64 \
    --num-epochs 10 --lr 2e-4 --batch-size 256 \
    --sparsity 0.99 --max-interference 0.1 \
    --dictionary-strategy gradient \
    --num-train 50000 --num-eval 10000 \
    --seed-base 1 \
    --device auto \
    --output-root outputs/synthetic_theorem1"

echo "=== Exp 1a: Mismatch × Capacity sweep ==="
python synthetic_theorem1_experiment.py \
    --experiment 1a \
    --alpha-sweep "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
    --latent-size-sweep "2048,4096,8192" \
    --n-shared 512 --n-image 256 --n-text 256 \
    $COMMON "$@"

echo "=== Exp 1b: Private × Capacity sweep ==="
python synthetic_theorem1_experiment.py \
    --experiment 1b \
    --n-private-sweep "0,128,256,512" \
    --latent-size-sweep "2048,4096,8192" \
    --n-shared 512 \
    $COMMON "$@"
