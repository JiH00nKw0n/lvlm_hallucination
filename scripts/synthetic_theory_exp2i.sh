#!/usr/bin/env bash
# Exp 2(i): Two SAEs -- shared-private entanglement
# Sweep n_private, post-hoc permutation matching

set -euo pipefail

python synthetic_sae_theory_experiment.py \
    --experiment 2i \
    --n-private-values "0,8,16,32,64,128" \
    --n-shared 64 \
    --representation-dim 768 \
    --k 4 \
    --num-epochs 3 \
    --num-seeds 5 \
    --seed-base 42 \
    --lr 2e-4 \
    --batch-size 256 \
    --sparsity 0.999 \
    --max-interference 0.3 \
    --dictionary-strategy gradient \
    --device auto \
    --output-root outputs/synthetic_theory \
    "$@"
