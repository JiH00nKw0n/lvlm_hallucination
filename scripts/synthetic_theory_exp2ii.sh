#!/usr/bin/env bash
# Exp 2(ii): Two SAEs -- latent non-identifiability
# No private features, vary seed only, post-hoc matching

set -euo pipefail

python synthetic_sae_theory_experiment.py \
    --experiment 2ii \
    --n-shared 64 \
    --representation-dim 768 \
    --k 4 \
    --num-epochs 3 \
    --num-seeds 10 \
    --seed-base 42 \
    --lr 2e-4 \
    --batch-size 256 \
    --sparsity 0.999 \
    --max-interference 0.3 \
    --dictionary-strategy gradient \
    --device auto \
    --output-root outputs/synthetic_theory \
    "$@"
