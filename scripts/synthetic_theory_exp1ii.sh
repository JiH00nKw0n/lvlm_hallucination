#!/usr/bin/env bash
# Exp 1(ii): Single SAE -- generative mapping mismatch
# Sweep contrastive loss target with no private features

set -euo pipefail

python synthetic_sae_theory_experiment.py \
    --experiment 1ii \
    --cl-target-values "0.1,0.2,0.5,1.0" \
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
    --logit-scale-init 2.6592 \
    --calibration-num-samples 2048 \
    --calibration-lr 0.01 \
    --calibration-max-iters 2000 \
    --lambda-interference 10.0 \
    --device auto \
    --output-root outputs/synthetic_theory \
    "$@"
