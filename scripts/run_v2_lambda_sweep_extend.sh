#!/usr/bin/env bash
# Extend lambda sweep with 1/64 and 64 multipliers to bracket the phase
# transitions predicted by Propositions 3 (Iso-Energy, lambda*) and 4
# (Group-Sparse, lambda*/lambda^dagger). Original sweep stays at
# [1/16, 1/4, 1, 4, 16]; we add only the missing endpoints here.
set -euo pipefail

cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log

python run_synthetic_v2.py \
  --config configs/synthetic/lambda_sweep_extend.yaml \
  2>&1 | tee .log/v2_lambda_sweep_extend.log
