#!/usr/bin/env bash
# Lambda sweep at fixed alpha (followup15 style)
set -euo pipefail

cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log

python run_synthetic_v2.py \
  --config configs/synthetic/lambda_sweep.yaml \
  2>&1 | tee .log/v2_lambda_sweep.log
