#!/usr/bin/env bash
# Lambda sweep on synthetic aux-alignment: 25 methods (8 variants x 3 lambdas + recon).
# Single (alpha=0.5, seed=1). Run on elice-40g.

set -euo pipefail
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log

python synthetic_aux_alignment.py \
  --config configs/synthetic/aux_alignment_lambda_sweep.yaml \
  2>&1 | tee .log/aux_alignment_synthetic_lambda.log
