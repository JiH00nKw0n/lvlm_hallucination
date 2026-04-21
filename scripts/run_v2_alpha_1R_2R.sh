#!/usr/bin/env bash
# Wide alpha sweep: single_recon vs two_recon (followup16 style)
set -euo pipefail

cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log

python run_synthetic_v2.py \
  --config configs/synthetic/alpha_1R_2R.yaml \
  2>&1 | tee .log/v2_alpha_1R_2R.log
