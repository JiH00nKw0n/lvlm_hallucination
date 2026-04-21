#!/usr/bin/env bash
# Coarse alpha sweep (0.0 step 0.2): single_recon vs two_recon, 5 seeds, L=8192.
# Purpose: provide α=0.0 params (missing from earlier runs) for fig1_v3 GRE panel.
set -euo pipefail

cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log

python run_synthetic_v2.py \
  --config configs/synthetic/alpha_1R_2R_L8192_5seeds_coarse.yaml \
  2>&1 | tee .log/v2_alpha_1R_2R_coarse.log
