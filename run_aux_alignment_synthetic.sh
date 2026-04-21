#!/usr/bin/env bash
# Synthetic aux-alignment ablation: 9 variants on Theorem-2 generator.
# Single (alpha=0.5, seed=1) point. Per-side latent = 4096.
# Run on the server (elice-40g).

set -euo pipefail
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log

python synthetic_aux_alignment.py \
  --config configs/synthetic/aux_alignment_compare.yaml \
  2>&1 | tee .log/aux_alignment_synthetic.log
