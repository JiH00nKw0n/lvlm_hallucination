#!/usr/bin/env bash
# InfoNCE-only low-lambda sweep on synthetic.
set -euo pipefail
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log
python synthetic_aux_alignment.py \
  --config configs/synthetic/aux_alignment_infonce_lowlam.yaml \
  2>&1 | tee .log/aux_alignment_synthetic_infonce_lowlam.log
