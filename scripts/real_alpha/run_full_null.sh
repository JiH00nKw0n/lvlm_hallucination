#!/usr/bin/env bash
set -euo pipefail
cd /mnt/elice/working/lvlm_hallucination
PY=.venv/bin/python
CK=outputs/rebuttal_models/cc3m_k32_r1/final
OUT=outputs/rebuttal_full/cc3m_k32_r1
date
$PY scripts/real_alpha/build_cross_pair_C.py \
  --ckpt-a "$CK" --panel img_txt --shuffle-b 1234 \
  --dataset cc3m --cache-dir cache/clip_b32_cc3m --max-samples 0 \
  --batch-size 8192 --out "$OUT/C_img_txt_full_shuffled.npz"
$PY scripts/real_alpha/analyze_match_confidence.py \
  --ckpt "$CK" --panel "$OUT/C_img_txt_full.npz" \
  --null-panel "$OUT/C_img_txt_full_shuffled.npz" \
  --out "$OUT/match_confidence"
date
echo FULL_NULL_DONE
