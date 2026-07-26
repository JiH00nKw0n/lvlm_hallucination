#!/usr/bin/env bash
set -euo pipefail
cd /mnt/elice/working/lvlm_hallucination
PY=.venv/bin/python
CK=outputs/rebuttal_models/cc3m_k32_r1/final
OUT=outputs/rebuttal_full/cc3m_k32_r1
mkdir -p "$OUT"

echo "=== [1/3] full-dataset img_txt correlation panel ==="
date
$PY scripts/real_alpha/build_cross_pair_C.py \
  --ckpt-a "$CK" --panel img_txt \
  --dataset cc3m --cache-dir cache/clip_b32_cc3m --max-samples 0 \
  --batch-size 8192 --out "$OUT/C_img_txt_full.npz"

echo "=== [2/3] match confidence on the full panel ==="
date
$PY scripts/real_alpha/analyze_match_confidence.py \
  --ckpt "$CK" --panel "$OUT/C_img_txt_full.npz" \
  --out "$OUT/match_confidence"

echo "=== [3/3] 1:N splitting on the full dataset ==="
date
$PY scripts/real_alpha/analyze_1toN_splitting.py \
  --ckpt "$CK" --panel "$OUT/C_img_txt_full.npz" \
  --dataset cc3m --cache-dir cache/clip_b32_cc3m --max-samples 0 \
  --batch-size 8192 --out "$OUT/splitting"

date
echo "FULL_CC3M_DONE"
