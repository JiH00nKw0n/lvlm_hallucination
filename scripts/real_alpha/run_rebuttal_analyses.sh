#!/usr/bin/env bash
# Everything that runs off the correlation matrices, once run_rebuttal_EA.sh has
# built them: match confidence, one-to-many span, alignment ceiling, stability.
#
#   bash scripts/real_alpha/run_rebuttal_analyses.sh                # setting F
#   SETTING=T bash scripts/real_alpha/run_rebuttal_analyses.sh      # setting T
#
# No training and no GPU-bound work except the noise-floor correlation pass,
# which re-encodes the dataset once with the image-caption pairing destroyed.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f env.sh ] && source env.sh || true
[ -f /mnt/elice/working/env.sh ] && source /mnt/elice/working/env.sh || true

PY=python; [ -x .venv/bin/python ] && PY=.venv/bin/python
SETTING="${SETTING:-F}"
RA="${RA:-1}"; RB="${RB:-2}"
TAU="${TAU:-0.4}"

if [ "$SETTING" = "F" ]; then
  TAG=coco_k8; DATASET=coco; CACHE=cache/clip_b32_coco; MAXS=0
else
  TAG=cc3m_k32; DATASET=cc3m; CACHE=cache/clip_b32_cc3m; MAXS="${MAXS:-500000}"
fi

EA="outputs/rebuttal_EA/${TAG}_r${RA}r${RB}"
CKPT_A="outputs/rebuttal_models/${TAG}_r${RA}/final"
CKPT_B="outputs/rebuttal_models/${TAG}_r${RB}/final"
[ -f "$EA/C_img_txt.npz" ] || { echo "run run_rebuttal_EA.sh first: $EA/C_img_txt.npz missing" >&2; exit 1; }

banner(){ echo; echo "======== $* ($(date -u +%H:%M:%SZ)) ========"; }

banner "noise floor: recompute correlations with the pairing destroyed"
NULL="outputs/rebuttal_EC/${TAG}_r${RA}/C_img_txt_shuffled.npz"
if [ -f "$NULL" ]; then echo "[skip] $NULL exists"; else
  $PY scripts/real_alpha/build_cross_pair_C.py \
      --ckpt-a "$CKPT_A" --panel img_txt --dataset "$DATASET" --cache-dir "$CACHE" \
      --split train --max-samples "$MAXS" --shuffle-b 7 --out "$NULL"
fi

banner "E-C  match confidence"
$PY scripts/real_alpha/analyze_match_confidence.py \
    --panel "$EA/C_img_txt.npz" --null-panel "$NULL" --ckpt "$CKPT_A" \
    --out "outputs/rebuttal_EC/${TAG}_r${RA}"

banner "E-E  one-to-many span residual"
$PY scripts/real_alpha/analyze_1toN_span.py \
    --panel "$EA/C_img_txt.npz" --ckpt "$CKPT_A" --tau "$TAU" \
    --out "outputs/rebuttal_EE/${TAG}_r${RA}"

banner "E-F  alignment ceiling"
$PY scripts/real_alpha/analyze_alignment_ceiling.py \
    --panel "$EA/C_img_txt.npz" --ckpt "$CKPT_A" \
    --out "outputs/rebuttal_EF/${TAG}_r${RA}"

if [ -f "$EA/C_img_img.npz" ]; then
  banner "E-D  stability-conditioned distance"
  $PY scripts/real_alpha/analyze_stability_conditioned.py \
      --panel-img-img "$EA/C_img_img.npz" --panel-img-txt "$EA/C_img_txt.npz" \
      --ckpt-a "$CKPT_A" --ckpt-b "$CKPT_B" \
      --out "outputs/rebuttal_ED/${TAG}_r${RA}r${RB}"
fi

echo
echo "REBUTTAL_ANALYSES_DONE ($TAG)"
