#!/usr/bin/env bash
# Real aux-alignment ablation on CLIP B/32 COCO.
# 7 variants at best-per-variant lambda (naive/barlow: 0.25, InfoNCE: 0.01).
# Single (seed=1) point. Run on elice-40g.
#
# Settings match experiment.sh (multi-model boxplot main):
#   model = openai/clip-vit-base-patch32, hidden=512
#   --latent 8192 -> two_sae per-side 4096
#   30 epochs, batch 1024

set -euo pipefail
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log outputs/aux_alignment_clip_b32

# variant -> lambda (from synthetic lambda sweep best HP).
# recon_only: no aux so lambda irrelevant (passed but ignored by the script).
declare -A LAM=(
  [recon_only]=0.0
  [naive_once]=0.25
  [barlow_once]=0.25
  [infonce_once]=0.01
  [naive_perepoch+revive]=0.25
  [barlow_perepoch+revive]=0.25
  [infonce_perepoch+revive]=0.01
)

VARIANTS=(
  recon_only
  naive_once
  barlow_once
  infonce_once
  "naive_perepoch+revive"
  "barlow_perepoch+revive"
  "infonce_perepoch+revive"
)

for V in "${VARIANTS[@]}"; do
  LAMBDA="${LAM[$V]}"
  echo "============================================"
  echo "=== variant: $V (lambda=$LAMBDA)"
  echo "============================================"
  V_SAFE="${V//+/__}"
  python scripts/real_alpha/real_aux_alignment.py \
    --variant-name "$V" \
    --cache-dir cache/clip_b32_coco \
    --output-dir "outputs/aux_alignment_clip_b32/$V" \
    --hidden-size 512 --latent 8192 --k 8 \
    --epochs 30 --batch-size 1024 \
    --lr 5e-4 --warmup-ratio 0.05 --weight-decay 1e-5 \
    --rho0 0.2 --lambda-aux "$LAMBDA" --seed 1 \
    --k-align 15 \
    2>&1 | tee ".log/aux_alignment_real_${V_SAFE}.log"
done

echo "============================================"
echo "=== ALL VARIANTS COMPLETE ==="
echo "============================================"
