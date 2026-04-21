#!/usr/bin/env bash
# Resume real sweep: only the 2 remaining per_epoch+revive variants.
# Uses EMA-C (already synced) so each should take ~30-45 min instead of 3.5h.

set -euo pipefail
cd /mnt/working/lvlm_hallucination
source .venv/bin/activate
mkdir -p .log outputs/aux_alignment_clip_b32

VARIANTS=(
  "barlow_perepoch+revive"
  "infonce_perepoch+revive"
)

declare -A LAM=(
  [barlow_perepoch+revive]=0.25
  [infonce_perepoch+revive]=0.01
)

# Clean up partial output from crashed run so we re-train from scratch.
for V in "${VARIANTS[@]}"; do
  rm -rf "outputs/aux_alignment_clip_b32/$V"
done

for V in "${VARIANTS[@]}"; do
  LAMBDA="${LAM[$V]}"
  echo "=== variant: $V (lambda=$LAMBDA, EMA-C) ==="
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

echo "=== ALL REMAINING VARIANTS COMPLETE ==="
