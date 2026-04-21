#!/usr/bin/env bash
# Compare dead-latent rescue strategies (3-variant, all recon_only, 2-SAE):
#   1) recon_only           — already trained (outputs/aux_alignment_clip_b32/recon_only)
#   2) recon_only_auxk_1ep  — AuxK with dead threshold = 1 epoch (566747 tokens)
#   3) recon_only_revive    — random-init dead slots every epoch
#
# After all three complete, run post-hoc Hungarian analysis to compare
# alive count / matched-pair joint_mgt / decoder cosine distribution.

set -euo pipefail

cd /mnt/working/lvlm_hallucination
source .venv/bin/activate

mkdir -p .log outputs/aux_alignment_clip_b32

# Common training flags.
COMMON_ARGS=(
    --variant two_sae
    --cache-dir cache/clip_b32_coco
    --latent 8192
    --k 8
    --hidden-size 512
    --epochs 30
    --batch-size 1024
    --lr 5e-4
    --warmup-ratio 0.05
    --weight-decay 1e-5
    --max-grad-norm 1.0
    --seed 1
    --dataloader-num-workers 2
)

# 1-epoch dead threshold = train split size (566747 paired samples).
ONE_EPOCH_TOKENS=566747

# -----------------------------------------------------------------------------
# Run A: recon_only + AuxK (k_aux=16, 1-epoch dead threshold)
# -----------------------------------------------------------------------------
echo "[$(date)] Launching recon_only_auxk_1ep"
python scripts/real_alpha/train_real_sae.py \
    "${COMMON_ARGS[@]}" \
    --output-dir outputs/aux_alignment_clip_b32/recon_only_auxk_1ep \
    --auxk-weight 1.0 \
    --k-aux 16 \
    --dead-feature-threshold "$ONE_EPOCH_TOKENS" \
    2>&1 | tee .log/recon_only_auxk_1ep.log

# -----------------------------------------------------------------------------
# Run B: recon_only + revive (random init dead slots every epoch)
# -----------------------------------------------------------------------------
echo "[$(date)] Launching recon_only_revive"
python scripts/real_alpha/train_real_sae.py \
    "${COMMON_ARGS[@]}" \
    --output-dir outputs/aux_alignment_clip_b32/recon_only_revive \
    --revive-every-epoch \
    2>&1 | tee .log/recon_only_revive.log

echo "[$(date)] All 3-variant training runs complete."
