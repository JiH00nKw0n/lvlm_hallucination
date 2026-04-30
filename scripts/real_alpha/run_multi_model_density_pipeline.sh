#!/usr/bin/env bash
# End-to-end pipeline for the 8-model multi-VLM density figure
# (paper §3.3, outputs/multi_model_density_base_large.{pdf,png,svg}).
#
# For each of 4 Base + 4 Large VLMs:
#   1. extract image/text embeddings on COCO Karpathy
#   2. train modality-specific TopK SAEs (--variant two_sae)
#   3. compute the train-time cross-correlation matrix C (Diagnostic B)
# Finally render the 2x5 density grid.
#
# Each step is idempotent: it skips when the canonical artifact already exists.
# Fail mid-way → re-run; the loop resumes from the failed step.
#
# ─── Usage ─────────────────────────────────────────────────────────────────
# Native:
#   export HF_TOKEN=hf_xxx                 # required for gated weights
#   bash scripts/real_alpha/run_multi_model_density_pipeline.sh \
#        2>&1 | tee .log/multi_model_density_pipeline.log
#
# Docker (recommended for collaborator handoff):
#   docker build -t lvlm-multi-density .
#   docker run --rm --gpus all \
#       -e HF_TOKEN=$HF_TOKEN \
#       -v $PWD/cache:/workspace/lvlm_hallucination/cache \
#       -v $PWD/outputs:/workspace/lvlm_hallucination/outputs \
#       -v $PWD/.log:/workspace/lvlm_hallucination/.log \
#       lvlm-multi-density
#
# ─── Env overrides ─────────────────────────────────────────────────────────
#   LATENT=8192   SAE width
#   K=8           TopK
#   EPOCHS=30
#   BATCH=1024    train batch size (drop to 512 on small GPUs)
#   DEVICE=cuda
#   PLOT_OUT=outputs/multi_model_density_base_large.pdf
#
# ─── Prerequisites ─────────────────────────────────────────────────────────
#   - Single GPU, ~10GB+ VRAM (B/32 fits easily; L/14 needs ~8GB train)
#   - ~40GB free disk (cache ~26GB + outputs 5–10GB + image)
#   - HF account + token; SigLIP2 weights are gated, accept the EULA on the
#     model page first (https://huggingface.co/google/siglip2-base-patch16-224)
#
# ─── Common failures ───────────────────────────────────────────────────────
#   401 from HF       → HF_TOKEN missing or model EULA not accepted.
#   CUDA OOM (train)  → re-run with `BATCH=512 bash …`.
#   open_clip missing → `pip install open_clip_torch` (Docker image has it).

set -euo pipefail

cd "$(dirname "$0")/../.."

EXTRACT="python scripts/real_alpha/extract_clip_coco_cache.py"
TRAIN="python scripts/real_alpha/train_real_sae.py"
DIAG="python scripts/real_alpha/run_diagnostic_B.py"
PLOT="python scripts/real_alpha/plot_multi_model_density_base_large.py"

LATENT="${LATENT:-8192}"
K="${K:-8}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-1024}"
DEVICE="${DEVICE:-cuda}"
PLOT_OUT="${PLOT_OUT:-outputs/multi_model_density_base_large.pdf}"

mkdir -p .log cache outputs

# ── 8-model definitions ────────────────────────────────────────────────────
# Format: "backend|model|pretrained|cache_dir|output_dir|hidden_size"
# output_dir basename must match plot_multi_model_density_base_large.py:
#   BASE_MODELS  = clip_b32, metaclip_b32, datacomp_b32, siglip2_base
#   LARGE_MODELS = clip_l14, metaclip_l14, datacomp_l14, siglip2_large
MODELS=(
  # Base
  "transformers|openai/clip-vit-base-patch32||cache/clip_b32_coco|outputs/clip_b32|512"
  "transformers|facebook/metaclip-b32-400m||cache/metaclip_b32_coco|outputs/metaclip_b32|512"
  "openclip|ViT-B-32|datacomp_xl_s13b_b90k|cache/datacomp_b32_coco|outputs/datacomp_b32|512"
  "transformers|google/siglip2-base-patch16-224||cache/siglip2_base_coco|outputs/siglip2_base|768"
  # Large
  "transformers|openai/clip-vit-large-patch14||cache/clip_l14_coco|outputs/clip_l14|768"
  "transformers|facebook/metaclip-l14-400m||cache/metaclip_l14_coco|outputs/metaclip_l14|768"
  "openclip|ViT-L-14|datacomp_xl_s13b_b90k|cache/datacomp_l14_coco|outputs/datacomp_l14|768"
  "transformers|google/siglip2-large-patch16-256||cache/siglip2_large_coco|outputs/siglip2_large|1024"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r backend model pretrained cache_dir output_dir hidden_size <<< "$entry"
  echo "============================================"
  echo "=== $model (backend=$backend, dim=$hidden_size)"
  echo "============================================"

  final_dir="$output_dir/two_sae/final"

  # 1. Extract embeddings
  if [ -f "$cache_dir/meta.json" ]; then
    echo "[1/3] cache exists → skip extract ($cache_dir)"
  else
    echo "[1/3] Extracting embeddings → $cache_dir"
    extract_args="--backend $backend --model $model --cache-dir $cache_dir --device $DEVICE"
    if [ -n "$pretrained" ]; then
      extract_args="$extract_args --pretrained $pretrained"
    fi
    $EXTRACT $extract_args
  fi

  # 2. Train two-SAE
  if [ -f "$final_dir/model.safetensors" ]; then
    echo "[2/3] checkpoint exists → skip train ($final_dir)"
  else
    echo "[2/3] Training two-SAE → $output_dir/two_sae"
    $TRAIN --variant two_sae \
      --cache-dir "$cache_dir" \
      --output-dir "$output_dir/two_sae" \
      --latent "$LATENT" --k "$K" --hidden-size "$hidden_size" \
      --epochs "$EPOCHS" --batch-size "$BATCH"
  fi

  # 3. Diagnostic B
  if [ -f "$final_dir/diagnostic_B_C_train.npy" ]; then
    echo "[3/3] diagnostic_B_C_train.npy exists → skip diag ($final_dir)"
  else
    echo "[3/3] Running Diagnostic B → $final_dir"
    $DIAG --run-dir "$final_dir" --cache-dir "$cache_dir"
  fi

  echo "=== DONE: $model ==="
  echo ""
done

echo "============================================"
echo "=== ALL MODELS COMPLETE ==="
echo "============================================"

# Summary
echo ""
echo "Diagnostic B summary:"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r backend model pretrained cache_dir output_dir hidden_size <<< "$entry"
  diag_json="$output_dir/two_sae/final/diagnostic_B.json"
  if [ -f "$diag_json" ]; then
    echo "  $model: $(python3 -c "import json; d=json.load(open('$diag_json')); print(f\"median={d['all']['median']:.4f}, mean={d['all']['mean']:.4f}\")")"
  else
    echo "  $model: MISSING ($diag_json)"
  fi
done

# Final plot
echo ""
echo "============================================"
echo "=== Rendering 2x4 density figure → $PLOT_OUT"
echo "============================================"
$PLOT --out "$PLOT_OUT"
echo ""
echo "Pipeline complete. Figure written to:"
echo "  ${PLOT_OUT}"
echo "  ${PLOT_OUT%.pdf}.png"
echo "  ${PLOT_OUT%.pdf}.svg"
