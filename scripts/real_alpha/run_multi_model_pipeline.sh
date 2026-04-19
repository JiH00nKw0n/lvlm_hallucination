#!/usr/bin/env bash
# Multi-model pipeline: extract embeddings → train two-SAE → Diagnostic B → boxplot
#
# Usage (on server):
#   cd /mnt/working/lvlm_hallucination
#   source .venv/bin/activate
#   export HF_TOKEN=<your_hf_token>
#   bash scripts/real_alpha/run_multi_model_pipeline.sh 2>&1 | tee .log/multi_model_pipeline.log
#
# Prerequisites:
#   pip install open_clip_torch

set -euo pipefail

EXTRACT="python scripts/real_alpha/extract_clip_coco_cache.py"
TRAIN="python scripts/real_alpha/train_real_sae.py"
DIAG="python scripts/real_alpha/run_diagnostic_B.py"

LATENT=8192
EPOCHS=30
BATCH=1024
DEVICE=cuda

# ── Model definitions ──
# Format: "backend|model|pretrained|cache_dir|output_dir|hidden_size"
MODELS=(
  "transformers|openai/clip-vit-large-patch14||cache/clip_l14_coco|outputs/clip_l14|768"
  "transformers|facebook/metaclip-l14-400m||cache/metaclip_l14_coco|outputs/metaclip_l14|768"
  "openclip|ViT-L-14|datacomp_xl_s13b_b90k|cache/datacomp_l14_coco|outputs/datacomp_l14|768"
  "openclip|MobileCLIP2-L-14|dfndr2b|cache/mobileclip2_l14_coco|outputs/mobileclip2_l14|768"
  "transformers|google/siglip2-large-patch16-256||cache/siglip2_large_coco|outputs/siglip2_large|1024"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r backend model pretrained cache_dir output_dir hidden_size <<< "$entry"
  echo "============================================"
  echo "=== $model (backend=$backend, dim=$hidden_size)"
  echo "============================================"

  # 1. Extract embeddings
  extract_args="--backend $backend --model $model --cache-dir $cache_dir --device $DEVICE"
  if [ -n "$pretrained" ]; then
    extract_args="$extract_args --pretrained $pretrained"
  fi
  echo "[1/3] Extracting embeddings..."
  $EXTRACT $extract_args

  # 2. Train two-SAE
  echo "[2/3] Training two-SAE..."
  $TRAIN --variant two_sae \
    --cache-dir "$cache_dir" \
    --output-dir "$output_dir/two_sae" \
    --latent $LATENT --hidden-size "$hidden_size" \
    --epochs $EPOCHS --batch-size $BATCH

  # 3. Diagnostic B
  echo "[3/3] Running Diagnostic B..."
  $DIAG --run-dir "$output_dir/two_sae/final" --cache-dir "$cache_dir"

  echo "=== DONE: $model ==="
  echo ""
done

echo "============================================"
echo "=== ALL MODELS COMPLETE ==="
echo "============================================"

# Print summary
echo ""
echo "Summary:"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r backend model pretrained cache_dir output_dir hidden_size <<< "$entry"
  if [ -f "$output_dir/two_sae/final/diagnostic_B.json" ]; then
    echo "  $model: $(python3 -c "import json; d=json.load(open('$output_dir/two_sae/final/diagnostic_B.json')); print(f\"median={d['all']['median']:.4f}, mean={d['all']['mean']:.4f}\")")"
  else
    echo "  $model: FAILED (no diagnostic_B.json)"
  fi
done
