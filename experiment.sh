#!/usr/bin/env bash
# Full experiment: extract embeddings → train two-SAE → Diagnostic B for all models
#
# Usage:
#   docker build -f Dockerfile.experiment -t sae-experiment .
#   docker run --gpus all -v $(pwd)/outputs:/workspace/outputs -v $(pwd)/cache:/workspace/cache sae-experiment
#
# Or without docker:
#   pip install transformers==5.5.3 datasets==4.8.4 open_clip_torch==3.3.0 safetensors==0.7.0 scipy pillow tqdm matplotlib
#   export HF_TOKEN=${HF_TOKEN:?\"Set HF_TOKEN environment variable\"}
#   bash experiment.sh

set -euo pipefail

EXTRACT="python scripts/real_alpha/extract_clip_coco_cache.py"
TRAIN="python scripts/real_alpha/train_real_sae.py"
DIAG="python scripts/real_alpha/run_diagnostic_B.py"
PLOT="python scripts/real_alpha/plot_multi_model_boxplot.py"

LATENT=8192
EPOCHS=30
BATCH=1024
DEVICE=cuda

# ── Model definitions ──
# Format: "backend|model|pretrained|cache_dir|output_dir|hidden_size|display_name|color"
MODELS=(
  # --- Base / B-32 models ---
  "transformers|openai/clip-vit-base-patch32||cache/clip_b32_coco|outputs/clip_b32|512|CLIP ViT-B/32|#df3a3d"
  "transformers|facebook/metaclip-b32-400m||cache/metaclip_b32_coco|outputs/metaclip_b32|512|MetaCLIP B/32|#d96627"
  "openclip|ViT-B-32|datacomp_xl_s13b_b90k|cache/datacomp_b32_coco|outputs/datacomp_b32|512|DataComp B/32|#dfb246"
  "openclip|MobileCLIP2-B|dfndr2b|cache/mobileclip2b_coco|outputs/mobileclip2b|512|MobileCLIP2-B|#389076"
  "transformers|google/siglip2-base-patch16-224||cache/siglip2_base_coco|outputs/siglip2_base|768|SigLIP2 Base|#206987"
  # --- Large / L-14 models ---
  "transformers|openai/clip-vit-large-patch14||cache/clip_l14_coco|outputs/clip_l14|768|CLIP ViT-L/14|#df3a3d"
  "transformers|facebook/metaclip-l14-400m||cache/metaclip_l14_coco|outputs/metaclip_l14|768|MetaCLIP L/14|#d96627"
  "openclip|ViT-L-14|datacomp_xl_s13b_b90k|cache/datacomp_l14_coco|outputs/datacomp_l14|768|DataComp L/14|#dfb246"
  "openclip|MobileCLIP2-L-14|dfndr2b|cache/mobileclip2_l14_coco|outputs/mobileclip2_l14|768|MobileCLIP2-L/14|#389076"
  "transformers|google/siglip2-large-patch16-256||cache/siglip2_large_coco|outputs/siglip2_large|1024|SigLIP2 Large|#206987"
)

for entry in "${MODELS[@]}"; do
  IFS='|' read -r backend model pretrained cache_dir output_dir hidden_size display_name color <<< "$entry"
  echo "============================================"
  echo "=== $display_name ($model)"
  echo "============================================"

  # Skip if Diagnostic B already exists
  if [ -f "$output_dir/two_sae/final/diagnostic_B.json" ]; then
    echo "SKIP: diagnostic_B.json already exists"
    continue
  fi

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

  echo "=== DONE: $display_name ==="
  echo ""
done

echo "============================================"
echo "=== ALL MODELS COMPLETE ==="
echo "============================================"

# ── Summary ──
echo ""
echo "Summary:"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r backend model pretrained cache_dir output_dir hidden_size display_name color <<< "$entry"
  if [ -f "$output_dir/two_sae/final/diagnostic_B.json" ]; then
    echo "  $display_name: $(python3 -c "import json; d=json.load(open('$output_dir/two_sae/final/diagnostic_B.json')); print(f\"median={d['all']['median']:.4f}, mean={d['all']['mean']:.4f}\")")"
  else
    echo "  $display_name: FAILED"
  fi
done

# ── Generate boxplot config + plot ──
echo ""
echo "Generating boxplot..."
python3 -c "
import json
models = []
entries = '''$(printf '%s\n' "${MODELS[@]}")'''
for line in entries.strip().split('\n'):
    parts = line.split('|')
    output_dir, display_name, color = parts[4], parts[6], parts[7]
    diag = f'{output_dir}/two_sae/final/diagnostic_B.json'
    import os
    if os.path.exists(diag):
        models.append({'name': display_name, 'run_dir': f'{output_dir}/two_sae/final', 'color': color})
json.dump(models, open('outputs/boxplot_models.json', 'w'), indent=2)
print(f'wrote outputs/boxplot_models.json with {len(models)} models')
"
$PLOT --models outputs/boxplot_models.json --out outputs/multi_model_boxplot.pdf
echo "Saved outputs/multi_model_boxplot.pdf"
