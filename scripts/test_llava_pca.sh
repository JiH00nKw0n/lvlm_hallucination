#!/bin/bash

# Run PCA analysis on LLaVA hidden states using llada_mask_generations.json
# Usage: ./scripts/test_llava_pca.sh [json_path] [model_name] [output_dir] [batch_size]

set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")

JSON_PATH=${1:-"$PROJECT_DIR/llada_mask_generations.json"}
MODEL_NAME=${2:-"llava-hf/llava-1.5-7b-hf"}
OUTPUT_DIR=${3:-"$PROJECT_DIR/llava_pca_outputs"}
BATCH_SIZE=${4:-2}

export HF_HOME="$PROJECT_DIR/.cache"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache"

echo "JSON path   : $JSON_PATH"
echo "Model name  : $MODEL_NAME"
echo "Output dir  : $OUTPUT_DIR"
echo "Batch size  : $BATCH_SIZE"
echo "Script dir  : $SCRIPT_DIR"
echo "Project dir : $PROJECT_DIR"
echo "-----------------------------------"

python "$PROJECT_DIR/test_llava_pca.py" \
  --json-path "$JSON_PATH" \
  --model-name "$MODEL_NAME" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "$BATCH_SIZE" \
  --top-k-pc 5 \
  --nearest-per-pc 5 \
  --dtype float16

echo "Done. Outputs saved to $OUTPUT_DIR"
