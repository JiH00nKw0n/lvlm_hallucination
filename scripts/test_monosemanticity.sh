#!/bin/bash

# Patch-level monosemanticity analysis for SAE features
# Usage: ./scripts/test_monosemanticity.sh [gpu_id]
# Example: ./scripts/test_monosemanticity.sh 0

# Get script directory and project root
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")

# Set cache directories
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache"
export HF_HOME="$PROJECT_DIR/.cache"
export TOKENIZERS_PARALLELISM=false

# GPU (default: 0)
DEVICE=${1:-0}
export CUDA_VISIBLE_DEVICES=$DEVICE

# Create log directory
mkdir -p "$PROJECT_DIR/.log"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_DIR/.log/monosemanticity_${TIMESTAMP}.log"

echo "=========================================="
echo "SAE Feature Monosemanticity Analysis"
echo "=========================================="
echo "GPU: $DEVICE"
echo "Log: $LOG_FILE"
echo "=========================================="

python "$PROJECT_DIR/test_monosemanticity.py" \
    --model_name llava-hf/llama3-llava-next-8b-hf \
    --sae_path lmms-lab/llama3-llava-next-8b-hf-sae-131k \
    --dinov2_name facebook/dinov2-large \
    --layer_index 24 \
    --num_samples 5000 \
    --k 256 \
    --top_patches 25 \
    --bin_width 0.05 \
    --seed 42 \
    --dtype float16 \
    --output_dir "$PROJECT_DIR/results/monosemanticity" \
    --weighted \
    2>&1 | tee -a "$LOG_FILE"

echo "=========================================="
echo "Done. Results in: $PROJECT_DIR/results/monosemanticity/"
echo "=========================================="
