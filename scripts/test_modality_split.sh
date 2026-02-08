#!/bin/bash

# Modality split analysis for SAE features
# Usage: ./scripts/test_modality_split.sh [gpu_id]
# Example: ./scripts/test_modality_split.sh 0

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
LOG_FILE="$PROJECT_DIR/.log/modality_split_${TIMESTAMP}.log"

echo "=========================================="
echo "SAE Feature Modality Split Analysis"
echo "=========================================="
echo "GPU: $DEVICE"
echo "Log: $LOG_FILE"
echo "=========================================="

python "$PROJECT_DIR/test/test_modality_split.py" \
    --model_name llava-hf/llama3-llava-next-8b-hf \
    --sae_path lmms-lab/llama3-llava-next-8b-hf-sae-131k \
    --layer_index 24 \
    --num_samples 5000 \
    --k 256 \
    --bin_width 0.05 \
    --seed 42 \
    --dtype float16 \
    --output_dir "$PROJECT_DIR/results/modality_split" \
    2>&1 | tee -a "$LOG_FILE"

echo "=========================================="
echo "Done. Results in: $PROJECT_DIR/results/modality_split/"
echo "=========================================="
