#!/bin/bash

# Evaluation script for LVLM benchmarks
# Usage: ./scripts/eval.sh [gpu_ids]
# Example: ./scripts/eval.sh 0,1,2,3  # Use GPUs 0,1,2,3
#          ./scripts/eval.sh 0        # Use GPU 0 only
# Note: device_map="auto" in config automatically distributes model across available GPUs

# Set cache directories
export HF_DATASETS_CACHE="./.cache"
export HF_HOME="./.cache"
export LOG_DIR="./.log"

# Get GPU configuration
DEVICES=${1:-"0,1,2,3"}  # Default to GPUs 0,1,2,3

# Get script directory and project root
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PARENT_DIR=$(dirname "$SCRIPT_DIR")

# Configuration files to evaluate
CFG_PATHS=(
    "$PARENT_DIR/config/evaluate/llava-1.5-7b-hf.yaml"
    # Add more config paths here as needed
)

echo "=========================================="
echo "Starting LVLM Evaluation Pipeline"
echo "=========================================="
echo "GPUs: $DEVICES"
echo "Total configs: ${#CFG_PATHS[@]}"
echo "=========================================="

# Loop through each config file
for i in "${!CFG_PATHS[@]}"
do
    CFG_PATH="${CFG_PATHS[$i]}"

    echo ""
    echo "[$((i+1))/${#CFG_PATHS[@]}] Currently Running with Config: $CFG_PATH"

    # Validate config file exists
    if [ ! -f "$CFG_PATH" ]; then
        echo "Error: Configuration file not found: $CFG_PATH"
        echo "Skipping..."
        continue
    fi

    echo "=========================================="

    # Run evaluation with Accelerate for distributed inference
    if CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch --multi_gpu "$PARENT_DIR/evaluate.py" --cfg-path "$CFG_PATH"; then
        echo "✓ Config $((i+1))/${#CFG_PATHS[@]} completed successfully"
    else
        echo "✗ Config $((i+1))/${#CFG_PATHS[@]} failed"
        exit 1
    fi

    echo "=========================================="
done

echo ""
echo "=========================================="
echo "All evaluations completed successfully!"
echo "=========================================="
