#!/bin/bash

# Run logit-lens analysis on local Colored MNIST images.
# Usage:
#   ./scripts/analyze_colored_mnist.sh [gpu_id] [num_samples] [start_index] [max_patches] [top_k] [samples_per_label] [output_path] [data_dir]
# Example:
#   ./scripts/analyze_colored_mnist.sh 0 -1 0 4 5 5 \
#       "$PROJECT_DIR/analysis_colored_mnist/logit_lens.json" "$PROJECT_DIR/test_2"

set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")

GPU_ID=${1:-0}
NUM_SAMPLES=${2:-50}
START_INDEX=${3:-0}
MAX_PATCHES=${4:-4}
TOP_K=${5:-5}
SAMPLES_PER_LABEL=${6:-5}
OUTPUT_PATH=${7:-"$PROJECT_DIR/analysis_colored_mnist/logit_lens_results.json"}
DATA_DIR=${8:-"$PROJECT_DIR/test_2"}
MODEL_NAME=${MODEL_NAME:-"llava-hf/llava-1.5-7b-hf"}
QUESTION=${QUESTION:-"What digit is shown in the image?"}
IMAGE_OUTPUT_DIR=${IMAGE_OUTPUT_DIR:-""}

if [[ -z "$IMAGE_OUTPUT_DIR" ]]; then
  IMAGE_OUTPUT_DIR=$(dirname "$OUTPUT_PATH")/images
fi

OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
if [[ -n "$OUTPUT_DIR" ]]; then
  mkdir -p "$OUTPUT_DIR"
fi
mkdir -p "$IMAGE_OUTPUT_DIR"

export HF_HOME="$PROJECT_DIR/.cache"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache"

echo "Running Colored MNIST logit-lens analysis"
echo "-----------------------------------------"
echo "GPU:            $GPU_ID"
echo "Model:          $MODEL_NAME"
echo "Num samples:    $NUM_SAMPLES"
echo "Start index:    $START_INDEX"
echo "Max patches:    $MAX_PATCHES"
echo "Top-K:          $TOP_K"
echo "Samples/label:  $SAMPLES_PER_LABEL"
echo "Output:         $OUTPUT_PATH"
echo "Image dir:      $IMAGE_OUTPUT_DIR"
echo "Data dir:       $DATA_DIR"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID python "$PROJECT_DIR/analyze_colored_mnist.py" \
  --model-name "$MODEL_NAME" \
  --num-samples "$NUM_SAMPLES" \
  --start-index "$START_INDEX" \
  --samples-per-label "$SAMPLES_PER_LABEL" \
  --max-patches "$MAX_PATCHES" \
  --top-k "$TOP_K" \
  --data-dir "$DATA_DIR" \
  --output-path "$OUTPUT_PATH" \
  --image-output-dir "$IMAGE_OUTPUT_DIR" \
  --question "$QUESTION"
