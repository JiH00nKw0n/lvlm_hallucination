#!/bin/bash

# Run greedy vs. contrastive vs. VCD decoding on the MME color subset.
# Usage:
#   ./scripts/test_mme_color_contrast.sh [gpu] [max_samples] [noise_scale] [max_new_tokens] [output_json] [cd_alpha] [cd_beta]
# Example:
#   ./scripts/test_mme_color_contrast.sh 0 200 0.5 32 results/mme_color_contrast.json 0.5 0.1

set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")

GPU_ID=${1:-0}
MAX_SAMPLES=${2:-100}
NOISE_SCALE=${3:-0.5}
MAX_NEW_TOKENS=${4:-32}
OUTPUT_JSON=${5:-"$PROJECT_DIR/results/mme_color_contrast.json"}
CD_ALPHA=${6:-0.5}
CD_BETA=${7:-0.1}
MODEL_NAME=${MODEL_NAME:-"llava-hf/llava-1.5-7b-hf"}
NOISE_MODE=${NOISE_MODE:-"vibrant_hue"}
IMAGE_OUTPUT_DIR=${IMAGE_OUTPUT_DIR:-"$PROJECT_DIR/results/mme_color_contrast_images"}

OUTPUT_DIR=$(dirname "$OUTPUT_JSON")
if [[ -n "$OUTPUT_DIR" ]]; then
  mkdir -p "$OUTPUT_DIR"
fi
mkdir -p "$IMAGE_OUTPUT_DIR"

export HF_HOME="$PROJECT_DIR/.cache"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache"

echo "Running MME color contrastive test"
echo "----------------------------------"
echo "GPU:             $GPU_ID"
echo "Model:           $MODEL_NAME"
echo "Max samples:     $MAX_SAMPLES"
echo "Noise scale:     $NOISE_SCALE"
echo "Max new tokens:  $MAX_NEW_TOKENS"
echo "CD alpha/beta:   $CD_ALPHA / $CD_BETA"
echo "Noise mode:      $NOISE_MODE"
echo "Output JSON:     $OUTPUT_JSON"
echo "Image dir:       $IMAGE_OUTPUT_DIR"
echo ""

CUDA_VISIBLE_DEVICES=$GPU_ID python "$PROJECT_DIR/test_mme_color_contrast.py" \
  --model-name "$MODEL_NAME" \
  --max-samples "$MAX_SAMPLES" \
  --noise-scale "$NOISE_SCALE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --noise-mode "$NOISE_MODE" \
  --cd-alpha "$CD_ALPHA" \
  --cd-beta "$CD_BETA" \
  --image-output-dir "$IMAGE_OUTPUT_DIR" \
  --output-json "$OUTPUT_JSON"
