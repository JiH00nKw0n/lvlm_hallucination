#!/bin/bash

# Run unified decoding experiments (greedy / noise-contrastive / instruction-rotation) and optional PCA diagnostics.
# Usage:
#   ./scripts/test_decoding.sh [gpu] [max_samples] [max_new_tokens] [noise_scale] [rotation_degrees]
# Examples:
#   ./scripts/test_decoding.sh 0 200 32 0.5 "5,10,15"
#   RUN_PCA_TEXT=1 RUN_PCA_IMAGE=1 ./scripts/test_decoding.sh 0 -1 32 0.5 "5,10"

set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")

# Set cache directories
export LOG_DIR="$PROJECT_DIR/.log"

GPU_ID=${1:-0}
RAW_MAX_SAMPLES=${2:--1}
MAX_SAMPLES=$RAW_MAX_SAMPLES
if [[ "$RAW_MAX_SAMPLES" == "all" ]]; then
  MAX_SAMPLES=-1
fi
MAX_NEW_TOKENS=${3:-32}
NOISE_SCALE=${4:-0.5}
ROTATION_DEGREES=${5:-"5,10,15"}

MODEL_NAME=${MODEL_NAME:-"llava-hf/llava-1.5-7b-hf"}
QUESTION_SUFFIX=${QUESTION_SUFFIX:-"Please answer with Yes or No."}
PICO_JSONL=${PICO_JSONL:-"$PROJECT_DIR/pico_banana/multi_turn_with_local_source_image_path.jsonl"}
PCA_OUTPUT_DIR=${PCA_OUTPUT_DIR:-"$PROJECT_DIR/results/pico_pca"}
TEXT_BATCH_SIZE=${TEXT_BATCH_SIZE:-2}
OUTPUT_JSON=${OUTPUT_JSON:-"$PROJECT_DIR/results/test_decoding_summary.json"}

# Toggle strategies (1 = enabled, 0 = disabled)
RUN_GREEDY=${RUN_GREEDY:-1}
RUN_NOISE_CONTRASTIVE=${RUN_NOISE_CONTRASTIVE:-1}
RUN_SIMPLE_ROTATION=${RUN_SIMPLE_ROTATION:-1}
RUN_PCA_TEXT=${RUN_PCA_TEXT:-1}
RUN_PCA_IMAGE=${RUN_PCA_IMAGE:-1}

export HF_HOME="$PROJECT_DIR/.cache"
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache"
LOG_DIR="$PROJECT_DIR/.log"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/test_decoding_${TIMESTAMP}.log"

CMD=(python "$PROJECT_DIR/test_decoding.py"
  --model-name "$MODEL_NAME"
  --max-samples "$MAX_SAMPLES"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --noise-scale "$NOISE_SCALE"
  --rotation-degrees "$ROTATION_DEGREES"
  --question-suffix "$QUESTION_SUFFIX"
  --pico-jsonl "$PICO_JSONL"
  --pca-output-dir "$PCA_OUTPUT_DIR"
  --text-batch-size "$TEXT_BATCH_SIZE"
  --output-json "$OUTPUT_JSON"
  --use-cache
)

[[ "$RUN_GREEDY" == "1" ]] && CMD+=(--run-greedy)
[[ "$RUN_NOISE_CONTRASTIVE" == "1" ]] && CMD+=(--run-noise-contrastive)
[[ "$RUN_SIMPLE_ROTATION" == "1" ]] && CMD+=(--run-simple-rotation)
[[ "$RUN_PCA_TEXT" == "1" ]] && CMD+=(--run-pca-text)
[[ "$RUN_PCA_IMAGE" == "1" ]] && CMD+=(--run-pca-image)

{
  echo "Logging to: $LOG_FILE"
  echo "Running test_decoding.py with:"
  echo "  GPU:               $GPU_ID"
  echo "  Model:             $MODEL_NAME"
  echo "  Max samples:       $MAX_SAMPLES"
  echo "  Max new tokens:    $MAX_NEW_TOKENS"
  echo "  Noise scale:       $NOISE_SCALE"
  echo "  Rotation degrees:  $ROTATION_DEGREES"
  echo "  Run greedy:        $RUN_GREEDY"
  echo "  Run contrastive:   $RUN_NOISE_CONTRASTIVE"
  echo "  Run rotation:      $RUN_SIMPLE_ROTATION"
  echo "  Run PCA text:      $RUN_PCA_TEXT"
  echo "  Run PCA image:     $RUN_PCA_IMAGE"
  echo "  Output JSON:       $OUTPUT_JSON"
  echo ""
} | tee "$LOG_FILE"

CUDA_VISIBLE_DEVICES=$GPU_ID "${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
