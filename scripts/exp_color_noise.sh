#!/bin/bash

# Color noise experiment script
# Analyzes the impact of color noise on LVLM object recognition
# Usage: ./scripts/exp_color_noise.sh [gpu_id] [num_samples] [vlm_model] [sam_model]
# Example: ./scripts/exp_color_noise.sh 0 100                                      # Use GPU 0, 100 samples, default models
#          ./scripts/exp_color_noise.sh 0 10 llava-hf/llava-1.5-13b-hf            # Use 13B model
#          ./scripts/exp_color_noise.sh 0 100 llava-hf/llava-1.5-7b-hf sam-vit-large  # Use larger SAM

# Get script directory and project root
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")

# Set cache directories
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache"
export HF_HOME="$PROJECT_DIR/.cache"
export LOG_DIR="$PROJECT_DIR/.log"

# Get configuration
GPU_ID=${1:-"0"}  # Default to GPU 0
NUM_SAMPLES=${2:-"100"}  # Default to 100 samples
VLM_MODEL=${3:-"llava-hf/llava-1.5-7b-hf"}  # Default to LLaVA-1.5-7B
SAM_MODEL=${4:-"facebook/sam-vit-base"}  # Default to SAM base

# Create log directory if it doesn't exist
mkdir -p "$PROJECT_DIR/.log"

# Create log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_DIR/.log/exp_color_noise_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo "Project directory: $PROJECT_DIR"
echo "Using GPU: $GPU_ID"

echo "==========================================" | tee -a "$LOG_FILE"
echo "Starting Color Noise Experiment" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "GPU: $GPU_ID" | tee -a "$LOG_FILE"
echo "Number of samples: $NUM_SAMPLES" | tee -a "$LOG_FILE"
echo "VLM Model: $VLM_MODEL" | tee -a "$LOG_FILE"
echo "SAM Model: $SAM_MODEL" | tee -a "$LOG_FILE"
echo "Cache directory: $HF_DATASETS_CACHE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Run color noise analysis
echo "" | tee -a "$LOG_FILE"
echo "Running analyze_color_noise.py..." | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

if CUDA_VISIBLE_DEVICES=$GPU_ID python "$PROJECT_DIR/analyze_color_noise.py" \
    --num_samples $NUM_SAMPLES \
    --vlm_model "$VLM_MODEL" \
    --sam_model "$SAM_MODEL" \
    --image_dir "$PROJECT_DIR/images/images" \
    --output_dir "$PROJECT_DIR/analysis_color_noise" \
    --device "auto" 2>&1 | tee -a "$LOG_FILE"; then
    echo "✓ Color noise analysis completed successfully" | tee -a "$LOG_FILE"
else
    echo "✗ Color noise analysis failed" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Experiment failed!" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    exit 1
fi

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Experiment completed successfully!" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Results saved to:" | tee -a "$LOG_FILE"
echo "  - Charts: $PROJECT_DIR/analysis_color_noise/" | tee -a "$LOG_FILE"
echo "    - hue_range_effect.png" | tee -a "$LOG_FILE"
echo "    - blend_strength_effect.png" | tee -a "$LOG_FILE"
echo "    - results.pt" | tee -a "$LOG_FILE"
echo "  - Logs: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"