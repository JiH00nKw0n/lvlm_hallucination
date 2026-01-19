#!/bin/bash

# Evaluation script for LVLM benchmarks
# Usage: ./scripts/eval.sh [gpu_ids] [cfg_path]
# Example: ./scripts/eval.sh 0,1,2,3  # Use GPUs 0,1,2,3
#          ./scripts/eval.sh 0        # Use GPU 0 only
#          ./scripts/eval.sh 0 config/evaluate/llama3-llava-next-8b-hf.yaml
# Note: device_map="auto" in config automatically distributes model across available GPUs

# Get script directory and project root
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
PARENT_DIR=$(dirname "$PROJECT_DIR")

# Set cache directories
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache"
export HF_HOME="$PROJECT_DIR/.cache"
export LOG_DIR="$PROJECT_DIR/.log"

# Get GPU configuration
DEVICES=${1:-"0,1,2,3"}  # Default to GPUs 0,1,2,3
CFG_OVERRIDE=${2:-""}

# Create log directory if it doesn't exist
mkdir -p "$PROJECT_DIR/.log"

# Create log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_DIR/.log/eval_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo "Project directory: $PROJECT_DIR"
echo "Parent directory: $PARENT_DIR"

# Configuration files to evaluate
if [ -n "$CFG_OVERRIDE" ]; then
    if [[ "$CFG_OVERRIDE" = /* ]]; then
        CFG_PATHS=("$CFG_OVERRIDE")
    else
        CFG_PATHS=("$PROJECT_DIR/$CFG_OVERRIDE")
    fi
else
    CFG_PATHS=(
        "$PROJECT_DIR/config/evaluate/llama3-llava-next-8b-hf.yaml"
        # Add more config paths here as needed
    )
fi

echo "==========================================" | tee -a "$LOG_FILE"
echo "Starting LVLM Evaluation Pipeline" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "GPUs: $DEVICES" | tee -a "$LOG_FILE"
echo "Total configs: ${#CFG_PATHS[@]}" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Loop through each config file
for i in "${!CFG_PATHS[@]}"
do
    CFG_PATH="${CFG_PATHS[$i]}"

    echo "" | tee -a "$LOG_FILE"
    echo "[$((i+1))/${#CFG_PATHS[@]}] Currently Running with Config: $CFG_PATH" | tee -a "$LOG_FILE"

    # Validate config file exists
    if [ ! -f "$CFG_PATH" ]; then
        echo "Error: Configuration file not found: $CFG_PATH" | tee -a "$LOG_FILE"
        echo "Skipping..." | tee -a "$LOG_FILE"
        continue
    fi

    echo "==========================================" | tee -a "$LOG_FILE"

    # Calculate number of GPUs from DEVICES
    NUM_GPUS=$(echo "$DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')

    # Run evaluation with Accelerate for distributed inference
    # Redirect both stdout and stderr to log file while showing on console
    if [ "$NUM_GPUS" -ge 2 ]; then
        ACCELERATE_ARGS=(--num_processes="$NUM_GPUS" --multi_gpu)
    else
        ACCELERATE_ARGS=(--num_processes="$NUM_GPUS")
    fi

    if CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch \
        "${ACCELERATE_ARGS[@]}" \
        "$PROJECT_DIR/evaluate.py" --cfg-path "$CFG_PATH" 2>&1 | tee -a "$LOG_FILE"; then
        echo "✓ Config $((i+1))/${#CFG_PATHS[@]} completed successfully" | tee -a "$LOG_FILE"
    else
        echo "✗ Config $((i+1))/${#CFG_PATHS[@]} failed" | tee -a "$LOG_FILE"
        echo "" | tee -a "$LOG_FILE"
        echo "==========================================" | tee -a "$LOG_FILE"
        echo "Evaluation pipeline failed!" | tee -a "$LOG_FILE"
        echo "==========================================" | tee -a "$LOG_FILE"
        exit 1
    fi

    echo "==========================================" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "All evaluations completed successfully!" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
