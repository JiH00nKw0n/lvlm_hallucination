#!/bin/bash

# Training script for LVLM with Accelerate & DeepSpeed
# Usage: ./scripts/train.sh [gpu_ids] [config_path] [wandb_key]
# Example: ./scripts/train.sh 0,1,2,3 config/train/llava_lrv_lora.yaml YOUR_WANDB_KEY
#          ./scripts/train.sh 0 config/train/llava_lrv_lora.yaml  # Single GPU, no wandb
# Note: DeepSpeed config should be specified in the training config YAML file
#       under trainer.deepspeed: "path/to/ds_config.json"

# Get script directory and project root
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
PARENT_DIR=$(dirname "$PROJECT_DIR")

# Set cache directories
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache"
export HF_HOME="$PROJECT_DIR/.cache"
export LOG_DIR="$PARENT_DIR/.log"

# Get GPU configuration (default to 0,1,2,3)
DEVICES=${1:-"0,1,2,3"}

# Get config path (default to example config)
CFG_PATH=${2:-"$PROJECT_DIR/config/train/llava_lrv_lora.yaml"}

# Get wandb key (optional)
WANDB_KEY=${3:-"3314a9f18c06914b9c333abc68130f93f2cb1a23"}

# Create log directory if it doesn't exist
mkdir -p "$PARENT_DIR/.log"

# Create log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PARENT_DIR/.log/train_${TIMESTAMP}.log"

echo "==========================================" | tee -a "$LOG_FILE"
echo "Starting LVLM Training Pipeline" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Project directory: $PROJECT_DIR" | tee -a "$LOG_FILE"
echo "Parent directory: $PARENT_DIR" | tee -a "$LOG_FILE"
echo "GPUs: $DEVICES" | tee -a "$LOG_FILE"
echo "Config: $CFG_PATH" | tee -a "$LOG_FILE"
if [ -n "$WANDB_KEY" ]; then
    echo "W&B logging: Enabled" | tee -a "$LOG_FILE"
else
    echo "W&B logging: Disabled" | tee -a "$LOG_FILE"
fi
echo "Log file: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Validate config file exists
if [ ! -f "$CFG_PATH" ]; then
    echo "Error: Configuration file not found: $CFG_PATH" | tee -a "$LOG_FILE"
    exit 1
fi

# Calculate number of GPUs from DEVICES
NUM_GPUS=$(echo "$DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')

echo "" | tee -a "$LOG_FILE"
echo "Number of GPUs: $NUM_GPUS" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Build command using accelerate launch
# DeepSpeed will be automatically used if specified in TrainingArguments (deepspeed parameter in config)
CMD="CUDA_VISIBLE_DEVICES=$DEVICES accelerate launch"

# Add accelerate arguments
CMD="$CMD --num_processes=$NUM_GPUS"

# Multi-GPU settings
if [ "$NUM_GPUS" -gt 1 ]; then
    CMD="$CMD --multi_gpu"
    echo "Using multi-GPU training" | tee -a "$LOG_FILE"
    echo "DeepSpeed will be used if 'deepspeed' is specified in trainer config" | tee -a "$LOG_FILE"
fi

# Add the training script
CMD="$CMD $PROJECT_DIR/train.py --cfg-path $CFG_PATH"

# Add wandb key if provided
if [ -n "$WANDB_KEY" ]; then
    CMD="$CMD --wandb-key $WANDB_KEY"
fi

echo "Command: $CMD" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Run training
if eval $CMD 2>&1 | tee -a "$LOG_FILE"; then
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Training completed successfully!" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
else
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Training failed!" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    exit 1
fi
