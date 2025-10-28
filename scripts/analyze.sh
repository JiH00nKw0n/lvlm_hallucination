#!/bin/bash

# Analysis pipeline script
# Runs inference, attention analysis, and decoding attention analysis sequentially
# Usage: ./scripts/analyze.sh [gpu_id] [num_samples]
# Example: ./scripts/analyze.sh 0 5    # Use GPU 0, process 5 samples
#          ./scripts/analyze.sh 0 188  # Use GPU 0, process all 188 samples
#          ./scripts/analyze.sh 0      # Use GPU 0, process 5 samples (default)

# Get script directory and project root
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")

# Set cache directories
export HF_DATASETS_CACHE="$PROJECT_DIR/.cache"
export HF_HOME="$PROJECT_DIR/.cache"
export LOG_DIR="$PROJECT_DIR/.log"

# Get GPU configuration and sample count
GPU_ID=${1:-"0"}  # Default to GPU 0
NUM_SAMPLES=${2:-"5"}  # Default to 5 samples

# Create log directory if it doesn't exist
mkdir -p "$PROJECT_DIR/.log"

# Create log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$PROJECT_DIR/.log/analyze_${TIMESTAMP}.log"

echo "Logging to: $LOG_FILE"
echo "Project directory: $PROJECT_DIR"
echo "Using GPU: $GPU_ID"

echo "==========================================" | tee -a "$LOG_FILE"
echo "Starting Analysis Pipeline" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "GPU: $GPU_ID" | tee -a "$LOG_FILE"
echo "Number of samples: $NUM_SAMPLES" | tee -a "$LOG_FILE"
echo "Cache directory: $HF_DATASETS_CACHE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

# Step 1: Run inference and save results
echo "" | tee -a "$LOG_FILE"
echo "[1/3] Running inference_and_save.py..." | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

if CUDA_VISIBLE_DEVICES=$GPU_ID python "$PROJECT_DIR/inference_and_save.py" --num_samples $NUM_SAMPLES 2>&1 | tee -a "$LOG_FILE"; then
    echo "✓ Step 1/3: Inference completed successfully" | tee -a "$LOG_FILE"
else
    echo "✗ Step 1/3: Inference failed" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Analysis pipeline failed at inference step!" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    exit 1
fi

# Step 2: Analyze attention patterns
echo "" | tee -a "$LOG_FILE"
echo "[2/3] Running analyze_attention.py..." | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

if CUDA_VISIBLE_DEVICES=$GPU_ID python "$PROJECT_DIR/analyze_attention.py" 2>&1 | tee -a "$LOG_FILE"; then
    echo "✓ Step 2/3: Attention analysis completed successfully" | tee -a "$LOG_FILE"
else
    echo "✗ Step 2/3: Attention analysis failed" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Analysis pipeline failed at attention analysis step!" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    exit 1
fi

# Step 3: Analyze decoding attention
echo "" | tee -a "$LOG_FILE"
echo "[3/3] Running analyze_decoding_attention.py..." | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"

if CUDA_VISIBLE_DEVICES=$GPU_ID python "$PROJECT_DIR/analyze_decoding_attention.py" 2>&1 | tee -a "$LOG_FILE"; then
    echo "✓ Step 3/3: Decoding attention analysis completed successfully" | tee -a "$LOG_FILE"
else
    echo "✗ Step 3/3: Decoding attention analysis failed" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "Analysis pipeline failed at decoding attention step!" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    exit 1
fi

echo "" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "All analysis steps completed successfully!" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
echo "Results saved to:" | tee -a "$LOG_FILE"
echo "  - Inference: $PROJECT_DIR/inference_results/" | tee -a "$LOG_FILE"
echo "  - Analysis: $PROJECT_DIR/analysis/" | tee -a "$LOG_FILE"
echo "  - Logs: $LOG_FILE" | tee -a "$LOG_FILE"
echo "==========================================" | tee -a "$LOG_FILE"
