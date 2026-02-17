#!/bin/bash

# Loss Recovery Score analysis for SAE reconstruction — multi-model loop
# Usage: ./scripts/test_loss_recovery.sh [gpu_id]
# Example: ./scripts/test_loss_recovery.sh 0

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

# SAE model paths to evaluate
SAE_PATHS=(
    "lmms-lab/llama3-llava-next-8b-hf-sae-131k"
    "Mayfull/LLaVA_Next_VLTopKSAE"
    "Mayfull/LLaVA_Next_VLTopKSAE_SL"
    "Mayfull/LLaVA_Next_VLBatchTopKSAE"
    "Mayfull/LLaVA_Next_VLBatchTopKSAE_SL"
    "Mayfull/LLaVA_Next_VLMatryoshkaSAE"
    "Mayfull/LLaVA_Next_VLMatryoshkaSAE_SL"
    "Mayfull/LLaVA_Next_MatryoshkaSAE"
    "Mayfull/LLaVA_Next_BatchTopKSAE"
)

run_experiment() {
    local sae_path="$1"
    local sae_name="${sae_path##*/}"
    local TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    local LOG_FILE="$PROJECT_DIR/.log/loss_recovery_${sae_name}_${TIMESTAMP}.log"

    echo "=========================================="
    echo "SAE Loss Recovery Score Analysis"
    echo "SAE: $sae_path"
    echo "GPU: $DEVICE"
    echo "Log: $LOG_FILE"
    echo "=========================================="

    python "$PROJECT_DIR/test_loss_recovery.py" \
        --model_name llava-hf/llama3-llava-next-8b-hf \
        --sae_path "$sae_path" \
        --layer_index 24 \
        --num_samples 90 \
        --k 256 \
        --seed 42 \
        --dtype float16 \
        --dataset_name lmms-lab/llava-bench-coco \
        --output_dir "$PROJECT_DIR/results/loss_recovery" \
        2>&1 | tee -a "$LOG_FILE"

    echo "=========================================="
    echo "Done: $sae_name"
    echo "=========================================="
    echo ""
}

for sae_path in "${SAE_PATHS[@]}"; do
    run_experiment "$sae_path"
done

echo "=========================================="
echo "All experiments done. Results in: $PROJECT_DIR/results/loss_recovery/"
echo "=========================================="
