#!/bin/bash

# Patch-level monosemanticity analysis for SAE features — multi-model loop
# Usage: ./scripts/test_monosemanticity.sh [gpu_id]
# Example: ./scripts/test_monosemanticity.sh 0

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
    local extra_args="$2"
    local sae_name="${sae_path##*/}"
    local suffix=""
    if [[ "$extra_args" == *"--weighted"* ]]; then
        suffix="_weighted"
    fi
    local TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    local LOG_FILE="$PROJECT_DIR/.log/monosemanticity_${sae_name}${suffix}_${TIMESTAMP}.log"

    echo "=========================================="
    echo "SAE Feature Monosemanticity Analysis"
    echo "SAE: $sae_path"
    echo "Extra args: $extra_args"
    echo "GPU: $DEVICE"
    echo "Log: $LOG_FILE"
    echo "=========================================="

    python "$PROJECT_DIR/test_llava_monosemanticity.py" \
        --model_name llava-hf/llama3-llava-next-8b-hf \
        --sae_path "$sae_path" \
        --dinov2_name facebook/dinov2-large \
        --layer_index 24 \
        --num_samples 5000 \
        --k 256 \
        --top_patches 25 \
        --bin_width 0.05 \
        --seed 42 \
        --dtype float16 \
        --output_dir "$PROJECT_DIR/results/monosemanticity" \
        $extra_args \
        2>&1 | tee -a "$LOG_FILE"

    echo "=========================================="
    echo "Done: $sae_name $extra_args"
    echo "=========================================="
    echo ""
}

for sae_path in "${SAE_PATHS[@]}"; do
    run_experiment "$sae_path" ""           # unweighted
    run_experiment "$sae_path" "--weighted"  # weighted
done

echo "=========================================="
echo "All experiments done. Results in: $PROJECT_DIR/results/monosemanticity/"
echo "=========================================="
