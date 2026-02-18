#!/bin/bash

# CLIP modality split analysis for SAE features — multi-model loop
# Usage: ./scripts/test_clip_modality_split.sh [gpu_id]
# Example: ./scripts/test_clip_modality_split.sh 0

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
    "Mayfull/CLIP_TopKSAE"
    "Mayfull/CLIP_TopKSAE_GS"
    "Mayfull/CLIP_VLTopKSAE_6_4_6"
    "Mayfull/CLIP_VLTopKSAE_4_8_4"
    "Mayfull/CLIP_VLTopKSAE_2_12_2"
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
    local LOG_FILE="$PROJECT_DIR/.log/clip_modality_split_${sae_name}${suffix}_${TIMESTAMP}.log"

    echo "=========================================="
    echo "CLIP SAE Feature Modality Split Analysis"
    echo "SAE: $sae_path"
    echo "Extra args: $extra_args"
    echo "GPU: $DEVICE"
    echo "Log: $LOG_FILE"
    echo "=========================================="

    python "$PROJECT_DIR/test_clip_modality_split.py" \
        --clip_model_name openai/clip-vit-large-patch14 \
        --sae_path "$sae_path" \
        --k 32 \
        --num_samples 5000 \
        --bin_width 0.05 \
        --seed 42 \
        --dtype float32 \
        --output_dir "$PROJECT_DIR/results/clip_modality_split" \
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
echo "All experiments done. Results in: $PROJECT_DIR/results/clip_modality_split/"
echo "=========================================="
