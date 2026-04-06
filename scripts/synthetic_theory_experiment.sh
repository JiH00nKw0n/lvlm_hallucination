#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_DIR"

# ── Fixed experiment parameters ──────────────────────────────────────
REPRESENTATION_DIM="${REPRESENTATION_DIM:-768}"
FEATURE_DIM="${FEATURE_DIM:-1024}"        # n_I + n_S + n_T = 1024
LATENT_SIZE="${LATENT_SIZE:-8192}"
K="${K:-64}"
SPARSITY="${SPARSITY:-0.99}"
MAX_INTERFERENCE="${MAX_INTERFERENCE:-0.1}"
DICTIONARY_STRATEGY="${DICTIONARY_STRATEGY:-gradient}"

# ── Budget ───────────────────────────────────────────────────────────
NUM_TRAIN="${NUM_TRAIN:-50000}"
NUM_EVAL="${NUM_EVAL:-10000}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
NUM_SEEDS="${NUM_SEEDS:-3}"
SEED_BASE="${SEED_BASE:-42}"

# ── Training ─────────────────────────────────────────────────────────
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-auto}"
GT_THRESHOLD="${GT_THRESHOLD:-0.8}"
LOG_EVERY="${LOG_EVERY:-50}"

# ── Sweep: n_private values (n_I = n_T each) ────────────────────────
# n_shared = FEATURE_DIM - 2*n_private
N_PRIVATE_VALUES="${N_PRIVATE_VALUES:-0 64 128 256 384 448}"

# ── Sweep: alpha targets (Exp 1ii) ──────────────────────────────────
ALPHA_VALUES="${ALPHA_VALUES:-1.0-1.0,0.7-0.9,0.5-0.7,0.3-0.5,0.1-0.3}"

# ── Alpha calibration ───────────────────────────────────────────────
CALIBRATION_LR="${CALIBRATION_LR:-0.005}"
CALIBRATION_MAX_ITERS="${CALIBRATION_MAX_ITERS:-2000}"
CALIBRATION_TOL="${CALIBRATION_TOL:-0.005}"

# ── Experiments to run ──────────────────────────────────────────────
EXPERIMENTS="${EXPERIMENTS:-1i,1ii,2i,2ii}"

# ── Output ──────────────────────────────────────────────────────────
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/synthetic_theory}"

# ── CLI overrides ────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --representation-dim) REPRESENTATION_DIM="$2"; shift 2 ;;
        --feature-dim) FEATURE_DIM="$2"; shift 2 ;;
        --latent-size) LATENT_SIZE="$2"; shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --sparsity) SPARSITY="$2"; shift 2 ;;
        --max-interference) MAX_INTERFERENCE="$2"; shift 2 ;;
        --dictionary-strategy) DICTIONARY_STRATEGY="$2"; shift 2 ;;
        --num-train) NUM_TRAIN="$2"; shift 2 ;;
        --num-eval) NUM_EVAL="$2"; shift 2 ;;
        --num-epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --num-seeds) NUM_SEEDS="$2"; shift 2 ;;
        --seed-base) SEED_BASE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --gt-threshold) GT_THRESHOLD="$2"; shift 2 ;;
        --log-every) LOG_EVERY="$2"; shift 2 ;;
        --n-private-values) N_PRIVATE_VALUES="$2"; shift 2 ;;
        --alpha-values) ALPHA_VALUES="$2"; shift 2 ;;
        --experiments) EXPERIMENTS="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$PROJECT_DIR/.log" "$OUTPUT_ROOT"

# Convert space-separated to comma-separated for python CLI
N_PRIV_CSV=$(echo "$N_PRIVATE_VALUES" | tr ' ' ',')
# Alpha values are already comma-separated with range syntax (e.g., "0.7-0.9,0.5-0.7")
ALPHA_CSV="$ALPHA_VALUES"

function run_exp() {
    local exp="$1"
    local n_shared_override="$2"  # 0 = compute from sweep

    local ts
    ts=$(date +"%Y%m%d_%H%M%S")
    local run_tag
    run_tag="fd${FEATURE_DIM}_lat${LATENT_SIZE}_b${NUM_TRAIN}_e${NUM_EPOCHS}_s${NUM_SEEDS}_${ts}"
    local log_file
    log_file="$PROJECT_DIR/.log/syn_theory_exp${exp}_${ts}.log"

    echo "======================================================"
    echo "Experiment=${exp} | feature_dim=${FEATURE_DIM} | latent=${LATENT_SIZE} | rep=${REPRESENTATION_DIM} | k=${K}"
    echo "Budget: train=${NUM_TRAIN} eval=${NUM_EVAL} epochs=${NUM_EPOCHS} seeds=${NUM_SEEDS}"
    echo "Log: ${log_file}"
    echo "======================================================"

    # n_shared_arg = total feature_dim for python (it computes n_shared_actual = n_shared - 2*n_private)
    local n_shared_arg="$FEATURE_DIM"
    if [[ "$n_shared_override" -gt 0 ]]; then
        n_shared_arg="$n_shared_override"
    fi

    python "$PROJECT_DIR/synthetic_sae_theory_experiment.py" \
        --experiment "$exp" \
        --n-private-values "$N_PRIV_CSV" \
        --alpha-values "$ALPHA_CSV" \
        --n-shared "$n_shared_arg" \
        --representation-dim "$REPRESENTATION_DIM" \
        --latent-size "$LATENT_SIZE" \
        --k "$K" \
        --gt-recovery-threshold "$GT_THRESHOLD" \
        --num-train "$NUM_TRAIN" \
        --num-eval "$NUM_EVAL" \
        --num-epochs "$NUM_EPOCHS" \
        --num-seeds "$NUM_SEEDS" \
        --seed-base "$SEED_BASE" \
        --sparsity "$SPARSITY" \
        --max-interference "$MAX_INTERFERENCE" \
        --dictionary-strategy "$DICTIONARY_STRATEGY" \
        --calibration-lr "$CALIBRATION_LR" \
        --calibration-max-iters "$CALIBRATION_MAX_ITERS" \
        --calibration-tol "$CALIBRATION_TOL" \
        --lr "$LR" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --log-every "$LOG_EVERY" \
        --output-root "$OUTPUT_ROOT" \
        --run-tag "$run_tag" \
        --verbose \
        2>&1 | tee -a "$log_file"
}

# ── Run requested experiments ───────────────────────────────────────
# Exp 1i/2i: n_shared = FEATURE_DIM - 2*n_private (computed per sweep point in python)
# Exp 1ii/2ii: n_shared = FEATURE_DIM (no private features)

# Use comma-delimited matching to avoid "1i" matching "1ii"
IFS=',' read -ra EXP_ARRAY <<< "$EXPERIMENTS"
for _exp in "${EXP_ARRAY[@]}"; do
    case "$_exp" in
        1i)
            echo ">>> Running Exp 1(i): Single SAE -- shared-private interference"
            run_exp "1i" 0
            ;;
        1ii)
            echo ">>> Running Exp 1(ii): Single SAE -- generative mapping mismatch"
            run_exp "1ii" "$FEATURE_DIM"
            ;;
        2i)
            echo ">>> Running Exp 2(i): Two SAEs -- shared-private entanglement"
            run_exp "2i" 0
            ;;
        2ii)
            echo ">>> Running Exp 2(ii): Two SAEs -- latent non-identifiability"
            run_exp "2ii" "$FEATURE_DIM"
            ;;
        *)
            echo "Unknown experiment: $_exp"
            exit 1
            ;;
    esac
done

echo "All requested experiments completed."
