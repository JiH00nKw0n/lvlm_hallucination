#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_DIR"

# Default budget (override with flags or env vars).
NUM_TRAIN_PAIRS="${NUM_TRAIN_PAIRS:-50000}"
NUM_EVAL_PAIRS="${NUM_EVAL_PAIRS:-10000}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
NUM_SEEDS="${NUM_SEEDS:-3}"
SEED_BASE="${SEED_BASE:-42}"

# Core defaults.
FEATURE_DIMS_STR="${FEATURE_DIMS_STR:-128 192 256 320 384}"
LAMBDAS_STR="${LAMBDAS_STR:-0.01 0.05 0.1 0.5 1.0 5.0 10.0}"
REPRESENTATION_DIM="${REPRESENTATION_DIM:-128}"
K="${K:-2}"
BLOCK_TOP_K="${BLOCK_TOP_K:-}" # e.g. 1,1 => allocate k by [modality_specific,shared] ratio
VL_SPLIT_RATIO="${VL_SPLIT_RATIO:-1,2,1}"
GT_THRESHOLD="${GT_THRESHOLD:-0.8}"
LATENT_SIZE="${LATENT_SIZE:--1}"
ALIGN_EXPERIMENTS="${ALIGN_EXPERIMENTS:-1,2,3}" # comma-separated subset, e.g. 1,3

LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-auto}"
SPARSITY_SHARED="${SPARSITY_SHARED:-0.999}"
SPARSITY_IMAGE="${SPARSITY_IMAGE:-0.999}"
SPARSITY_TEXT="${SPARSITY_TEXT:-0.999}"
MIN_ACTIVE_SHARED="${MIN_ACTIVE_SHARED:-1}"
MIN_ACTIVE_IMAGE="${MIN_ACTIVE_IMAGE:-1}"
MIN_ACTIVE_TEXT="${MIN_ACTIVE_TEXT:-1}"
MAX_INTERFERENCE="${MAX_INTERFERENCE:-0.3}"
DICTIONARY_STRATEGY="${DICTIONARY_STRATEGY:-gradient}"
BLOCK_ORTHOGONALITY="${BLOCK_ORTHOGONALITY:-true}" # true|false
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/synthetic_multimodal}"
LOG_EVERY="${LOG_EVERY:-0}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --num-train-pairs) NUM_TRAIN_PAIRS="$2"; shift 2 ;;
        --num-eval-pairs) NUM_EVAL_PAIRS="$2"; shift 2 ;;
        --num-epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --num-seeds) NUM_SEEDS="$2"; shift 2 ;;
        --seed-base) SEED_BASE="$2"; shift 2 ;;
        --feature-dims) FEATURE_DIMS_STR="$2"; shift 2 ;;
        --lambdas) LAMBDAS_STR="$2"; shift 2 ;;
        --representation-dim) REPRESENTATION_DIM="$2"; shift 2 ;;
        --k) K="$2"; shift 2 ;;
        --block-top-k) BLOCK_TOP_K="$2"; shift 2 ;;
        --vl-split-ratio) VL_SPLIT_RATIO="$2"; shift 2 ;;
        --gt-threshold) GT_THRESHOLD="$2"; shift 2 ;;
        --latent-size) LATENT_SIZE="$2"; shift 2 ;;
        --experiments) ALIGN_EXPERIMENTS="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        --sparsity-shared) SPARSITY_SHARED="$2"; shift 2 ;;
        --sparsity-image) SPARSITY_IMAGE="$2"; shift 2 ;;
        --sparsity-text) SPARSITY_TEXT="$2"; shift 2 ;;
        --min-active-shared) MIN_ACTIVE_SHARED="$2"; shift 2 ;;
        --min-active-image) MIN_ACTIVE_IMAGE="$2"; shift 2 ;;
        --min-active-text) MIN_ACTIVE_TEXT="$2"; shift 2 ;;
        --max-interference) MAX_INTERFERENCE="$2"; shift 2 ;;
        --dictionary-strategy) DICTIONARY_STRATEGY="$2"; shift 2 ;;
        --block-orthogonality) BLOCK_ORTHOGONALITY="$2"; shift 2 ;;
        --log-every) LOG_EVERY="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$PROJECT_DIR/.log" "$OUTPUT_ROOT"

IFS=' ' read -r -a FEATURE_DIMS <<< "$FEATURE_DIMS_STR"
IFS=' ' read -r -a LAMBDAS <<< "$LAMBDAS_STR"

function run_one() {
    local experiment="$1"
    local feature_dim="$2"
    local alignment_mode="$3"
    local lambda_align="$4"

    local ts
    ts=$(date +"%Y%m%d_%H%M%S")
    local run_tag
    run_tag="b${NUM_TRAIN_PAIRS}_${NUM_EVAL_PAIRS}_e${NUM_EPOCHS}_s${NUM_SEEDS}_${ts}"
    local log_file
    log_file="$PROJECT_DIR/.log/syn_mm_exp${experiment}_fd${feature_dim}_${alignment_mode}_lam${lambda_align}_${ts}.log"

    echo "======================================================"
    echo "Exp=${experiment} | fd=${feature_dim} | mode=${alignment_mode} | lambda=${lambda_align}"
    echo "Budget: train=${NUM_TRAIN_PAIRS} eval=${NUM_EVAL_PAIRS} epochs=${NUM_EPOCHS} seeds=${NUM_SEEDS}"
    echo "Log: ${log_file}"
    echo "======================================================"

    local ORTHO_FLAG=""
    if [[ "$BLOCK_ORTHOGONALITY" == "false" ]]; then
        ORTHO_FLAG="--disable-block-orthogonality"
    fi
    local BLOCK_TOP_K_ARGS=()
    if [[ -n "$BLOCK_TOP_K" ]]; then
        BLOCK_TOP_K_ARGS=(--block-top-k "$BLOCK_TOP_K")
    fi

    python "$PROJECT_DIR/synthetic_multimodal_experiment.py" \
        --experiment "$experiment" \
        --alignment-mode "$alignment_mode" \
        --lambda-align "$lambda_align" \
        --feature-dim "$feature_dim" \
        --representation-dim "$REPRESENTATION_DIM" \
        --latent-size "$LATENT_SIZE" \
        --k "$K" \
        "${BLOCK_TOP_K_ARGS[@]}" \
        --vl-split-ratio "$VL_SPLIT_RATIO" \
        --gt-recovery-threshold "$GT_THRESHOLD" \
        --num-train-pairs "$NUM_TRAIN_PAIRS" \
        --num-eval-pairs "$NUM_EVAL_PAIRS" \
        --num-epochs "$NUM_EPOCHS" \
        --num-seeds "$NUM_SEEDS" \
        --seed-base "$SEED_BASE" \
        --sparsity-shared "$SPARSITY_SHARED" \
        --sparsity-image "$SPARSITY_IMAGE" \
        --sparsity-text "$SPARSITY_TEXT" \
        --min-active-shared "$MIN_ACTIVE_SHARED" \
        --min-active-image "$MIN_ACTIVE_IMAGE" \
        --min-active-text "$MIN_ACTIVE_TEXT" \
        --max-interference "$MAX_INTERFERENCE" \
        --dictionary-strategy "$DICTIONARY_STRATEGY" \
        ${ORTHO_FLAG:+$ORTHO_FLAG} \
        --lr "$LR" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --log-every "$LOG_EVERY" \
        --output-root "$OUTPUT_ROOT" \
        --run-tag "$run_tag" \
        2>&1 | tee -a "$log_file"
}

if [[ "$ALIGN_EXPERIMENTS" == *"1"* ]]; then
    for feature_dim in "${FEATURE_DIMS[@]}"; do
        run_one 1 "$feature_dim" "none" "0"
    done
fi

if [[ "$ALIGN_EXPERIMENTS" == *"2"* ]]; then
    for feature_dim in "${FEATURE_DIMS[@]}"; do
        for lam in "${LAMBDAS[@]}"; do
            run_one 2 "$feature_dim" "shared_topk_l2" "$lam"
        done
    done
fi

if [[ "$ALIGN_EXPERIMENTS" == *"3"* ]]; then
    for feature_dim in "${FEATURE_DIMS[@]}"; do
        for lam in "${LAMBDAS[@]}"; do
            run_one 3 "$feature_dim" "all_topk_l2" "$lam"
            run_one 3 "$feature_dim" "shared_topk_l2" "$lam"
        done
    done
fi

python - "$OUTPUT_ROOT" <<'PY'
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

output_root = Path(sys.argv[1])
run_files = sorted(output_root.glob("runs/**/result.json"))

rows = []
for path in run_files:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    m = payload["metadata"]
    a = payload["aggregate"]
    rows.append({
        "path": str(path),
        "experiment": m["experiment"],
        "alignment_mode": m["alignment_mode"],
        "lambda_align": m["lambda_align"],
        "feature_dim": m["feature_dim"],
        "latent_size": m["latent_size"],
        "k": m["k"],
        "block_top_k": m.get("block_top_k"),
        "num_train_pairs": m["num_train_pairs"],
        "num_eval_pairs": m["num_eval_pairs"],
        "num_epochs": m["num_epochs"],
        "num_seeds": m["num_seeds"],
        "mgt_shared_mean": a["mgt_shared_mean"],
        "mgt_shared_std": a["mgt_shared_std"],
        "mip_shared_mean": a["mip_shared_mean"],
        "mip_shared_std": a["mip_shared_std"],
        "mgt_full_mean": a["mgt_full_mean"],
        "mgt_full_std": a["mgt_full_std"],
        "mip_full_mean": a["mip_full_mean"],
        "mip_full_std": a["mip_full_std"],
        "eval_total_loss_mean": a["eval_total_loss_mean"],
        "eval_total_loss_std": a["eval_total_loss_std"],
    })

rows.sort(key=lambda x: (x["experiment"], x["feature_dim"], x["alignment_mode"], float(x["lambda_align"])))

summary_csv = output_root / "summary_all.csv"
if rows:
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

summary_md = output_root / "summary_all.md"
with open(summary_md, "w", encoding="utf-8") as f:
    f.write("# Synthetic Multimodal SAE Summary\n\n")
    f.write(f"- Total runs: {len(rows)}\n")
    f.write(f"- Source root: `{output_root}`\n\n")
    if rows:
        headers = [
            "experiment",
            "feature_dim",
            "alignment_mode",
            "lambda_align",
            "mgt_shared_mean",
            "mip_shared_mean",
            "mgt_full_mean",
            "mip_full_mean",
            "eval_total_loss_mean",
        ]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            vals = [str(r[h]) for h in headers]
            f.write("| " + " | ".join(vals) + " |\n")

# Exp3 comparison: all_topk_l2 vs shared_topk_l2 at same feature_dim/lambda.
grouped = defaultdict(dict)
for r in rows:
    if int(r["experiment"]) != 3:
        continue
    key = (int(r["feature_dim"]), float(r["lambda_align"]))
    grouped[key][r["alignment_mode"]] = r

compare_rows = []
for (fd, lam), sub in sorted(grouped.items()):
    if "all_topk_l2" not in sub or "shared_topk_l2" not in sub:
        continue
    all_row = sub["all_topk_l2"]
    shared_row = sub["shared_topk_l2"]
    compare_rows.append({
        "feature_dim": fd,
        "lambda_align": lam,
        "mgt_shared_all": all_row["mgt_shared_mean"],
        "mgt_shared_shared": shared_row["mgt_shared_mean"],
        "delta_mgt_shared_shared_minus_all": shared_row["mgt_shared_mean"] - all_row["mgt_shared_mean"],
        "mip_shared_all": all_row["mip_shared_mean"],
        "mip_shared_shared": shared_row["mip_shared_mean"],
        "delta_mip_shared_shared_minus_all": shared_row["mip_shared_mean"] - all_row["mip_shared_mean"],
        "eval_total_all": all_row["eval_total_loss_mean"],
        "eval_total_shared": shared_row["eval_total_loss_mean"],
    })

compare_csv = output_root / "summary_exp3_comparison.csv"
if compare_rows:
    with open(compare_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(compare_rows[0].keys()))
        writer.writeheader()
        writer.writerows(compare_rows)

compare_md = output_root / "summary_exp3_comparison.md"
with open(compare_md, "w", encoding="utf-8") as f:
    f.write("# Exp3 Comparison (all_topk_l2 vs shared_topk_l2)\n\n")
    if compare_rows:
        headers = list(compare_rows[0].keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in compare_rows:
            vals = [str(r[h]) for h in headers]
            f.write("| " + " | ".join(vals) + " |\n")

print(f"Wrote: {summary_csv}")
print(f"Wrote: {summary_md}")
print(f"Wrote: {compare_csv}")
print(f"Wrote: {compare_md}")
PY

echo "All experiments completed."
