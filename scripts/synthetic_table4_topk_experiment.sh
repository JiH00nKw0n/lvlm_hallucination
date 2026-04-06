#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_DIR"

REPRESENTATION_DIM="${REPRESENTATION_DIM:-768}"
LATENT_SIZE="${LATENT_SIZE:-16384}"
FEATURE_DIMS_STR="${FEATURE_DIMS_STR:-800 1000 1200 1400}"
K_VALUES_STR="${K_VALUES_STR:-32 64 128}"

NUM_TRAIN_PAIRS="${NUM_TRAIN_PAIRS:-50000}"
NUM_EVAL_PAIRS="${NUM_EVAL_PAIRS:-10000}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
NUM_SEEDS="${NUM_SEEDS:-3}"
SEED_BASE="${SEED_BASE:-42}"

SPARSITY="${SPARSITY:-0.99}"
MIN_ACTIVE="${MIN_ACTIVE:-1}"
CMIN="${CMIN:-0.0}"
BETA="${BETA:-1.0}"
MAX_INTERFERENCE="${MAX_INTERFERENCE:-0.1}"
DICTIONARY_STRATEGY="${DICTIONARY_STRATEGY:-gradient}"

LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-auto}"
GT_THRESHOLD="${GT_THRESHOLD:-0.9}"
LOG_EVERY="${LOG_EVERY:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/synthetic_table4_topk}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --representation-dim) REPRESENTATION_DIM="$2"; shift 2 ;;
        --latent-size) LATENT_SIZE="$2"; shift 2 ;;
        --feature-dims) FEATURE_DIMS_STR="$2"; shift 2 ;;
        --k-values) K_VALUES_STR="$2"; shift 2 ;;
        --num-train-pairs) NUM_TRAIN_PAIRS="$2"; shift 2 ;;
        --num-eval-pairs) NUM_EVAL_PAIRS="$2"; shift 2 ;;
        --num-epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --num-seeds) NUM_SEEDS="$2"; shift 2 ;;
        --seed-base) SEED_BASE="$2"; shift 2 ;;
        --sparsity) SPARSITY="$2"; shift 2 ;;
        --min-active) MIN_ACTIVE="$2"; shift 2 ;;
        --cmin) CMIN="$2"; shift 2 ;;
        --beta) BETA="$2"; shift 2 ;;
        --max-interference) MAX_INTERFERENCE="$2"; shift 2 ;;
        --dictionary-strategy) DICTIONARY_STRATEGY="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --gt-threshold) GT_THRESHOLD="$2"; shift 2 ;;
        --log-every) LOG_EVERY="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$PROJECT_DIR/.log" "$OUTPUT_ROOT"

IFS=' ' read -r -a FEATURE_DIMS <<< "$FEATURE_DIMS_STR"
IFS=' ' read -r -a K_VALUES <<< "$K_VALUES_STR"

function run_one() {
    local feature_dim="$1"
    local k="$2"

    local ts
    ts=$(date +"%Y%m%d_%H%M%S")
    local run_tag
    run_tag="b${NUM_TRAIN_PAIRS}_${NUM_EVAL_PAIRS}_e${NUM_EPOCHS}_s${NUM_SEEDS}_${ts}"
    local log_file
    log_file="$PROJECT_DIR/.log/syn_table4_topk_fd${feature_dim}_k${k}_${ts}.log"

    echo "======================================================"
    echo "feature_dim=${feature_dim} | rep=${REPRESENTATION_DIM} | latent=${LATENT_SIZE} | k=${k}"
    echo "Budget: train=${NUM_TRAIN_PAIRS} eval=${NUM_EVAL_PAIRS} epochs=${NUM_EPOCHS} seeds=${NUM_SEEDS}"
    echo "Log: ${log_file}"
    echo "======================================================"

    python "$PROJECT_DIR/synthetic_table4_topk_experiment.py" \
        --feature-dim "$feature_dim" \
        --representation-dim "$REPRESENTATION_DIM" \
        --latent-size "$LATENT_SIZE" \
        --k "$k" \
        --sparsity "$SPARSITY" \
        --min-active "$MIN_ACTIVE" \
        --cmin "$CMIN" \
        --beta "$BETA" \
        --max-interference "$MAX_INTERFERENCE" \
        --dictionary-strategy "$DICTIONARY_STRATEGY" \
        --gt-recovery-threshold "$GT_THRESHOLD" \
        --num-train-pairs "$NUM_TRAIN_PAIRS" \
        --num-eval-pairs "$NUM_EVAL_PAIRS" \
        --num-epochs "$NUM_EPOCHS" \
        --num-seeds "$NUM_SEEDS" \
        --seed-base "$SEED_BASE" \
        --lr "$LR" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --log-every "$LOG_EVERY" \
        --output-root "$OUTPUT_ROOT" \
        --run-tag "$run_tag" \
        2>&1 | tee -a "$log_file"
}

for k in "${K_VALUES[@]}"; do
    for feature_dim in "${FEATURE_DIMS[@]}"; do
        run_one "$feature_dim" "$k"
    done
done

python - "$OUTPUT_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

from synthetic_table4_topk_experiment import summarize_trends

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
        "condition": m["condition"],
        "feature_dim": m["feature_dim"],
        "representation_dim": m["representation_dim"],
        "latent_size": m["latent_size"],
        "k": m["k"],
        "sparsity": m["sparsity"],
        "min_active": m["min_active"],
        "num_train_pairs": m["num_train_pairs"],
        "num_eval_pairs": m["num_eval_pairs"],
        "num_epochs": m["num_epochs"],
        "num_seeds": m["num_seeds"],
        "gt_recovery_threshold": m["gt_recovery_threshold"],
        "feature_to_rep_ratio": m["feature_to_rep_ratio"],
        "expected_active_gt": m["expected_active_gt"],
        "mgt_full_mean": a["mgt_full_mean"],
        "mgt_full_std": a["mgt_full_std"],
        "mip_full_mean": a["mip_full_mean"],
        "mip_full_std": a["mip_full_std"],
        "eval_recon_loss_mean": a["eval_recon_loss_mean"],
        "eval_recon_loss_std": a["eval_recon_loss_std"],
    })

rows.sort(key=lambda x: (int(x["k"]), int(x["feature_dim"])))

summary_csv = output_root / "summary_all.csv"
if rows:
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

trend_summary = summarize_trends(rows)
needs_recoverable_note = any(
    (not diag["mgt_non_increasing"]) or (not diag["mip_non_increasing"])
    for diag in trend_summary.values()
)

summary_md = output_root / "summary_all.md"
with open(summary_md, "w", encoding="utf-8") as f:
    f.write("# Synthetic Unimodal TopK Superposition Summary\n\n")
    f.write(f"- Total runs: {len(rows)}\n")
    f.write(f"- Source root: `{output_root}`\n")
    f.write("- Fixed setting: unimodal TopKSAE, tau=0.9, sparsity=0.99, min_active=1, latent_size=16384\n\n")
    if rows:
        headers = [
            "feature_dim",
            "k",
            "feature_to_rep_ratio",
            "expected_active_gt",
            "mgt_full_mean",
            "mip_full_mean",
            "eval_recon_loss_mean",
        ]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            vals = [str(r[h]) for h in headers]
            f.write("| " + " | ".join(vals) + " |\n")
        f.write("\n")

    f.write("## Acceptance Diagnostics\n\n")
    for k in sorted(trend_summary):
        diag = trend_summary[k]
        f.write(
            f"- k={k}: mgt_non_increasing={diag['mgt_non_increasing']}, "
            f"mip_non_increasing={diag['mip_non_increasing']}, "
            f"feature_dims={diag['feature_dims']}\n"
        )
    if needs_recoverable_note:
        f.write("\nTopK regime remains too recoverable under current min_active=1 setting\n")

print(f"Wrote: {summary_csv}")
print(f"Wrote: {summary_md}")
PY

echo "All experiments completed."
