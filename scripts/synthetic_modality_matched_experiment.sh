#!/bin/bash
set -euo pipefail

# Same as synthetic_modality_experiment.sh but with LATENT_SIZE = FEATURE_DIM
# (SAE dictionary size matches ground-truth feature count)

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_DIR"

# ── Fixed experiment parameters ──────────────────────────────────────
REPRESENTATION_DIM="${REPRESENTATION_DIM:-768}"
K="${K:-64}"
SPARSITY="${SPARSITY:-0.99}"

# ── Sweep variables ──────────────────────────────────────────────────
FEATURE_DIMS_STR="${FEATURE_DIMS_STR:-800 1000 1200 1400}"
CONDITIONS_STR="${CONDITIONS_STR:-unimodal multimodal}"

# ── Budget ───────────────────────────────────────────────────────────
NUM_TRAIN_PAIRS="${NUM_TRAIN_PAIRS:-50000}"
NUM_EVAL_PAIRS="${NUM_EVAL_PAIRS:-10000}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
NUM_SEEDS="${NUM_SEEDS:-3}"
SEED_BASE="${SEED_BASE:-42}"

# ── Data generation ──────────────────────────────────────────────────
MIN_ACTIVE="${MIN_ACTIVE:-1}"
CMIN="${CMIN:-0.0}"
BETA="${BETA:-1.0}"
MAX_INTERFERENCE="${MAX_INTERFERENCE:-0.1}"
DICTIONARY_STRATEGY="${DICTIONARY_STRATEGY:-gradient}"
VL_SPLIT_RATIO="${VL_SPLIT_RATIO:-1,2,1}"

# ── Training ─────────────────────────────────────────────────────────
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-256}"
DEVICE="${DEVICE:-auto}"
GT_THRESHOLD="${GT_THRESHOLD:-0.8}"
LOG_EVERY="${LOG_EVERY:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/synthetic_modality_4}"

mkdir -p "$PROJECT_DIR/.log" "$OUTPUT_ROOT"

IFS=' ' read -r -a FEATURE_DIMS <<< "$FEATURE_DIMS_STR"
IFS=' ' read -r -a CONDITIONS <<< "$CONDITIONS_STR"

function run_one() {
    local condition="$1"
    local feature_dim="$2"
    # KEY CHANGE: latent_size = feature_dim
    local latent_size="$feature_dim"

    local ts
    ts=$(date +"%Y%m%d_%H%M%S")
    local run_tag
    run_tag="b${NUM_TRAIN_PAIRS}_${NUM_EVAL_PAIRS}_e${NUM_EPOCHS}_s${NUM_SEEDS}_${ts}"
    local log_file
    log_file="$PROJECT_DIR/.log/syn_mod_matched_${condition}_fd${feature_dim}_${ts}.log"

    echo "======================================================"
    echo "Condition=${condition} | fd=${feature_dim} | rep=${REPRESENTATION_DIM} | latent=${latent_size} (=fd) | k=${K} | s=${SPARSITY}"
    echo "Budget: train=${NUM_TRAIN_PAIRS} eval=${NUM_EVAL_PAIRS} epochs=${NUM_EPOCHS} seeds=${NUM_SEEDS}"
    echo "Log: ${log_file}"
    echo "======================================================"

    python "$PROJECT_DIR/synthetic_modality_experiment.py" \
        --condition "$condition" \
        --feature-dim "$feature_dim" \
        --representation-dim "$REPRESENTATION_DIM" \
        --latent-size "$latent_size" \
        --k "$K" \
        --sparsity "$SPARSITY" \
        --min-active "$MIN_ACTIVE" \
        --cmin "$CMIN" \
        --beta "$BETA" \
        --max-interference "$MAX_INTERFERENCE" \
        --dictionary-strategy "$DICTIONARY_STRATEGY" \
        --vl-split-ratio "$VL_SPLIT_RATIO" \
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

# ── Run all combinations ─────────────────────────────────────────────
for condition in "${CONDITIONS[@]}"; do
    for feature_dim in "${FEATURE_DIMS[@]}"; do
        run_one "$condition" "$feature_dim"
    done
done

# ── Aggregate results ────────────────────────────────────────────────
python - "$OUTPUT_ROOT" <<'PY'
import csv
import json
import sys
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
        "condition": m["condition"],
        "feature_dim": m["feature_dim"],
        "representation_dim": m["representation_dim"],
        "latent_size": m["latent_size"],
        "k": m["k"],
        "sparsity": m["sparsity"],
        "num_train_pairs": m["num_train_pairs"],
        "num_eval_pairs": m["num_eval_pairs"],
        "num_epochs": m["num_epochs"],
        "num_seeds": m["num_seeds"],
        "mgt_full_mean": a["mgt_full_mean"],
        "mgt_full_std": a["mgt_full_std"],
        "mip_full_mean": a["mip_full_mean"],
        "mip_full_std": a["mip_full_std"],
        "mgt_shared_mean": a.get("mgt_shared_mean", ""),
        "mip_shared_mean": a.get("mip_shared_mean", ""),
        "mgt_image_private_mean": a.get("mgt_image_private_mean", ""),
        "mgt_text_private_mean": a.get("mgt_text_private_mean", ""),
        "eval_recon_loss_mean": a["eval_recon_loss_mean"],
        "eval_recon_loss_std": a["eval_recon_loss_std"],
    })

rows.sort(key=lambda x: (x["condition"], int(x["feature_dim"])))

summary_csv = output_root / "summary_all.csv"
if rows:
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

summary_md = output_root / "summary_all.md"
with open(summary_md, "w", encoding="utf-8") as f:
    f.write("# Synthetic Modality Comparison Summary (latent_size = feature_dim)\n\n")
    f.write(f"- Total runs: {len(rows)}\n")
    f.write(f"- Source root: `{output_root}`\n\n")
    if rows:
        headers = [
            "condition",
            "feature_dim",
            "latent_size",
            "mgt_full_mean",
            "mip_full_mean",
            "mgt_shared_mean",
            "mip_shared_mean",
            "eval_recon_loss_mean",
        ]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            vals = [str(r[h]) for h in headers]
            f.write("| " + " | ".join(vals) + " |\n")

# Pairwise comparison: unimodal vs multimodal at same feature_dim.
uni = {int(r["feature_dim"]): r for r in rows if r["condition"] == "unimodal"}
multi = {int(r["feature_dim"]): r for r in rows if r["condition"] == "multimodal"}

compare_rows = []
for fd in sorted(set(uni.keys()) & set(multi.keys())):
    u, mm = uni[fd], multi[fd]
    compare_rows.append({
        "feature_dim": fd,
        "mgt_full_uni": u["mgt_full_mean"],
        "mgt_full_multi": mm["mgt_full_mean"],
        "delta_mgt_full": float(mm["mgt_full_mean"]) - float(u["mgt_full_mean"]),
        "mip_full_uni": u["mip_full_mean"],
        "mip_full_multi": mm["mip_full_mean"],
        "delta_mip_full": float(mm["mip_full_mean"]) - float(u["mip_full_mean"]),
        "eval_recon_uni": u["eval_recon_loss_mean"],
        "eval_recon_multi": mm["eval_recon_loss_mean"],
    })

compare_csv = output_root / "summary_comparison.csv"
if compare_rows:
    with open(compare_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(compare_rows[0].keys()))
        writer.writeheader()
        writer.writerows(compare_rows)

compare_md = output_root / "summary_comparison.md"
with open(compare_md, "w", encoding="utf-8") as f:
    f.write("# Unimodal vs Multimodal Comparison (latent_size = feature_dim)\n\n")
    if compare_rows:
        headers = list(compare_rows[0].keys())
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in compare_rows:
            vals = [str(r[h]) for h in headers]
            f.write("| " + " | ".join(vals) + " |\n")

print(f"Wrote: {summary_csv}")
print(f"Wrote: {summary_md}")
if compare_rows:
    print(f"Wrote: {compare_csv}")
    print(f"Wrote: {compare_md}")
PY

echo "All experiments completed."