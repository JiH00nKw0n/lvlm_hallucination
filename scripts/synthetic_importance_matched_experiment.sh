#!/bin/bash
set -euo pipefail

# Same as synthetic_importance_experiment.sh but with LATENT_SIZE = FEATURE_DIM
# (SAE dictionary size matches ground-truth feature count)

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_DIR"

# ── Fixed experiment parameters ──────────────────────────────────────
FEATURE_DIM="${FEATURE_DIM:-1200}"
REPRESENTATION_DIM="${REPRESENTATION_DIM:-768}"
# KEY CHANGE: latent_size = feature_dim (no overcomplete dictionary)
LATENT_SIZE="${LATENT_SIZE:-$FEATURE_DIM}"
K="${K:-64}"
SPARSITY="${SPARSITY:-0.99}"

# ── Sweep variables ──────────────────────────────────────────────────
CONDITIONS_STR="${CONDITIONS_STR:-unimodal multimodal}"
IMPORTANCE_DECAYS_STR="${IMPORTANCE_DECAYS_STR:-1.0 0.999 0.997 0.995}"
IMPORTANCE_TARGETS_STR="${IMPORTANCE_TARGETS_STR:-shared image text}"

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
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/synthetic_importance_4}"

mkdir -p "$PROJECT_DIR/.log" "$OUTPUT_ROOT"

IFS=' ' read -r -a CONDITIONS <<< "$CONDITIONS_STR"
IFS=' ' read -r -a IMPORTANCE_DECAYS <<< "$IMPORTANCE_DECAYS_STR"
IFS=' ' read -r -a IMPORTANCE_TARGETS <<< "$IMPORTANCE_TARGETS_STR"

function run_one() {
    local condition="$1"
    local decay="$2"
    local target="$3"

    local ts
    ts=$(date +"%Y%m%d_%H%M%S")
    local run_tag
    run_tag="b${NUM_TRAIN_PAIRS}_e${NUM_EPOCHS}_s${NUM_SEEDS}_${ts}"
    local log_file
    log_file="$PROJECT_DIR/.log/syn_imp_matched_${condition}_d${decay}_t${target}_${ts}.log"

    echo "======================================================"
    echo "Condition=${condition} | decay=${decay} | target=${target}"
    echo "fd=${FEATURE_DIM} | rep=${REPRESENTATION_DIM} | latent=${LATENT_SIZE} (=fd) | k=${K}"
    echo "Log: ${log_file}"
    echo "======================================================"

    # Multimodal pairs produce 2x samples (img+txt), so use half the budget
    local train_pairs="$NUM_TRAIN_PAIRS"
    if [[ "$condition" == "multimodal" ]]; then
        train_pairs=$((NUM_TRAIN_PAIRS / 2))
    fi

    python "$PROJECT_DIR/synthetic_modality_experiment.py" \
        --condition "$condition" \
        --feature-dim "$FEATURE_DIM" \
        --representation-dim "$REPRESENTATION_DIM" \
        --latent-size "$LATENT_SIZE" \
        --k "$K" \
        --sparsity "$SPARSITY" \
        --min-active "$MIN_ACTIVE" \
        --cmin "$CMIN" \
        --beta "$BETA" \
        --max-interference "$MAX_INTERFERENCE" \
        --dictionary-strategy "$DICTIONARY_STRATEGY" \
        --vl-split-ratio "$VL_SPLIT_RATIO" \
        --gt-recovery-threshold "$GT_THRESHOLD" \
        --importance-decay "$decay" \
        --importance-target "$target" \
        --num-train-pairs "$train_pairs" \
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
    for decay in "${IMPORTANCE_DECAYS[@]}"; do
        if [[ "$condition" == "unimodal" ]]; then
            # Unimodal has no block distinction; run once with default target
            run_one "$condition" "$decay" "shared"
        else
            # Multimodal: sweep across importance targets
            # Skip target sweep when decay=1.0 (uniform => target is irrelevant)
            if [[ "$decay" == "1.0" ]]; then
                run_one "$condition" "$decay" "shared"
            else
                for target in "${IMPORTANCE_TARGETS[@]}"; do
                    run_one "$condition" "$decay" "$target"
                done
            fi
        fi
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
        "condition": m["condition"],
        "importance_decay": m.get("importance_decay", 1.0),
        "importance_target": m.get("importance_target", "shared"),
        "feature_dim": m["feature_dim"],
        "latent_size": m["latent_size"],
        "num_seeds": m["num_seeds"],
        "mgt_full_mean": a["mgt_full_mean"],
        "mgt_full_std": a["mgt_full_std"],
        "mip_full_mean": a["mip_full_mean"],
        "mip_full_std": a["mip_full_std"],
        "mgt_top10_mean": a.get("mgt_top10_mean", ""),
        "mgt_top10_std": a.get("mgt_top10_std", ""),
        "mip_top10_mean": a.get("mip_top10_mean", ""),
        "mip_top10_std": a.get("mip_top10_std", ""),
        "mgt_shared_mean": a.get("mgt_shared_mean", ""),
        "mip_shared_mean": a.get("mip_shared_mean", ""),
        "mgt_image_private_mean": a.get("mgt_image_private_mean", ""),
        "mgt_text_private_mean": a.get("mgt_text_private_mean", ""),
        "eval_recon_loss_mean": a["eval_recon_loss_mean"],
        "eval_recon_loss_std": a["eval_recon_loss_std"],
        "path": str(path),
    })

rows.sort(key=lambda x: (x["condition"], float(x["importance_decay"]), x["importance_target"]))

summary_csv = output_root / "summary_importance.csv"
if rows:
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

summary_md = output_root / "summary_importance.md"
with open(summary_md, "w", encoding="utf-8") as f:
    f.write("# Importance Hierarchy Experiment Summary (latent_size = feature_dim)\n\n")
    f.write(f"- Total runs: {len(rows)}\n\n")
    if rows:
        headers = [
            "condition", "importance_decay", "importance_target",
            "latent_size",
            "mgt_full_mean", "mip_full_mean",
            "mgt_top10_mean", "mip_top10_mean",
            "eval_recon_loss_mean",
        ]
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            vals = [str(r.get(h, "")) for h in headers]
            f.write("| " + " | ".join(vals) + " |\n")

print(f"Wrote: {summary_csv}")
print(f"Wrote: {summary_md}")
PY

echo "All importance experiments completed."