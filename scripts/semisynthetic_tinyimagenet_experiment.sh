#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_DIR"

# ---- Defaults ---- #
LATENT_SIZE="${LATENT_SIZE:-4096}"
K_VALUES_STR="${K_VALUES_STR:-16 32 64}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
NUM_SEEDS="${NUM_SEEDS:-3}"
SEED_BASE="${SEED_BASE:-42}"
LR="${LR:-2e-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DEVICE="${DEVICE:-auto}"
GT_THRESHOLD="${GT_THRESHOLD:-0.8}"
CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-256}"
FEATURE_DIR="${FEATURE_DIR:-$PROJECT_DIR/outputs/tinyimagenet_features}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/outputs/semisynthetic_tinyimagenet}"

# ---- CLI override ---- #
while [[ $# -gt 0 ]]; do
    case "$1" in
        --latent-size) LATENT_SIZE="$2"; shift 2 ;;
        --k-values) K_VALUES_STR="$2"; shift 2 ;;
        --num-epochs) NUM_EPOCHS="$2"; shift 2 ;;
        --num-seeds) NUM_SEEDS="$2"; shift 2 ;;
        --seed-base) SEED_BASE="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --gt-threshold) GT_THRESHOLD="$2"; shift 2 ;;
        --clip-batch-size) CLIP_BATCH_SIZE="$2"; shift 2 ;;
        --feature-dir) FEATURE_DIR="$2"; shift 2 ;;
        --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

mkdir -p "$PROJECT_DIR/.log" "$OUTPUT_ROOT" "$FEATURE_DIR"

IFS=' ' read -r -a K_VALUES <<< "$K_VALUES_STR"

# ---- Step 1: Feature extraction (skips if already done) ---- #
echo "======================================================"
echo "Step 1: Extract CLIP embeddings + GT features"
echo "  feature_dir=${FEATURE_DIR}"
echo "======================================================"

ts=$(date +"%Y%m%d_%H%M%S")
log_file="$PROJECT_DIR/.log/semisyn_extract_${ts}.log"

python "$PROJECT_DIR/semisynthetic_extract_features.py" \
    --output-dir "$FEATURE_DIR" \
    --clip-batch-size "$CLIP_BATCH_SIZE" \
    --device "$DEVICE" \
    2>&1 | tee -a "$log_file"

# ---- Step 2: SAE experiments (sweep over k) ---- #
for k in "${K_VALUES[@]}"; do
    ts=$(date +"%Y%m%d_%H%M%S")
    run_tag="b${BATCH_SIZE}_e${NUM_EPOCHS}_s${NUM_SEEDS}_${ts}"
    log_file="$PROJECT_DIR/.log/semisyn_tinyimagenet_k${k}_${ts}.log"

    echo "======================================================"
    echo "Step 2: SAE experiment | latent=${LATENT_SIZE} | k=${k}"
    echo "  epochs=${NUM_EPOCHS} seeds=${NUM_SEEDS} batch=${BATCH_SIZE}"
    echo "  Log: ${log_file}"
    echo "======================================================"

    python "$PROJECT_DIR/semisynthetic_tinyimagenet_experiment.py" \
        --feature-dir "$FEATURE_DIR" \
        --latent-size "$LATENT_SIZE" \
        --k "$k" \
        --num-epochs "$NUM_EPOCHS" \
        --num-seeds "$NUM_SEEDS" \
        --seed-base "$SEED_BASE" \
        --lr "$LR" \
        --batch-size "$BATCH_SIZE" \
        --device "$DEVICE" \
        --gt-recovery-threshold "$GT_THRESHOLD" \
        --output-root "$OUTPUT_ROOT" \
        --run-tag "$run_tag" \
        2>&1 | tee -a "$log_file"
done

# ---- Step 3: Aggregate results ---- #
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
    row = {
        "path": str(path),
        "latent_size": m["latent_size"],
        "k": m["k"],
        "num_epochs": m["num_epochs"],
        "num_seeds": m["num_seeds"],
        "gt_threshold": m["gt_recovery_threshold"],
    }
    # Add all aggregate metrics
    for key, val in sorted(a.items()):
        row[key] = val
    rows.append(row)

rows.sort(key=lambda x: (int(x["k"]), int(x["latent_size"])))

if not rows:
    print("No result files found.")
    sys.exit(0)

# CSV
summary_csv = output_root / "summary_all.csv"
with open(summary_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

# Markdown summary
summary_md = output_root / "summary_all.md"
key_metrics = [
    "single_mgt_lp_img", "single_mip_lp_img",
    "single_mgt_me_img", "single_mip_me_img",
    "img_sae_mgt_lp_img", "img_sae_mip_lp_img",
    "txt_sae_mgt_lp_txt", "txt_sae_mip_lp_txt",
]
with open(summary_md, "w", encoding="utf-8") as f:
    f.write("# Semi-Synthetic Tiny ImageNet SAE Summary\n\n")
    f.write(f"- Total runs: {len(rows)}\n")
    f.write(f"- Source root: `{output_root}`\n\n")

    headers = ["k", "latent_size"] + [f"{m}_mean" for m in key_metrics]
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
    for r in rows:
        vals = [str(r.get(h, "")) for h in headers]
        f.write("| " + " | ".join(vals) + " |\n")

print(f"Wrote: {summary_csv}")
print(f"Wrote: {summary_md}")
PY

echo "All experiments completed."
