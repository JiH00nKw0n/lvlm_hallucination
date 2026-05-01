#!/usr/bin/env bash
# Aggregate cc3m_siglip2 3-seed sweep into mean ± std tables, hammer-bar, MS/MMS curves.
# Run AFTER scripts/run_seed_sweep.sh cc3m_siglip2 {0,1,2} all complete.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

PREFIX="outputs/real_exp_cc3m_siglip2"
ROOTS="${PREFIX}_s0,${PREFIX}_s1,${PREFIX}_s2"
MEAN="${PREFIX}_mean"
mkdir -p "$MEAN"

# 1. Mean ± std table (CC3M / ImageNet / COCO retrieval+recon)
python scripts/real_alpha/aggregate_seed_table.py \
  --config configs/real/cc3m_siglip2.yaml \
  --roots "$ROOTS" \
  --out "$MEAN"

# 2. Steering 3-seed aggregate
python scripts/real_alpha/aggregate_steering.py \
  --roots "${PREFIX}_s0/cross_modal_steering,${PREFIX}_s1/cross_modal_steering,${PREFIX}_s2/cross_modal_steering" \
  --out "$MEAN/cross_modal_steering"

# 3. Hammer-bar plot
python scripts/plot_steering_map_bar.py --root "$MEAN/cross_modal_steering"

# 4. MS / MMS curves (read seed roots, write to mean dir)
python scripts/plot_ms_curves.py  --prefix "$PREFIX"
python scripts/plot_mms_curves.py --prefix "$PREFIX"

echo "DONE: aggregated under $MEAN"
ls -la "$MEAN" "$MEAN/cross_modal_steering"
