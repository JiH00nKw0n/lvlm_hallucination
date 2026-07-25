#!/usr/bin/env bash
# Same-modality control: build the four correlation matrices, then the figure.
#
#   bash scripts/real_alpha/run_rebuttal_EA.sh                 # setting F (COCO, K=8)
#   SETTING=T bash scripts/real_alpha/run_rebuttal_EA.sh       # setting T (CC3M, k=32)
#
# Setting T has one caption per image, so its different-caption panel is skipped.
set -euo pipefail
cd "$(dirname "$0")/../.."
[ -f env.sh ] && source env.sh || true
[ -f /mnt/elice/working/env.sh ] && source /mnt/elice/working/env.sh || true

PY=python; [ -x .venv/bin/python ] && PY=.venv/bin/python
SETTING="${SETTING:-F}"
RA="${RA:-1}"; RB="${RB:-2}"      # which two runs to compare

if [ "$SETTING" = "F" ]; then
  TAG=coco_k8; DATASET=coco; CACHE=cache/clip_b32_coco
  PANELS="img_img txt_txt txt_txt_diffcap img_txt"
  MAXS=0
else
  TAG=cc3m_k32; DATASET=cc3m; CACHE=cache/clip_b32_cc3m
  PANELS="img_img txt_txt img_txt"
  MAXS="${MAXS:-500000}"          # CC3M has 2.8M pairs; an even subsample is plenty
fi

MODELS=outputs/rebuttal_models
CKPT_A="$MODELS/${TAG}_r${RA}/final"
CKPT_B="$MODELS/${TAG}_r${RB}/final"
OUT="outputs/rebuttal_EA/${TAG}_r${RA}r${RB}"
mkdir -p "$OUT"

for c in "$CKPT_A" "$CKPT_B"; do
  [ -f "$c/model.safetensors" ] || { echo "missing checkpoint: $c" >&2; exit 1; }
done

for panel in $PANELS; do
  npz="$OUT/C_${panel}.npz"
  if [ -f "$npz" ]; then echo "[skip] $panel"; continue; fi
  echo "======== $panel ($(date -u +%H:%M:%SZ)) ========"
  # img_txt lives inside a single run; the others compare two runs. Spelled out
  # twice rather than with an optional array: bash 3.2 (macOS) errors on an
  # empty array expansion under `set -u`.
  if [ "$panel" = "img_txt" ]; then
    $PY scripts/real_alpha/build_cross_pair_C.py \
        --ckpt-a "$CKPT_A" \
        --panel "$panel" --dataset "$DATASET" --cache-dir "$CACHE" \
        --split train --max-samples "$MAXS" --out "$npz"
  else
    $PY scripts/real_alpha/build_cross_pair_C.py \
        --ckpt-a "$CKPT_A" --ckpt-b "$CKPT_B" \
        --panel "$panel" --dataset "$DATASET" --cache-dir "$CACHE" \
        --split train --max-samples "$MAXS" --out "$npz"
  fi
done

$PY - "$OUT" "$CKPT_A" "$CKPT_B" "$PANELS" <<'PY'
import json, sys
out, ckpt_a, ckpt_b, panels = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split()
spec = [{"panel": p, "npz": f"{out}/C_{p}.npz",
         "ckpt_a": ckpt_a, "ckpt_b": ckpt_a if p == "img_txt" else ckpt_b}
        for p in panels]
json.dump(spec, open(f"{out}/panels.json", "w"), indent=2)
print(f"wrote {out}/panels.json")
PY

$PY scripts/real_alpha/plot_same_modality_control.py \
    --panels "$OUT/panels.json" --out "$OUT/fig_same_modality.pdf"

echo "REBUTTAL_EA_DONE  -> $OUT"
