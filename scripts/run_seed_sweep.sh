#!/usr/bin/env bash
# Per-seed driver: trains all SAE variants for a single seed and (cc3m only)
# also runs cross-modal steering eval and MMS eval at the per-seed output root.
#
# Usage:
#   bash scripts/run_seed_sweep.sh cc3m 1
#   bash scripts/run_seed_sweep.sh coco 2
#
# Output goes under outputs/real_exp_<dataset>_s<seed>/.

set -euo pipefail

DATASET=${1:?"Usage: $0 <coco|cc3m|cc3m_*> <seed>"}
SEED=${2:?"Usage: $0 <coco|cc3m|cc3m_*> <seed>"}

case "$DATASET" in
  coco|cc3m|cc3m_*) ;;
  *) echo "Unknown dataset: $DATASET (expected: coco, cc3m, or cc3m_<variant>)"; exit 2 ;;
esac

# Any cc3m or cc3m_<variant> dataset goes through the post-train evals
# (steering / MMS / MS). coco does not.
IS_CC3M_LIKE=0
if [[ "$DATASET" == cc3m || "$DATASET" == cc3m_* ]]; then
  IS_CC3M_LIKE=1
fi

# TRAIN_CACHE: the embedding cache the SAE was trained on (model-matched).
# EXT_CACHE: independent external encoder used as monosemanticity probe (MetaCLIP).
# Per-variant defaults; env overrides win so callers can plug in arbitrary models.
if [[ -z "${TRAIN_CACHE_COCO:-}" ]]; then
  case "$DATASET" in
    cc3m_siglip2) TRAIN_CACHE_COCO="cache/siglip2_base_coco" ;;
    *)            TRAIN_CACHE_COCO="cache/clip_b32_coco" ;;
  esac
fi
if [[ -z "${TRAIN_CACHE_CC3M_VAL:-}" ]]; then
  case "$DATASET" in
    cc3m_siglip2) TRAIN_CACHE_CC3M_VAL="cache/siglip2_base_cc3m_val" ;;
    *)            TRAIN_CACHE_CC3M_VAL="cache/clip_b32_cc3m_val" ;;
  esac
fi
if [[ -z "${TRAIN_CACHE_IMAGENET:-}" ]]; then
  case "$DATASET" in
    cc3m_siglip2) TRAIN_CACHE_IMAGENET="cache/siglip2_base_imagenet" ;;
    *)            TRAIN_CACHE_IMAGENET="cache/clip_b32_imagenet" ;;
  esac
fi
MAX_FIRE_RATE=${MAX_FIRE_RATE:-0.5}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

mkdir -p .log

ROOT="outputs/real_exp_${DATASET}_s${SEED}"
SRC_CFG="configs/real/${DATASET}.yaml"
TMP_CFG="/tmp/real_exp_${DATASET}_s${SEED}.yaml"

# Generate per-seed YAML config: rewrite seed and output.root only.
sed -E \
  -e "s/^( *)seed: *[0-9]+/\1seed: ${SEED}/" \
  -e "s|^( *)root: *outputs/real_exp_${DATASET}\$|\1root: ${ROOT}|" \
  "$SRC_CFG" > "$TMP_CFG"

now() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[$(now)] [seed=${SEED}/${DATASET}] $*"; }

log "BEGIN sweep"
log "config=$TMP_CFG  root=$ROOT"

# 1. Main pipeline (train + perm + eval + table)
log "STEP train+eval pipeline"
python scripts/real_alpha/run_real_v2.py --config "$TMP_CFG"

# 2. cc3m / cc3m_siglip2: cross-modal steering + MS (CC3M val only) at ROOT.
# MMS (Kaushik 2026) and MS-COCO MS deprecated — not reported in current paper.
if [[ "$IS_CC3M_LIKE" == "1" ]]; then
  log "STEP cross-modal steering cache=$TRAIN_CACHE_COCO"
  ROOT="$ROOT" CACHE_DIR="$TRAIN_CACHE_COCO" \
    bash scripts/run_eval_cross_modal_steering.sh

  log "STEP MS (CC3M val, MetaCLIP) train_cache=$TRAIN_CACHE_CC3M_VAL"
  ROOT="$ROOT" \
    DATASET=cc3m SPLIT=validation \
    TRAIN_CACHE="$TRAIN_CACHE_CC3M_VAL" \
    EXT_CACHE=cache/metaclip_b32_cc3m_val \
    OUT_ROOT="$ROOT/ms_cc3m_val" \
    bash scripts/run_eval_ms.sh
fi

# 3. ImageNet zero-shot with always-on latent filter (drop fire_rate > 0.5).
log "STEP ImageNet zero-shot filtered (max_fire_rate=$MAX_FIRE_RATE)"
ROOT="$ROOT" CACHE_DIR="$TRAIN_CACHE_IMAGENET" MAX_FIRE_RATE="$MAX_FIRE_RATE" \
  bash scripts/run_eval_zs_filtered.sh

log "DONE sweep"
