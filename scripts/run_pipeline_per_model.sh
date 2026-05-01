#!/usr/bin/env bash
# End-to-end CC3M-trained SAE pipeline for one VLM.
#
# Pipeline per model:
#   1. Extract embedding caches (COCO test, ImageNet val, CC3M val, CC3M train).
#      All extractions are streaming — no full dataset download required.
#   2. Preprocess CC3M train cache into a stacked tensor for mmap-loading.
#   3. For each requested seed, run the full SAE sweep:
#        train (5 variants) + COCO retrieval/recon + ImageNet zero-shot
#        + cross-modal steering + MMS + MS.
#   4. Aggregate per-seed outputs into mean ± std tables and figures.
#
# Each step is idempotent: re-running re-uses cached artifacts.
#
# ─── Usage ─────────────────────────────────────────────────────────────────
# Native:
#   export HF_TOKEN=hf_xxx           # required (CC3M, ImageNet, gated weights)
#   bash scripts/run_pipeline_per_model.sh <model_key> [<seeds_csv>]
# Example:
#   bash scripts/run_pipeline_per_model.sh siglip_l 0,1,2
#
# Docker:
#   docker build -t vlm-sae-pipeline -f Dockerfile.vlm_large .
#   docker run --rm --gpus all \
#       -e HF_TOKEN=$HF_TOKEN \
#       -e MODEL=siglip_l -e SEEDS=0,1,2 \
#       -v $PWD/cache:/workspace/lvlm_hallucination/cache \
#       -v $PWD/outputs:/workspace/lvlm_hallucination/outputs \
#       -v $PWD/.log:/workspace/lvlm_hallucination/.log \
#       vlm-sae-pipeline
#
# ─── Available model keys ──────────────────────────────────────────────────
#   clip_l        OpenAI CLIP ViT-L/14            transformers, dim=768
#   openclip_l    OpenCLIP ViT-L/14 (DataComp-XL) openclip,    dim=768
#   siglip_l      Google SigLIP2 Large 256        transformers, dim=1024
#
# ─── Env overrides ─────────────────────────────────────────────────────────
#   CC3M_MAX_SAMPLES=1000000   cap CC3M train extraction (full = 2.87M)
#   CC3M_NUM_WORKERS=2         dataloader workers for CC3M streaming
#   BATCH=1024                 SAE training batch size
#   K=32                       SAE TopK
#   LATENT=8192                SAE width
#   EPOCHS=10                  CC3M training epochs
#
# ─── Prerequisites ─────────────────────────────────────────────────────────
#   - Single GPU, ≥16 GB VRAM (training uses ~10 GB at L=8192, batch=1024)
#   - ~30 GB free disk per model (cache ~12 GB, ckpts+evals ~5 GB)
#   - HF account; SigLIP2 weights are gated — accept the EULA at:
#       https://huggingface.co/google/siglip2-large-patch16-256
#   - For openclip_l, no gating but `open_clip_torch` must be installed
#     (pip install open_clip_torch).

set -euo pipefail

# Associative arrays require bash 4+. Docker image (linux) ships bash 5;
# native macOS ships bash 3.2 — install via `brew install bash`.
if (( BASH_VERSINFO[0] < 4 )); then
  echo "ERROR: bash >= 4 required (found ${BASH_VERSION}). Use the Docker image"
  echo "       or run 'brew install bash' and invoke with /opt/homebrew/bin/bash."
  exit 2
fi

MODELS_CSV=${1:-${MODEL:-}}
SEEDS_CSV=${2:-${SEEDS:-0,1,2}}

if [[ -z "$MODELS_CSV" ]]; then
  echo "ERROR: model key(s) required. Pass as arg-1 or via MODEL env."
  echo "Usage: $0 <model_key>[,<model_key>,...] [<seeds_csv>]"
  echo "       e.g. $0 siglip_l         (single model)"
  echo "            $0 clip_l,openclip_l,siglip_l 0,1,2  (3 models × 3 seeds)"
  exit 2
fi

IFS=',' read -r -a MODEL_KEYS <<< "$MODELS_CSV"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARNING: HF_TOKEN unset — gated weights and ImageNet streaming will fail."
fi

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"

cd "$(dirname "$0")/.."
if [[ -f .venv/bin/activate ]]; then source .venv/bin/activate; fi

# === Model registry ====================================================
# fields per row: BACKEND  MODEL_ID  PRETRAINED_TAG  HIDDEN  CACHE_PREFIX
declare -A MODEL_BACKEND
declare -A MODEL_HF_ID
declare -A MODEL_PRETRAINED
declare -A MODEL_HIDDEN
declare -A MODEL_PREFIX

#                       backend         model_id                                       pretrained                hidden  prefix
MODEL_BACKEND[clip_l]="transformers";   MODEL_HF_ID[clip_l]="openai/clip-vit-large-patch14";       MODEL_PRETRAINED[clip_l]="";                  MODEL_HIDDEN[clip_l]=768;   MODEL_PREFIX[clip_l]="clip_l14"
MODEL_BACKEND[openclip_l]="openclip";   MODEL_HF_ID[openclip_l]="ViT-L-14";                        MODEL_PRETRAINED[openclip_l]="datacomp_xl_s13b_b90k"; MODEL_HIDDEN[openclip_l]=768; MODEL_PREFIX[openclip_l]="openclip_l14"
MODEL_BACKEND[siglip_l]="transformers"; MODEL_HF_ID[siglip_l]="google/siglip2-large-patch16-256";  MODEL_PRETRAINED[siglip_l]="";                MODEL_HIDDEN[siglip_l]=1024; MODEL_PREFIX[siglip_l]="siglip2_large"
MODEL_BACKEND[openclip_b]="openclip";   MODEL_HF_ID[openclip_b]="ViT-B-32";                        MODEL_PRETRAINED[openclip_b]="datacomp_xl_s13b_b90k"; MODEL_HIDDEN[openclip_b]=512; MODEL_PREFIX[openclip_b]="datacomp_b32"

for KEY in "${MODEL_KEYS[@]}"; do
  if [[ -z "${MODEL_HF_ID[$KEY]:-}" ]]; then
    echo "ERROR: unknown model key '$KEY'"
    echo "Available keys: ${!MODEL_HF_ID[@]}"
    exit 2
  fi
done

CC3M_MAX_SAMPLES=${CC3M_MAX_SAMPLES:-1000000}
CC3M_NUM_WORKERS=${CC3M_NUM_WORKERS:-2}

mkdir -p .log cache outputs

now() { date "+%Y-%m-%d %H:%M:%S"; }

# Idempotent extraction wrapper: skip if meta.json already exists.
maybe_extract() {
  local desc=$1; local outdir=$2; shift 2
  if [[ -f "$outdir/meta.json" ]]; then
    log "[extract] $desc — already cached at $outdir, skipping"
    return 0
  fi
  log "[extract] $desc → $outdir"
  "$@"
}

# === Per-model loop =====================================================
for MODEL_KEY in "${MODEL_KEYS[@]}"; do
log() { echo "[$(now)] [$MODEL_KEY] $*"; }

BACKEND=${MODEL_BACKEND[$MODEL_KEY]}
HF_MODEL=${MODEL_HF_ID[$MODEL_KEY]}
PRETRAINED=${MODEL_PRETRAINED[$MODEL_KEY]}
HIDDEN=${MODEL_HIDDEN[$MODEL_KEY]}
PREFIX=${MODEL_PREFIX[$MODEL_KEY]}

CACHE_CC3M="cache/${PREFIX}_cc3m"
CACHE_CC3M_VAL="cache/${PREFIX}_cc3m_val"
CACHE_COCO="cache/${PREFIX}_coco"
CACHE_IMAGENET="cache/${PREFIX}_imagenet"

PRETRAINED_FLAG=()
if [[ -n "$PRETRAINED" ]]; then
  PRETRAINED_FLAG=(--pretrained "$PRETRAINED")
fi

log "========================================================="
log "BEGIN pipeline  model=$HF_MODEL  backend=$BACKEND  hidden=$HIDDEN"
log "  cache prefix:    $PREFIX"
log "  seeds:           ${SEEDS[*]}"
log "  cc3m max samples: $CC3M_MAX_SAMPLES"
log "========================================================="

# ─── 1. Extract caches (idempotent) ──────────────────────────────────────
log "===== STEP 1: extract caches ====="

maybe_extract "COCO Karpathy (5 captions × image)" "$CACHE_COCO" \
  python scripts/real_alpha/extract_clip_coco_cache.py \
    --backend "$BACKEND" --model "$HF_MODEL" "${PRETRAINED_FLAG[@]}" \
    --cache-dir "$CACHE_COCO"

maybe_extract "ImageNet val + 80 templates × 1k classes" "$CACHE_IMAGENET" \
  python scripts/real_alpha/extract_imagenet_cache.py \
    --backend "$BACKEND" --model "$HF_MODEL" "${PRETRAINED_FLAG[@]}" \
    --cache-dir "$CACHE_IMAGENET" \
    --splits validation

maybe_extract "CC3M val (~13k, streaming)" "$CACHE_CC3M_VAL" \
  python scripts/real_alpha/extract_clip_cc3m_cache.py \
    --backend "$BACKEND" --model "$HF_MODEL" "${PRETRAINED_FLAG[@]}" \
    --cache-dir "$CACHE_CC3M_VAL" \
    --hf-split validation \
    --num-workers "$CC3M_NUM_WORKERS"

maybe_extract "CC3M train (cap=$CC3M_MAX_SAMPLES, streaming)" "$CACHE_CC3M" \
  python scripts/real_alpha/extract_clip_cc3m_cache.py \
    --backend "$BACKEND" --model "$HF_MODEL" "${PRETRAINED_FLAG[@]}" \
    --cache-dir "$CACHE_CC3M" \
    --hf-split train \
    --num-workers "$CC3M_NUM_WORKERS" \
    --max-samples "$CC3M_MAX_SAMPLES"

# ─── 2. Preprocess CC3M ──────────────────────────────────────────────────
log "===== STEP 2: preprocess CC3M (image + text → stacked tensors) ====="
for MOD in image text; do
  if [[ -f "$CACHE_CC3M/${MOD}_embeddings_stack.pt" ]]; then
    log "[preprocess] CC3M $MOD already stacked, skipping"
  else
    python scripts/real_alpha/preprocess_cc3m_cache.py \
      --cache-dir "$CACHE_CC3M" --modality "$MOD"
  fi
done

# ─── 3. Per-model config (parameterized cc3m.yaml) ───────────────────────
DATASET_NAME="cc3m_${MODEL_KEY}"
TARGET_CFG="configs/real/${DATASET_NAME}.yaml"
SRC_CFG="configs/real/cc3m.yaml"

# Rewrite cache_dir / hidden_size / output.root / name from cc3m.yaml.
sed -E \
  -e "s|^( *cache_dir:) cache/clip_b32_cc3m\$|\1 ${CACHE_CC3M}|" \
  -e "s|^( *cache_dir:) cache/clip_b32_coco\$|\1 ${CACHE_COCO}|" \
  -e "s|^( *cache_dir:) cache/clip_b32_imagenet\$|\1 ${CACHE_IMAGENET}|" \
  -e "s|^( *hidden_size:) 512\$|\1 ${HIDDEN}|" \
  -e "s|^( *root:) outputs/real_exp_cc3m\$|\1 outputs/real_exp_${DATASET_NAME}|" \
  -e "s|^name: cc3m\$|name: ${DATASET_NAME}|" \
  "$SRC_CFG" > "$TARGET_CFG"

log "Wrote per-model config: $TARGET_CFG"

# ─── 4. 3-seed sweep ─────────────────────────────────────────────────────
for SEED in "${SEEDS[@]}"; do
  log "===== STEP 3.${SEED}: sweep seed=${SEED} ====="
  TRAIN_CACHE_COCO="$CACHE_COCO" \
  TRAIN_CACHE_CC3M_VAL="$CACHE_CC3M_VAL" \
  TRAIN_CACHE_IMAGENET="$CACHE_IMAGENET" \
    bash scripts/run_seed_sweep.sh "$DATASET_NAME" "$SEED"
done

# ─── 5. Aggregate ────────────────────────────────────────────────────────
PREFIX_OUT="outputs/real_exp_${DATASET_NAME}"
MEAN_DIR="${PREFIX_OUT}_mean"
mkdir -p "$MEAN_DIR"

ROOTS_CSV=""
for s in "${SEEDS[@]}"; do
  [[ -n "$ROOTS_CSV" ]] && ROOTS_CSV+=","
  ROOTS_CSV+="${PREFIX_OUT}_s${s}"
done

log "===== STEP 4: aggregate N=${#SEEDS[@]} tables + plots ====="
python scripts/real_alpha/aggregate_seed_table.py \
  --config "$TARGET_CFG" \
  --roots "$ROOTS_CSV" \
  --out "$MEAN_DIR"

if (( ${#SEEDS[@]} > 1 )); then
  STEER_ROOTS_CSV=""
  for s in "${SEEDS[@]}"; do
    [[ -n "$STEER_ROOTS_CSV" ]] && STEER_ROOTS_CSV+=","
    STEER_ROOTS_CSV+="${PREFIX_OUT}_s${s}/cross_modal_steering"
  done
  python scripts/real_alpha/aggregate_steering.py \
    --roots "$STEER_ROOTS_CSV" \
    --out "${MEAN_DIR}/cross_modal_steering"
  python scripts/plot_steering_map_bar.py --root "${MEAN_DIR}/cross_modal_steering"
  python scripts/plot_ms_curves.py  --prefix "$PREFIX_OUT"
  python scripts/plot_mms_curves.py --prefix "$PREFIX_OUT"
fi

log "===== MODEL DONE ====="
log "  per-seed roots: ${ROOTS_CSV//,/  }"
log "  mean tables/plots: $MEAN_DIR"

done  # per-MODEL loop

echo "[$(now)] ALL MODELS DONE: ${MODEL_KEYS[*]}"
