#!/bin/bash
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_DIR"

# Device: auto, cuda, mps, cpu
DEVICE="${DEVICE:-auto}"

# Grid search: latent 1-10, k 1-5, min-active 1-5
TOTAL=0
SKIPPED=0
RUN=0

for latent in $(seq 1 10); do
    for k in $(seq 1 5); do
        for active in $(seq 1 5); do
            for wt in false true; do
                TOTAL=$((TOTAL + 1))

                # skip if k > latent (can't select more than available)
                if [ "$k" -gt "$latent" ]; then
                    SKIPPED=$((SKIPPED + 1))
                    continue
                fi

                RUN=$((RUN + 1))
                TAG="latent${latent}_k${k}_active${active}_wt${wt}"
                OUT_DIR="outputs/synthetic_sae"
                VIDEO="${OUT_DIR}/${TAG}.mp4"

                WT_FLAG=""
                if [ "$wt" = "true" ]; then
                    WT_FLAG="--weight-tie"
                fi

                echo "=== [${RUN}] ${TAG} ==="

                python synthetic_train.py \
                    --feature-latent-dim 5 \
                    --representation-dim 2 \
                    --num-train 50000 \
                    --sparsity 0.999 \
                    --min-active "$active" \
                    --max-interference 0.3 \
                    --latent-size "$latent" \
                    --k "$k" \
                    --lr 1e-3 \
                    --batch-size 32 \
                    --num-epochs 1 \
                    --viz-every 5 \
                    --seed 42 \
                    --output-dir "$OUT_DIR" \
                    --video-path "$VIDEO" \
                    --device "$DEVICE" \
                    $WT_FLAG
            done
        done
    done
done

echo "=== Done: ${RUN} runs completed, ${SKIPPED} skipped (k > latent) ==="