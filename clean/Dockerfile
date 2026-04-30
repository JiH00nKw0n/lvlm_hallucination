# GPU image for the clean/ VLM-SAE experiment suite.
# Inherit pytorch+CUDA from official image; bake deps + repo, defer config to runtime.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY clean /app/clean
RUN pip install --no-cache-dir -e /app/clean

# Mount cache + outputs at runtime, e.g.:
#   docker run --rm --gpus all \
#     -e HF_TOKEN=$HF_TOKEN \
#     -e CONFIG=clean/configs/cc3m/overrides/clip_l14.yaml \
#     -v $PWD/cache:/app/cache -v $PWD/outputs:/app/outputs \
#     vlm-sae

COPY clean/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
