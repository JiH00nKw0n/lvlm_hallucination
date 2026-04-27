FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.hf \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /workspace/lvlm_hallucination

RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential ca-certificates \
    libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN mkdir -p /workspace/.hf cache outputs .log

CMD ["bash", "scripts/real_alpha/run_multi_model_density_pipeline.sh"]
