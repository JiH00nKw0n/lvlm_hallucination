FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-dev \
        python3.13-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /workspace/requirements.txt

RUN python3.13 -m ensurepip && \
    python3.13 -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python3.13 -m pip install --no-cache-dir \
        torch==2.8.0+cu121 \
        torchvision==0.23.0+cu121 \
        torchaudio==2.8.0+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    python3.13 -m pip install --no-cache-dir -r /workspace/requirements.txt

ENV HF_HOME=/workspace/.cache
ENV HF_DATASETS_CACHE=/workspace/.cache
ENV LOG_DIR=/workspace/.log

CMD ["bash", "-lc"]
