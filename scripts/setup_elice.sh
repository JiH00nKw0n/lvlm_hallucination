#!/bin/bash
# Minimal environment bootstrap for elice-40g (or any linux host with
# cuda 12.1 / python 3.10 available). Creates .venv in the current repo
# root, installs PyTorch 2.5.1 + cu121 and the repo requirements, and
# stops there. Does NOT run any HF model evaluation like setup.sh does.
set -e

cd "$(dirname "$0")/.."

mkdir -p .log .cache outputs

if command -v python3 &> /dev/null; then
  PYTHON_CMD="python3"
else
  echo "Error: python3 not found"
  exit 1
fi
echo "Using $($PYTHON_CMD --version)"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  $PYTHON_CMD -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip

pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

pip install -r requirements.txt

echo ""
echo "Environment ready."
echo "Activate with: source .venv/bin/activate"
