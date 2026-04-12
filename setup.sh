#!/bin/bash
set -e

mkdir -p .log .cache outputs

echo "=== LVLM Hallucination Project Setup ==="

# Use existing Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
    echo "Using $(python3 --version)"
else
    echo "Error: Python 3 not found"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Detect CUDA driver version and install matching PyTorch
echo "Detecting CUDA driver..."
if command -v nvidia-smi &> /dev/null; then
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    CUDA_VERSION=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
    echo "NVIDIA driver: $DRIVER_VERSION"

    # Try CUDA 12.4 first (most common on recent drivers), fallback to 12.1
    echo "Installing PyTorch with CUDA 12.4..."
    if pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124 2>/dev/null; then
        echo "Installed PyTorch 2.6.0+cu124"
    else
        echo "Falling back to CUDA 12.1..."
        pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu121
        echo "Installed PyTorch 2.6.0+cu121"
    fi
else
    echo "No NVIDIA GPU detected, installing CPU-only PyTorch..."
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
fi

# Install other dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "=== Verification ==="
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
python -c "from synthetic_theorem2_method import _make_sae; print('synthetic_theorem2_method: OK')"

echo ""
echo "Setup complete!"
echo "To activate: source .venv/bin/activate"
