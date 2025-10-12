#!/bin/bash

# Required Python version
REQUIRED_PYTHON="3.12"

# Check if Python 3.12 is available
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
    if [ "$PYTHON_VERSION" = "$REQUIRED_PYTHON" ]; then
        PYTHON_CMD="python3"
    else
        echo "Error: Python 3.12 is required, but found $PYTHON_VERSION"
        exit 1
    fi
else
    echo "Error: Python 3.12 not found"
    exit 1
fi

echo "Using $PYTHON_CMD ($(${PYTHON_CMD} --version))"

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

# Install PyTorch ecosystem (CUDA 12.8 version)
echo "Installing PyTorch 2.8.0 with torchvision and torchaudio..."
pip install torch==2.8.0 torchvision==0.21.0 torchaudio==2.8.0

# Install other dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete!"