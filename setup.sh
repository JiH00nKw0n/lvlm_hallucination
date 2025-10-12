#!/bin/bash

# Required Python version
REQUIRED_PYTHON="3.12"
REQUIRED_DRIVER="570"
REQUIRED_CUDA="12.8"

# Function to upgrade NVIDIA driver and CUDA
upgrade_driver_cuda() {
    local driver_version=$1

    echo "Driver $driver_version detected. Upgrading to Driver $REQUIRED_DRIVER+ for CUDA $REQUIRED_CUDA support..."

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        echo "Installing NVIDIA Driver $REQUIRED_DRIVER and CUDA $REQUIRED_CUDA on Linux (Ubuntu 22.04 base)..."

        # Download CUDA keyring
        echo "Downloading CUDA keyring..."
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb

        if [ -f cuda-keyring_1.1-1_all.deb ]; then
            echo "Installing CUDA repository..."
            sudo dpkg -i cuda-keyring_1.1-1_all.deb
            sudo apt-get update

            echo "Installing NVIDIA Driver $REQUIRED_DRIVER and CUDA Toolkit 12-8..."
            if sudo apt-get -y install cuda-drivers-$REQUIRED_DRIVER cuda-toolkit-12-8; then
                # Update environment variables
                export PATH=/usr/local/cuda-12.8/bin:$PATH
                export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH

                # Add to bashrc for persistence
                if ! grep -q "/usr/local/cuda-12.8/bin" ~/.bashrc; then
                    echo "export PATH=/usr/local/cuda-12.8/bin:\$PATH" >> ~/.bashrc
                    echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
                fi

                echo ""
                echo "=========================================="
                echo "Driver $REQUIRED_DRIVER and CUDA $REQUIRED_CUDA installation complete!"
                echo "IMPORTANT: Reboot required for driver update."
                echo "After reboot, run this script again."
                echo "=========================================="
                echo ""

                read -p "Reboot now? (y/N): " -n 1 -r
                echo
                if [[ $REPLY =~ ^[Yy]$ ]]; then
                    sudo reboot
                else
                    echo "Please reboot and re-run this script."
                    exit 0
                fi
            else
                echo "Error: Driver/CUDA installation failed."
                return 1
            fi

            rm -f cuda-keyring_1.1-1_all.deb
        else
            echo "Error: Failed to download CUDA keyring."
            echo "Please install manually: https://developer.nvidia.com/cuda-12-8-0-download-archive"
            return 1
        fi
    else
        echo "Warning: Automatic upgrade only supported on Linux."
        echo "Please install manually: https://developer.nvidia.com/cuda-12-8-0-download-archive"
        return 1
    fi

    return 0
}

# Function to install Python 3.12
install_python312() {
    echo "Python 3.12 not found. Attempting to install..."

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt &> /dev/null; then
            echo "Installing Python 3.12 via apt..."
            sudo apt update
            sudo apt install -y software-properties-common
            sudo add-apt-repository -y ppa:deadsnakes/ppa
            sudo apt update
            sudo apt install -y python3.12 python3.12-venv python3.12-dev
            curl -sS https://bootstrap.pypa.io/get-pip.py | sudo python3.12
        elif command -v yum &> /dev/null; then
            echo "Installing Python 3.12 via yum (building from source)..."
            sudo yum install -y gcc openssl-devel bzip2-devel libffi-devel zlib-devel wget make
            cd /tmp
            wget https://www.python.org/ftp/python/3.12.7/Python-3.12.7.tgz
            tar xzf Python-3.12.7.tgz
            cd Python-3.12.7
            ./configure --enable-optimizations
            sudo make altinstall
            cd -
        else
            echo "Error: Unsupported package manager"
            return 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v brew &> /dev/null; then
            echo "Installing Python 3.12 via Homebrew..."
            brew install python@3.12
            brew link python@3.12
        else
            echo "Error: Homebrew not found. Install from https://brew.sh"
            return 1
        fi
    else
        echo "Error: Unsupported OS"
        return 1
    fi

    return 0
}

# Check if Python 3.12 is available
if command -v python3.12 &> /dev/null; then
    PYTHON_CMD="python3.12"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
    if [ "$PYTHON_VERSION" = "$REQUIRED_PYTHON" ]; then
        PYTHON_CMD="python3"
    else
        echo "Found Python $PYTHON_VERSION, but Python 3.12 is required"
        install_python312
        if command -v python3.12 &> /dev/null; then
            PYTHON_CMD="python3.12"
        else
            echo "Error: Failed to install Python 3.12"
            exit 1
        fi
    fi
else
    echo "Python 3 not found"
    install_python312
    if command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
    else
        echo "Error: Failed to install Python 3.12"
        exit 1
    fi
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

# Check NVIDIA Driver version and upgrade if necessary
if command -v nvidia-smi &> /dev/null; then
    echo "Checking NVIDIA Driver version..."
    DRIVER_VERSION=$(nvidia-smi | grep "Driver Version" | awk '{print $6}')
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')

    if [ -n "$DRIVER_VERSION" ]; then
        echo "Current Driver Version: $DRIVER_VERSION"
        echo "Current CUDA Version: $CUDA_VERSION"

        # Extract driver major version
        DRIVER_MAJOR=$(echo "$DRIVER_VERSION" | cut -d. -f1)

        if [ "$DRIVER_MAJOR" -lt "$REQUIRED_DRIVER" ]; then
            upgrade_driver_cuda "$DRIVER_VERSION"
        else
            echo "Driver version $DRIVER_VERSION (>= $REQUIRED_DRIVER). No upgrade needed."
        fi
    else
        echo "Warning: Could not detect driver version."
    fi
else
    echo "Warning: nvidia-smi not found."
    echo "Please install NVIDIA drivers first."
fi

# Install PyTorch ecosystem (CUDA 12.8 version)
echo "Installing PyTorch 2.8.0 with torchvision and torchaudio..."
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0

# Install other dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Setup complete!"