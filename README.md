# LVLM Hallucination

Research project for studying hallucinations in Large Vision-Language Models.

## Requirements

### System Requirements
- **Python**: 3.12.x
- **CUDA**: 12.1+ (for GPU support)
- **NVIDIA Driver**: 535+ (required for CUDA 12.1)
- **Operating System**: Linux (Ubuntu 22.04+ recommended), macOS, or Windows with WSL2

### Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA Compute Capability 7.0+ (A100 recommended)
- **VRAM**: 80GB+ recommended for evaluation (40GB minimum with MIG)
- **RAM**: 32GB+ recommended

## Installation

### Quick Setup

Run the setup script to automatically configure the environment:

```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
1. Use your existing Python 3.12 if needed
2. Create a virtual environment (`.venv`)
3. Install PyTorch 2.5.1 with CUDA 12.1 support
4. Install all project dependencies from requirements.txt

### Manual Setup

If you prefer to install manually:

```bash
# Create virtual environment with Python 3
python3.12 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support
pip install --no-cache-dir --force-reinstall \
  --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Install other dependencies
pip install -r requirements.txt
```

## Package Versions

### Core ML Framework
- **torch**: 2.5.1 (with CUDA 12.1)
- **torchvision**: 0.20.1
- **torchaudio**: 2.5.1
- **transformers**: 4.56.2

### Data Processing
- **datasets**: 4.1.1
- **pillow**: 11.3.0

### Async & Networking
- **aiohttp**: 3.12.15
- **tqdm**: 4.67.1

### Configuration Management
- **omegaconf**: 2.3.0
- **pydantic**: 2.11.10

### Utilities
- **packaging**: 25.0
- **requests**: 2.32.5

## Project Structure

```
lvlm_hallucination/
├── src/
│   ├── common/          # Common utilities and base classes
│   ├── integrations/    # Custom integrations (e.g., flex_attention)
│   ├── models/          # Model implementations
│   │   ├── llama_real/  # Custom LLaMA implementation
│   │   └── reweighting_module/  # Attention reweighting module
│   └── utils.py         # Utility functions
├── test/                # Test scripts
├── requirements.txt     # Python dependencies
├── setup.sh            # Setup script
└── README.md           # This file
```

## Usage

### Activate Environment

```bash
source .venv/bin/activate
```

### Run Tests

```bash
# Test reweighting module
python test/test_reweighting_module.py
```

## Features

### Custom Models
- **LlamaReal**: Enhanced LLaMA implementation with attention reweighting
- **ReweightAttentionModule**: Attention reweighting mechanism for hallucination reduction

### Attention Mechanisms
- Flex Attention support (PyTorch 2.5.1+)
- Block-based attention pooling (mean/max pool)
- Learnable attention scaling

### Data Processing
- Async image fetching from URLs
- PIL Image processing with color model conversion
- Custom collators for vision-language data

## CUDA Compatibility

This project requires PyTorch 2.5.1 with CUDA 12.1 support. Ensure your system meets the following requirements:

### NVIDIA Driver Requirements
- **Linux**: NVIDIA Driver 535+ (required for CUDA 12.1)
- **Windows**: NVIDIA Driver 536+ or newer

### Verify Installation

```bash
# Check NVIDIA driver version
nvidia-smi

# Check CUDA toolkit version (if installed)
nvcc --version

# Verify PyTorch CUDA support
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

**Expected output:**
```
PyTorch: 2.5.1+cu121
CUDA available: True
CUDA version: 12.1
```

## Troubleshooting

### Python Version Issues
If you encounter Python version errors:
```bash
# Install Python 3.12 (Ubuntu/Debian)
sudo apt update
sudo apt install python3.12 python3.12-venv python3.12-dev

# Install Python 3 (macOS with Homebrew)
brew install python@3.12
```

### CUDA/Driver Issues

**Problem: `RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED`**

This error indicates driver-CUDA version mismatch. Solutions:

1. **Check current driver version:**
   ```bash
   nvidia-smi | grep "Driver Version"
   ```

2. **If driver < 535, upgrade manually:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install nvidia-driver-535
   sudo reboot
   ```

**Problem: Docker container environment**

If running in Docker, driver upgrade must be done on the **host system**, not inside the container:
- Exit container
- Upgrade host driver to 535+
- Restart container
- Install compatible PyTorch inside container

### Dependency Conflicts
If you encounter dependency conflicts:
```bash
# Clean install
rm -rf .venv
./setup.sh
```

### MIG (Multi-Instance GPU) Configuration
If using A100 with MIG enabled:
```bash
# Check MIG status
nvidia-smi -L

# Ensure sufficient memory per MIG instance (40GB+ recommended)
nvidia-smi
```

## Contributing

When contributing, ensure:
1. Python 3.12 compatibility
2. All tests pass
3. Code follows existing style conventions
4. Dependencies are pinned to exact versions

## License

[Add your license information here]

## Citation

[Add citation information if applicable]