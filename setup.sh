#!/usr/bin/env bash
# ==============================================================================
# Plug-and-Play Setup Script for Robust Video Matting (RVM)
# Supports: Linux (Ubuntu, Debian, CentOS, Fedora, Arch) & macOS
# Auto-detects NVIDIA GPU (CUDA) vs. CPU environments
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo "  🎬 Robust Video Matting (RVM) Plug-and-Play Setup"
echo "=========================================================="

# 1. Parse arguments or auto-detect
MODE="auto"
for arg in "$@"; do
    case "$arg" in
        --gpu)
            MODE="gpu"
            ;;
        --cpu)
            MODE="cpu"
            ;;
        -h|--help)
            echo "Usage: ./setup.sh [--gpu | --cpu]"
            echo "  --gpu: Force install ONNX Runtime with NVIDIA CUDA support"
            echo "  --cpu: Force install ONNX Runtime for CPU"
            echo "  (Default: auto-detects GPU via nvidia-smi)"
            exit 0
            ;;
    esac
done

# 2. Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 is not installed. Please install Python 3.8+."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Found Python $PYTHON_VERSION"

# 3. Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  ffmpeg not found in PATH."
    if command -v apt-get &> /dev/null && [ "$EUID" -eq 0 ]; then
        echo "Installing ffmpeg via apt-get..."
        apt-get update -qq && apt-get install -y -qq ffmpeg
    elif command -v apt-get &> /dev/null && command -v sudo &> /dev/null; then
        echo "Attempting to install ffmpeg with sudo apt-get..."
        sudo apt-get update -qq && sudo apt-get install -y -qq ffmpeg || echo "Could not auto-install ffmpeg. Please install ffmpeg manually."
    else
        echo "Please install ffmpeg manually for optimal video encoding (e.g. sudo apt install ffmpeg)."
    fi
else
    echo "✓ Found ffmpeg"
fi

# 4. Auto-detect GPU if mode is auto
if [ "$MODE" = "auto" ]; then
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
        echo "✓ NVIDIA GPU detected: $GPU_NAME"
        MODE="gpu"
    else
        echo "ℹ️  No NVIDIA GPU detected. Setting up CPU environment."
        MODE="cpu"
    fi
fi

# 5. Create Virtual Environment
VENV_DIR="${SCRIPT_DIR}/venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python virtual environment at ${VENV_DIR}..."
    if ! python3 -m venv "$VENV_DIR" 2>/dev/null; then
        echo "Installing virtual environment prerequisites..."
        if command -v apt-get &> /dev/null && command -v sudo &> /dev/null; then
            sudo apt-get install -y python3-venv python3-pip
            python3 -m venv "$VENV_DIR"
        else
            echo "❌ Failed to create virtual environment. Please install python3-venv."
            exit 1
        fi
    fi
fi

VENV_PYTHON="${VENV_DIR}/bin/python3"
VENV_PIP="${VENV_DIR}/bin/pip"

# 6. Install Python Dependencies
echo "Upgrading pip..."
"$VENV_PIP" install --upgrade pip -q

if [ "$MODE" = "gpu" ]; then
    echo "🚀 Installing GPU dependencies (onnxruntime-gpu, opencv, tqdm)..."
    "$VENV_PIP" install -r requirements-gpu.txt
else
    echo "💻 Installing CPU dependencies (onnxruntime, opencv, tqdm)..."
    "$VENV_PIP" install -r requirements.txt
fi

# 7. Pre-download Default Model (ResNet50)
mkdir -p "${SCRIPT_DIR}/models"
RESNET_MODEL="${SCRIPT_DIR}/models/rvm_resnet50_fp32.onnx"

if [ ! -f "$RESNET_MODEL" ]; then
    echo "⬇️  Downloading default RVM ResNet50 model (~103 MB)..."
    URL="https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx"
    if command -v curl &> /dev/null; then
        curl -L -o "$RESNET_MODEL" "$URL"
    elif command -v wget &> /dev/null; then
        wget -O "$RESNET_MODEL" "$URL"
    else
        "$VENV_PYTHON" -c "import urllib.request; urllib.request.urlretrieve('$URL', '$RESNET_MODEL')"
    fi
    echo "✓ ResNet50 model downloaded."
else
    echo "✓ Default ResNet50 model is already present."
fi

# 8. Verify Execution Provider
echo "Verifying ONNX Runtime environment..."
"$VENV_PYTHON" -c "
import onnxruntime as ort
providers = ort.get_available_providers()
print(f'Active Providers: {providers}')
if 'CUDAExecutionProvider' in providers:
    print('🎉 GPU acceleration (CUDA) is READY!')
else:
    print('✓ CPU execution is READY.')
"

# 9. Set permissions
chmod +x "${SCRIPT_DIR}/run.sh"
chmod +x "${SCRIPT_DIR}/process_video.py"
chmod +x "${SCRIPT_DIR}/setup.sh"

echo ""
echo "=========================================================="
echo "🎉 Setup Complete! You are ready to run video matting."
echo ""
echo "Quick Test:"
echo "  ./run.sh <path_to_video> --max-frames 90"
echo ""
echo "Full Run (High Quality):"
echo "  ./run.sh <path_to_video> --downsample-ratio 0.35"
echo "=========================================================="
