#!/usr/bin/env bash
# Helper script to run the Robust Video Matting pipeline

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/venv/bin/python3"
PROCESS_SCRIPT="${SCRIPT_DIR}/process_video.py"

# Auto-run setup if environment is not yet initialized
if [ ! -f "$VENV_PYTHON" ]; then
    echo "Virtual environment not detected. Running initial setup..."
    "${SCRIPT_DIR}/setup.sh"
fi

if [ -z "$1" ]; then
    echo "Usage: ./run.sh <path_to_video> [additional_options]"
    echo ""
    echo "Common Options:"
    echo "  --model resnet50 | mobilenetv3   (Default: resnet50 for maximum accuracy)"
    echo "  --downsample-ratio 0.35          (Use 0.35 - 0.4 for fine details / vertical video)"
    echo "  --alpha-threshold 0.05           (Suppress background shadows/noise, default: 0.03)"
    echo "  --max-frames 90                  (Process first N frames for quick tests)"
    echo "  --bg green | black | white       (Composite background color)"
    echo ""
    echo "Examples:"
    echo "  ./run.sh /home/benoi/Videos/Trimmed_1.mp4"
    echo "  ./run.sh /home/benoi/Videos/Trimmed_1.mp4 --downsample-ratio 0.35"
    echo "  ./run.sh /home/benoi/Videos/Trimmed_1.mp4 --model mobilenetv3 --max-frames 120"
    exit 1
fi

"$VENV_PYTHON" "$PROCESS_SCRIPT" "$@"
