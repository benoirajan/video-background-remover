# Base image with CUDA 12 and Python
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-gpu.txt requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements-gpu.txt

COPY . /workspace

# Pre-download the default ResNet50 model into image
RUN mkdir -p /workspace/models && \
    curl -L -o /workspace/models/rvm_resnet50_fp32.onnx https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_resnet50_fp32.onnx

ENTRYPOINT ["python3", "/workspace/process_video.py"]
CMD ["--help"]
