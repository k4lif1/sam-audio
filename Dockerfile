FROM --platform=linux/amd64 nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.11 and system dependencies
# DEBIAN_FRONTEND=noninteractive prevents interactive prompts hanging the build
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Ensure python3.11 is the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip via module to ensure we use the right python's pip
RUN python -m ensurepip --upgrade && python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Copy project files
COPY pyproject.toml .
COPY sam_audio/ ./sam_audio/

# Install the package
RUN pip install --no-cache-dir .

# Install runpod
RUN pip install --no-cache-dir runpod requests

# Copy handler
COPY handler.py .

# Set environment variables for model caching
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface
ENV PYTHONUNBUFFERED=1 

# Start the handler using full path to be 100% sure
CMD ["/usr/bin/python3.11", "-u", "handler.py"]
