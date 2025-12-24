FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies including git for pip git+ installs
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Copy project files
COPY pyproject.toml .
COPY sam_audio/ ./sam_audio/

# Install the package with legacy resolver to handle conflicts
RUN pip install --no-cache-dir --use-deprecated=legacy-resolver . || \
    pip install --no-cache-dir .

# Install runpod
RUN pip install --no-cache-dir runpod requests

# Copy handler
COPY handler.py .

# Set environment variables for model caching
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface

# Start the handler
CMD ["python", "-u", "handler.py"]
