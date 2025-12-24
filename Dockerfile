FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
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

# Start the handler
CMD ["python", "-u", "handler.py"]
