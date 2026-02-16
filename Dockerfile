ARG CUDA_IMAGE=nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

# Install python3 and system dependencies for audio
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libsndfile1 \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only runtime application files
COPY api.py main.py voice_runtime.py ./

# Set default environment variables for GPU inference
ENV WHISPER_DEVICE=cuda
ENV WHISPER_COMPUTE_TYPE=float16
ENV HF_HOME=/opt/hf-cache

# Pre-download Whisper model (small) to bake it into the image
# Note: we use device='cpu' here just for the download step, otherwise it fails during build if no GPU
RUN mkdir -p ${HF_HOME}
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

# Set offline mode AFTER download so runtime creates no new connections
ENV HF_HUB_OFFLINE=1

# Run as a non-root user at runtime
RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app ${HF_HOME}
USER appuser

# Expose port
EXPOSE 8000

# Container health check endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# Run API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
