FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

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

# Copy application code
COPY . .

# Set default environment variables for GPU inference
ENV WHISPER_DEVICE=cuda
ENV WHISPER_COMPUTE_TYPE=float16

# Pre-download Whisper model (small) to bake it into the image
# Note: we use device='cpu' here just for the download step, otherwise it fails during build if no GPU
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"

# Set offline mode AFTER download so runtime creates no new connections
ENV HF_HUB_OFFLINE=1

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
