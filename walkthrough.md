# Voice Agent API & Docker Walkthrough

This document explains how to run the Voice Agent STT/TTS services as an API using Docker.

## 🐳 Docker Deployment

### Prerequisites (GPU Support)
To run with GPU acceleration (Recommended), you must install the **NVIDIA Container Toolkit** on your host machine.

**Quick Install (Ubuntu/Debian):**
```bash
# 1. Configure the repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# 2. Install the toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 3. Configure Docker to use it
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
*Official Guide: [NVIDIA Docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)*

### 1. Build the Image
The build process includes a step to **download and cache the model**, so it might take a moment.
```bash
docker build -t voice-agent-api .
```

4.  **Update Config**:
    Edit `skills/voice-agent/scripts/start.sh` to use the new image name:
    ```bash
    IMAGE_NAME="trevisanricardo/ai-voice-backend:latest"
    ```
    *Note: `start.sh` will attempt to pull the latest version of this image on every run.*

### 2. Run the Container

**Option A: With GPU (Recommended)**
Requires NVIDIA Container Toolkit.
```bash
docker run --gpus all --env-file .env -p 8000:8000 voice-agent-api
```

**Option B: CPU Only (Fallback)**
If you don't have a GPU, override the default settings:
```bash
docker run --env-file .env -p 8000:8000 \
  -e WHISPER_DEVICE=cpu \
  -e WHISPER_COMPUTE_TYPE=int8 \
  voice-agent-api
```

**Option C: Strictly Offline**
To verify no internet is used, disconnect your network or use `--network none` (note: this will break OpenAI/Polly unless you mock them, so mainly for testing local STT).
```bash
docker run --gpus all --env-file .env -p 8000:8000 voice-agent-api
```
(The container now enforces `HF_HUB_OFFLINE=1` by default).

The API will be available at `http://localhost:8000`.

---

## 🔌 API Usage

Interactive documentation (Swagger UI) is available at:  
👉 **http://localhost:8000/docs**

### 1. Transcribe Audio (STT)
Upload an audio file (wav, mp3, etc.) to get the transcription.

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/audio.wav"
```

**Response:**
```json
{
  "text": "Olá, como posso ajudar?",
  "language": "pt"
}
```

### 2. Text-to-Speech (TTS)
Convert text to audio. Supports `mp3` (default), `ogg_vorbis`, or `pcm`.

**cURL Example (MP3 - Best for Messengers):**
```bash
curl -X POST "http://localhost:8000/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "language": "en", "format": "mp3"}' \
  --output output.mp3
```

**cURL Example (PCM - Raw):**
```bash
curl -X POST "http://localhost:8000/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "language": "en", "format": "pcm"}' \
  --output output.pcm
```

**Play the PCM (using basic play):**
```bash
play -t raw -r 16000 -e signed -b 16 -c 1 output.pcm
```
(Requires `sox` or similar tools)

---

## ⚠️ Notes
*   **Cold Start**: The first request might be slow as models load into memory, though we preload them at startup.
*   **GPU**: If you have a GPU, pass `--gpus all` to `docker run` for faster Whisper inference.
