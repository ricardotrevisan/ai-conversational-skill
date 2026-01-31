# AI Voice Backend

A high-performance, containerized Voice API designed for AI Agents.
It provides **GPU-accelerated Speech-to-Text (STT)** via Faster-Whisper and **Text-to-Speech (TTS)** via AWS Polly (or other engines) in a simple REST API.

This image is designed to be the "ears and mouth" of your AI infrastructure, decoupling heavy audio processing from your main agent logic.

## Features
-   ✅ **Speech-to-Text**: Faster-Whisper (OpenAI Whisper) with VAD (Voice Activity Detection).
-   ✅ **Text-to-Speech**: AWS Polly Neural integration.
-   ✅ **GPU Accelerated**: Built on `nvidia/cuda` runtime for sub-second inference.
-   ✅ **Offline Ready**: Pre-downloads models (Whisper small/medium) to avoid runtime downloads.
-   ✅ **Containerized**: Drop-in replacement for audio handling in any agent stack.

## Quick Start

### 1. Run with Docker (CPU)
```bash
docker run -d -p 8000:8000 \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  -e AWS_REGION=us-east-1 \
  -e OPENAI_API_KEY=sk-... \
  trevisanricardo/ai-voice-backend:latest
```

### 2. Run with Docker (GPU)
Requires NVIDIA Container Toolkit.
```bash
docker run -d --gpus all -p 8000:8000 \
  --env-file .env \
  trevisanricardo/ai-voice-backend:latest
```

## Configuration

Secrets and settings are passed via Environment Variables.

| Variable | Description | Default |
| :--- | :--- | :--- |
| `AWS_ACCESS_KEY_ID` | AWS Key for Polly TTS | Required |
| `AWS_SECRET_ACCESS_KEY` | AWS Secret for Polly TTS | Required |
| `AWS_REGION` | AWS Region (e.g., us-east-1) | `us-east-1` |
| `OPENAI_API_KEY` | OpenAI API Key (required for initialization) | Required |
| `LANGUAGE` | Default Language Code | `pt` |
| `WHISPER_DEVICE` | Inference device (`cuda` or `cpu`) | `cuda` |
| `WHISPER_COMPUTE_TYPE` | Precision (`float16` or `int8`) | `float16` |

## API Endpoints

The service exposes port `8000`.

### `POST /transcribe`
Upload an audio file to get text.
-   **Input**: Multipart form data (`file=@audio.mp3`)
-   **Output**: `{"text": "Hello world", "language": "en"}`

### `POST /tts`
Convert text to audio.
-   **Input**: JSON `{"text": "Hello world", "format": "mp3"}` (Formats: `mp3`, `pcm`, `ogg_vorbis`). **Note**: `mp3` is recommended for WhatsApp compatibility.
-   **Output**: Audio binary stream.

### `GET /health`
-   **Output**: `{"status": "ok", "mode": "pt"}`

## Agent Integration (Client Script)
This backend is designed to work with lightweight agents like **Openclaw** (formerly clawdbot).
It serves as a robust alternative to the plugnplay framework for Voice I/O.
See [https://github.com/ricardotrevisan/ai-conversational-skill](https://github.com/ricardotrevisan/ai-conversational-skill) for the `voice-agent` skill client implementation.

### OpenClaw Bundle
This Docker image is the foundational backend for the **[Voice Agent Skill on ClawHub](https://www.clawhub.com/ricardotrevisan/voice-agent)**. You can bundle this skill with your OpenClaw agents to instantly enable voice capabilities.

## License
MIT
