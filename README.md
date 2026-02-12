## Project Description

**Local-First Voice AI Agent (Python)**

This project is a **local-first, modular voice AI agent** built in Python, designed to evolve from a simple headset-based prototype into a full production-grade telephone agent.

It also serves as a specialized **Voice Skill** for **[Openclaw](https://github.com/ricardotrevisan/ai-conversational-skill)** (formerly clawdbot), providing a high-performance, containerized alternative to the default plugnplay TTS/STT framework.

The system implements a real-time conversational loop using:

* **Local Speech-to-Text (Whisper)** for high-quality Portuguese (PT-BR) transcription
* **LLM via API (OpenAI)** for conversational intelligence
* **Text-to-Speech (AWS Polly)** for natural voice responses
* **Python audio I/O** for microphone capture and speaker playback

The initial phase runs entirely on a local machine (headset → Python → voice), enabling fast iteration on latency, prompts, and conversational behavior **without the complexity of telephony or orchestration frameworks**.

---

## Core Architecture (Phase 1)

```
Microphone
   ↓
Whisper (local STT)
   ↓
LLM (OpenAI API)
   ↓
AWS Polly (TTS)
   ↓
Speaker
```

### Runtime Modules (v1.1.0)
- `voice_runtime.py`: shared runtime services and env-based configuration (Whisper, Polly, OpenAI clients).
- `api.py`: FastAPI app with lifespan-managed startup that initializes runtime services.
- `main.py`: interactive local voice loop that consumes the same shared runtime module.

This approach prioritizes:

* Low operational complexity
* Full control over the conversation loop
* Clear separation of concerns
* Easy transition to production infrastructure

---

## Design Philosophy

* **Local-first**: validate behavior before scaling infrastructure
* **API-driven**: STT, LLM, and TTS are cleanly replaceable
* **Real-time oriented**: optimized for conversational latency
* **Production-minded**: architecture mirrors real telephony systems

This project intentionally avoids premature integration with telephony, workflow engines, or tool calling. Those concerns are introduced only after the conversational core is proven.

---

## Planned Evolution

Future phases extend the same core into:

* Continuous conversational loops with VAD and barge-in
* Twilio Media Streams for real phone calls
* Tool calling and backend actions (scheduling, CRM, logging)
* Workflow orchestration (e.g., n8n)
* Compliance-aware call handling (healthcare, support, etc.)

---

## Use Cases

* Voice assistants (customer support, scheduling, triage)
* AI receptionists (medical, dental, professional services)
* Conversational AI research and prototyping
* Telephony-ready voice agent foundations

---

## Openclaw Integration (Voice Skill)

This project is designed to act as the "ears and mouth" for the Openclaw agent ecosystem. By running this service as a Docker container, it replaces heavier or more complex plugins, offering:
*   **Dedicated Resources**: Runs STT/TTS in its own container (GPU-accelerated).
*   **Standard API**: Exposes endpoints that Openclaw's `voice-agent` skill consumes.
*   **WhatsApp Compatibility**: Specifically tuned to generate MP3/Opus audio that works with mobile messaging apps.
*   **OpenClaw Bundle**: This project powers the **[Voice Agent Skill](https://www.clawhub.com/ricardotrevisan/voice-agent)** on OpenClaw. Installing this skill provides the necessary client logic to interact with this voice backend.

The `voice-agent` skill is client-only. It does not start Docker containers or manage backend runtime.
Run backend setup from this repository docs (`walkthrough.md`, `DOCKER_README.md`) and then use the skill client against `http://localhost:8000`.

---

## How to Run

### Local Chat Loop
To run the interactive voice agent locally:
1.  Install dependencies: `pip install -r requirements.txt`
2.  Setup `.env` (see `.env.example`). You will need your OpenAI API Key and AWS Credentials.
3.  Run: `python main.py`

### API & Docker (New!)
The project now exposes STT and TTS as an API, deployable via Docker with GPU support.
See **[walkthrough.md](walkthrough.md)** for detailed instructions on:
*   Building the Docker image
*   Running with GPU or CPU
*   Using the API endpoints (`/transcribe`, `/tts`)

### Testing
- Baseline API tests are included under `tests/`.
- Syntax validation command:
  - `python3 -m py_compile voice_runtime.py api.py main.py tests/test_api.py`
