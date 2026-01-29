## Phase 1 — Local Loop (headset → Python → voice)

First approach of a voice agent. In the future, with low latency, a quality local phone (Brazil) and PT-BR STT accuracy, we will use this as a base for a more complex system. NPL contact and renegotiations, etc.


```
🎤 Headset
   ↓ audio
Python
 ├─ STT: Whisper local
 ├─ LLM: OpenAI API
 ├─ Prompt (simple, PT-BR)
 └─ TTS: AWS Polly
   ↓ audio
🔊 Headset
```

No tools. No Twilio. No n8n.
**Just basic conversation.**

---

## Recommended Stack (local)

| Layer         | Choice                    |
| ------------- | ------------------------- |
| Audio Capture | `sounddevice`             |
| STT           | `faster-whisper`          |
| LLM           | OpenAI API                |
| TTS           | AWS Polly                 |
| Playback      | `sounddevice` / `pyaudio` |

---

## 1️⃣ Audio Capture (microphone)

```bash
pip install sounddevice soundfile numpy
```

```python
import sounddevice as sd
import soundfile as sf

DURATION = 5  # seconds
SAMPLE_RATE = 16000

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="int16"
)
sd.wait()
sf.write("input.wav", audio, SAMPLE_RATE)
```

---

## 2️⃣ STT — Whisper local (PT-BR)

```bash
pip install faster-whisper
```

```python
from faster_whisper import WhisperModel

model = WhisperModel(
    "medium",
    device="cpu",
    compute_type="int8"
)

segments, _ = model.transcribe(
    "input.wav",
    language="pt",
    vad_filter=True
)

user_text = " ".join(seg.text for seg in segments)
print("User:", user_text)
```

---

## 3️⃣ LLM — OpenAI API (simple prompt)

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI()

prompt = f"""
You are a polite and direct customer service assistant.
Answer in Brazilian Portuguese.
Do not be long-winded.

User: {user_text}
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

reply = response.choices[0].message.content
print("Assistant:", reply)
```

---

## 4️⃣ TTS — AWS Polly (PT-BR)

```bash
pip install boto3
```

```python
import boto3

polly = boto3.client("polly", region_name="us-east-1")

audio_stream = polly.synthesize_speech(
    Text=reply,
    VoiceId="Vitoria",
    OutputFormat="pcm",
    SampleRate="16000"
)["AudioStream"].read()

with open("output.pcm", "wb") as f:
    f.write(audio_stream)
```

---

## 5️⃣ Audio Playback

```python
import numpy as np
import sounddevice as sd

audio = np.frombuffer(audio_stream, dtype=np.int16)
sd.play(audio, samplerate=16000)
sd.wait()
```

---

## What you are validating in this phase

✔ STT PT-BR Quality
✔ Total loop latency
✔ Prompt tone
✔ TTS Naturalness
✔ Mental flow of conversation

Without any telephony noise.

---

## Phase 2 — Natural Evolution (later)

Only after that:

1. Transform script into **continuous loop**
2. Add **VAD / barge-in**
3. Create **conversation state**
4. Extract **voice gateway**
5. Integrate **Twilio Media Streams**
6. Only then plug in **n8n**

---

## What NOT to do now

* Tool calling
* n8n
* Telephony
* Multiple prompts
* Scheduling

All this comes **later**.

---
