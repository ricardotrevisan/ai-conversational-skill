import argparse
import logging
import os
import queue
import re
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

from voice_runtime import RuntimeServices

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"

# VAD Parameters
SILENCE_THRESHOLD = 0.015
SILENCE_DURATION = 0.8
MAX_DURATION = 20.0

ABBREVIATIONS = {
    "dr.",
    "dra.",
    "mr.",
    "mrs.",
    "ms.",
    "sr.",
    "sra.",
    "vs.",
    "etc.",
    "e.g.",
    "i.e.",
}


class VoiceAgent:
    def __init__(self, services: RuntimeServices):
        self.services = services
        self.history = []
        self.tts_queue: queue.Queue[str | None] = queue.Queue()
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

    def _tts_worker(self) -> None:
        while True:
            text = self.tts_queue.get()
            if text is None:
                self.tts_queue.task_done()
                break

            if not text.strip():
                self.tts_queue.task_done()
                continue

            try:
                audio_stream = self.services.synthesize_speech(text, self.services.language, "pcm")
                audio_np = np.frombuffer(audio_stream, dtype=np.int16)
                sd.play(audio_np, samplerate=16000)
                sd.wait()
            except Exception as e:
                logger.exception("TTS worker failed: %s", e)

            self.tts_queue.task_done()

    def shutdown(self) -> None:
        self.tts_queue.put(None)
        self.tts_queue.join()

    def record_vad(self, filename: str = "input.wav") -> bool:
        print("🎤 Listening... (Speak now)")

        q: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata, frames, callback_time, status):
            if status:
                print(status, file=sys.stderr)
            q.put(indata.copy())

        audio_data = []
        silence_start = None
        speaking_started = False

        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=callback)
        start_time = time.time()

        with stream:
            while True:
                chunk = q.get()
                audio_data.append(chunk)

                chunk_f = chunk.astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(chunk_f**2))

                if not speaking_started and rms > SILENCE_THRESHOLD:
                    speaking_started = True
                    print("   (Voice detected...)")

                if speaking_started:
                    if rms < SILENCE_THRESHOLD:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > SILENCE_DURATION:
                            print("   (Silence detected, stopping.)")
                            break
                    else:
                        silence_start = None

                if time.time() - start_time > MAX_DURATION:
                    print("   (Max duration reached.)")
                    break

        if not audio_data:
            return False

        full_audio = np.concatenate(audio_data, axis=0)
        sf.write(filename, full_audio, SAMPLE_RATE)
        return True

    def transcribe(self, filename: str) -> str:
        text, _ = self.services.transcribe_file(filename)
        return text

    @staticmethod
    def _is_sentence_boundary(buffer: str, idx: int) -> bool:
        char = buffer[idx]
        if char not in ".!?":
            return False

        if char == "." and idx > 0 and idx + 1 < len(buffer):
            if buffer[idx - 1].isdigit() and buffer[idx + 1].isdigit():
                return False

        token_match = re.search(r"(\b[\w.]+)$", buffer[: idx + 1])
        if token_match and token_match.group(1).lower() in ABBREVIATIONS:
            return False

        if idx + 1 >= len(buffer):
            return False

        return buffer[idx + 1].isspace()

    @classmethod
    def _extract_complete_sentences(cls, buffer: str) -> tuple[list[str], str]:
        sentences: list[str] = []
        cursor = 0
        for idx in range(len(buffer)):
            if cls._is_sentence_boundary(buffer, idx):
                sentence = buffer[cursor : idx + 1].strip()
                if sentence:
                    sentences.append(sentence)
                cursor = idx + 1

        remainder = buffer[cursor:].lstrip()
        return sentences, remainder

    def generate_and_speak(self, user_text: str) -> None:
        system_prompt = self.services.current_config["system_prompt"]

        messages = [{"role": "system", "content": system_prompt}]
        if len(self.history) > 4:
            self.history = self.history[-4:]
        messages += self.history
        messages.append({"role": "user", "content": user_text})

        print("🤖 Thinking...", end="", flush=True)
        stream = self.services.openai_client.chat.completions.create(
            model=self.services.config.llm_model,
            messages=messages,
            stream=True,
        )

        buffer = ""
        full_response = ""

        for chunk in stream:
            content = chunk.choices[0].delta.content
            if not content:
                continue

            buffer += content
            full_response += content
            ready_sentences, buffer = self._extract_complete_sentences(buffer)
            for sentence in ready_sentences:
                self.tts_queue.put(sentence)

        if buffer.strip():
            self.tts_queue.put(buffer.strip())

        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": full_response})
        print("\n✅ Full response generated.")

    def run(self, file_input: str | None = None) -> None:
        print(f"🌍 Language Mode: {self.services.language} (Voice: {self.services.current_config['voice_id']})")
        print("\n🟢 Agent Ready. (Ctrl+C to exit)")

        try:
            while True:
                if file_input:
                    filename = file_input
                    if not os.path.exists(filename):
                        print(f"❌ File not found: {filename}")
                        return
                    print(f"📂 Using file: {filename}")
                else:
                    filename = "current_input.wav"
                    if not self.record_vad(filename):
                        continue

                t0 = time.time()
                text = self.transcribe(filename)
                t_stt = time.time() - t0

                if not text.strip():
                    print("⚠️ (Nothing heard)")
                    if file_input:
                        break
                    continue

                print(f"🗣️  You: {text} ({t_stt:.2f}s)")
                self.generate_and_speak(text)
                self.tts_queue.join()

                if file_input:
                    print("✅ File processing complete.")
                    break
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
        finally:
            self.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Use existing audio file instead of microphone")
    args = parser.parse_args()

    print("🚀 Initializing clients...")
    try:
        services = RuntimeServices.from_env()
    except Exception as e:
        print(f"❌ Initialization error: {e}")
        sys.exit(1)

    agent = VoiceAgent(services)
    agent.run(file_input=args.file)


if __name__ == "__main__":
    main()
