import os
import sys
import time
import queue
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import boto3
import re
from openai import OpenAI
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
LLM_MODEL = "gpt-5-mini" 
WHISPER_SIZE = "small" 
WHISPER_SIZE = "small" 
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float16")
DEVICE = os.getenv("WHISPER_DEVICE", "cuda")

# Language Configuration
LANGUAGE = os.getenv("LANGUAGE", "pt").lower()
VOICE_CONFIG = {
    "pt": {
        "voice_id": "Camila", 
        "whisper_lang": "pt",
        "system_prompt": """
        Você é um assistente de voz útil e rápido. 
        Responda de forma direta e conversacional. 
        Use frases curtas. Não use markdown ou listas.
        """
    },
    "en": {
        "voice_id": "Joanna", 
        "whisper_lang": "en",
        "system_prompt": """
        You are a helpful and fast voice assistant.
        Answer directly and conversationally.
        Use short sentences. Do not use markdown or lists.
        """
    }
}

if LANGUAGE not in VOICE_CONFIG:
    print(f"⚠️ Language '{LANGUAGE}' not supported. Defaulting to 'pt'.")
    LANGUAGE = "pt"
    
CURRENT_CONFIG = VOICE_CONFIG[LANGUAGE]
print(f"🌍 Language Mode: {LANGUAGE} (Voice: {CURRENT_CONFIG['voice_id']})")

# VAD Parameters
SILENCE_THRESHOLD = 0.015  # Amplitude threshold for silence (normalized float)
SILENCE_DURATION = 0.8   # Seconds of silence to stop recording
MAX_DURATION = 20.0      # Max recording duration safety stop


print("🚀 Initializing clients...")
try:
    OPENAI_CLIENT = OpenAI()
    POLLY_CLIENT = boto3.client("polly", region_name=os.getenv("AWS_REGION", "us-east-1"))

    # Initialize Whisper
    print(f"🧠 Loading Whisper ({WHISPER_SIZE})...")
    WHISPER_MODEL = WhisperModel(WHISPER_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
except Exception as e:
    print(f"❌ Initialization error: {e}")
    sys.exit(1)

# --- Queues for Pipeline ---
# Use a queue to send sentences from LLM thread to TTS thread
tts_queue = queue.Queue()

def play_audio_from_buffer(audio_data, sample_rate=16000):
    sd.play(audio_data, samplerate=sample_rate)
    sd.wait()

def tts_worker():
    """Worker thread that consumes sentences and plays audio sequentially."""
    while True:
        text = tts_queue.get()
        if text is None: # Sentinel value to stop
            break
            
        if not text.strip():
            tts_queue.task_done()
            continue

        start_t = time.time()
        # print(f"🗣️ TTS Processando: '{text[:20]}...'")
        
        try:
            response = POLLY_CLIENT.synthesize_speech(
                Text=text,
                VoiceId=CURRENT_CONFIG["voice_id"],
                Engine="neural",
                OutputFormat="pcm",
                SampleRate="16000"
            )
            audio_stream = response["AudioStream"].read()
            audio_np = np.frombuffer(audio_stream, dtype=np.int16)
            
            # Play immediately
            # print(f"🔊 Playing ({len(audio_np)} samples, {time.time()-start_t:.2f}s latency)")
            sd.play(audio_np, samplerate=16000)
            sd.wait() # Block this thread until audio finishes (serial playback)
            
        except Exception as e:
            print(f"❌ TTS Error: {e}")
        
        tts_queue.task_done()

# Start TTS Worker Thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

class VoiceAgent:
    def __init__(self):
        self.history = []
        
    def record_vad(self, filename="input.wav"):
        """Records audio until silence is detected."""
        print("🎤 Listening... (Speak now)")
        
        q = queue.Queue()
        
        def callback(indata, frames, time, status):
            if status:
                print(status, file=sys.stderr)
            q.put(indata.copy())

        # Buffer for recording
        audio_data = []
        silence_start = None
        speaking_started = False
        
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype=DTYPE, callback=callback)
        start_time = time.time()
        
        with stream:
            while True:
                chunk = q.get()
                audio_data.append(chunk)
                
                # Simple Energy VAD
                # Calculate RMS of the chunk
                chunk_f = chunk.astype(np.float32) / 32768.0
                rms = np.sqrt(np.mean(chunk_f ** 2))
                
                # Logic: Wait for speech to start, then wait for silence
                if not speaking_started:
                    if rms > SILENCE_THRESHOLD:
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
                        silence_start = None # Reset silence counter if noise returns
                
                # Safety timeout
                if time.time() - start_time > MAX_DURATION:
                    print("   (Max duration reached.)")
                    break
        
        # Concatenate and save
        if not audio_data:
            return False
            
        full_audio = np.concatenate(audio_data, axis=0)
        sf.write(filename, full_audio, SAMPLE_RATE)
        return True

    def transcribe(self, filename):
        segments, _ = WHISPER_MODEL.transcribe(
            filename,
            language=CURRENT_CONFIG["whisper_lang"],
            vad_filter=True,
            beam_size=1, # Speed optimization
            temperature=0.0
        )
        return " ".join(seg.text for seg in segments)

    def generate_and_speak(self, user_text):
        """Streaming LLM -> Buffer Sentences -> TTS Queue"""
        
        system_prompt = CURRENT_CONFIG["system_prompt"]
        
        messages = [{"role": "system", "content": system_prompt}]
        # Add limited history if desired, avoiding infinite context growth
        if len(self.history) > 4: 
             self.history = self.history[-4:]
        messages += self.history
        messages.append({"role": "user", "content": user_text})
        
        # Stream response
        print("🤖 Thinking...", end="", flush=True)
        stream = OPENAI_CLIENT.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            stream=True
        )
        
        buffer = ""
        full_response = ""
        
        # Sentence delimiters
        delimiters = re.compile(r'([.!?])')
        
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                buffer += content
                full_response += content
                
                # Check for sentence end
                # If we find a punctuation mark, split and send valid sentence to queue
                parts = delimiters.split(buffer)
                
                # If we have [sentence, punct, remainder] or [sentence, punct], etc.
                if len(parts) > 2:
                    # We have at least one complete sentence + punctuation
                    # Reconstruct the sentence with its punctuation
                    sentence = parts[0] + parts[1]
                    if sentence.strip():
                        # print(f"-> Queue: {sentence}")
                        tts_queue.put(sentence)
                    
                    # The rest is the new buffer (could be multiple sentences, but usually one chunk has little)
                    buffer = "".join(parts[2:])
        
        # Flush remaining buffer
        if buffer.strip():
            tts_queue.put(buffer)
            
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": full_response})
        print("\n✅ Full response generated.")

    def run(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--file", type=str, help="Use existing audio file instead of microphone")
        args = parser.parse_args()

        print("\n🟢 Agent Ready. (Ctrl+C to exit)")
        try:
            while True:
                # 1. Capture (Mic or File)
                if args.file:
                    filename = args.file
                    if not os.path.exists(filename):
                         print(f"❌ File not found: {filename}")
                         return
                    print(f"📂 Using file: {filename}")
                    # Prepare to exit after one loop if file mode
                else:
                    filename = "current_input.wav"
                    if not self.record_vad(filename):
                        continue
                
                # Play Earcon (Ack)
                # (Optional: generate a beep or short sound here to acknowledge recording end)
                
                # 2. Transcribe
                t0 = time.time()
                text = self.transcribe(filename)
                t_stt = time.time() - t0
                
                if not text.strip():
                    print("⚠️ (Nothing heard)")
                    if args.file: break
                    continue
                    
                print(f"🗣️  You: {text} ({t_stt:.2f}s)")
                
                # 3. LLM + TTS Streaming
                self.generate_and_speak(text)
                
                # Wait for TTS to finish speaking before loop restarts
                tts_queue.join() 
                
                if args.file:
                    print("✅ File processing complete.")
                    break

        except KeyboardInterrupt:
            print("\n👋 Exiting...")

if __name__ == "__main__":
    agent = VoiceAgent()
    agent.run()
