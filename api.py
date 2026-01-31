import os
import shutil
import tempfile
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from main import WHISPER_MODEL, POLLY_CLIENT, VOICE_CONFIG, LANGUAGE

app = FastAPI(title="Voice Agent API", description="API for STT and TTS services")

class TTSRequest(BaseModel):
    text: str
    language: str = None # Optional, defaults to env LANGUAGE
    format: str = "mp3"  # Optional: pcm, mp3, ogg_vorbis

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """
    Upload an audio file (wav, mp3, etc.) and get the transcription.
    """
    # Create a temp file to save the upload
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp:
        shutil.copyfileobj(file.file, temp)
        temp_path = temp.name

    try:
        # Determine language based on global config or default to None (auto-detect)
        # Using the globally configured Whisper language
        current_lang = VOICE_CONFIG.get(LANGUAGE, {}).get("whisper_lang", "pt")
        
        segments, _ = WHISPER_MODEL.transcribe(
            temp_path,
            language=current_lang, 
            vad_filter=True,
            beam_size=1,
            temperature=0.0
        )
        
        transcription = " ".join(seg.text for seg in segments)
        return {"text": transcription, "language": current_lang}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    """
    Convert text to speech audio.
    Supported formats: pcm, mp3, ogg_vorbis
    """
    target_lang = request.language or LANGUAGE
    output_format = request.format
    
    if target_lang not in VOICE_CONFIG:
        # Fallback or error? Let's error to be explicit, or fallback to pt
        raise HTTPException(status_code=400, detail=f"Language '{target_lang}' not supported. Options: {list(VOICE_CONFIG.keys())}")
    
    if output_format not in ["pcm", "mp3", "ogg_vorbis", "json"]:
         raise HTTPException(status_code=400, detail=f"Format '{output_format}' not supported. Options: pcm, mp3, ogg_vorbis")

    voice_id = VOICE_CONFIG[target_lang]["voice_id"]
    print(f"🗣️ TTS Request: '{request.text}' (Length: {len(request.text)})")
    
    
    try:
        response = POLLY_CLIENT.synthesize_speech(
            Text=request.text,
            VoiceId=voice_id,
            Engine="neural",
            OutputFormat=output_format,
            SampleRate="16000"
        )
        audio_stream = response["AudioStream"]
        
        # Determine correct media type
        media_types = {
            "pcm": "application/octet-stream",
            "mp3": "audio/mpeg",
            "ogg_vorbis": "audio/ogg",
            "json": "application/json"
        }
        
        return StreamingResponse(audio_stream, media_type=media_types.get(output_format, "application/octet-stream"))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "mode": LANGUAGE}
