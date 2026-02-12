import io
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from voice_runtime import RuntimeServices

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

ALLOWED_TTS_FORMATS = {"pcm", "mp3", "ogg_vorbis"}
MEDIA_TYPES = {
    "pcm": "application/octet-stream",
    "mp3": "audio/mpeg",
    "ogg_vorbis": "audio/ogg",
}


class TTSRequest(BaseModel):
    text: str
    language: str | None = None
    format: str = "mp3"


def create_app(services_factory: Callable[[], RuntimeServices] = RuntimeServices.from_env) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Initializing runtime services")
        app.state.services = services_factory()
        logger.info("Runtime services initialized (language=%s)", app.state.services.language)
        yield

    app = FastAPI(title="Voice Agent API", description="API for STT and TTS services", lifespan=lifespan)

    @app.post("/transcribe")
    async def transcribe_audio(request: Request, file: UploadFile = File(...)):
        suffix = os.path.splitext(file.filename or "upload.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name

        try:
            services: RuntimeServices = request.app.state.services
            current_lang = services.current_config["whisper_lang"]
            transcription, lang = services.transcribe_file(temp_path, language=current_lang)
            return {"text": transcription, "language": lang}
        except Exception as e:
            logger.exception("Transcription failed")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    @app.post("/tts")
    async def text_to_speech(request: Request, payload: TTSRequest):
        services: RuntimeServices = request.app.state.services
        target_lang = payload.language or services.language
        output_format = payload.format

        if target_lang not in services.voice_config:
            raise HTTPException(
                status_code=400,
                detail=f"Language '{target_lang}' not supported. Options: {list(services.voice_config.keys())}",
            )

        if output_format not in ALLOWED_TTS_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Format '{output_format}' not supported. Options: pcm, mp3, ogg_vorbis",
            )

        try:
            logger.info("TTS request language=%s chars=%s format=%s", target_lang, len(payload.text), output_format)
            audio_data = services.synthesize_speech(payload.text, target_lang, output_format)
            return StreamingResponse(io.BytesIO(audio_data), media_type=MEDIA_TYPES[output_format])
        except Exception as e:
            logger.exception("TTS failed")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/health")
    def health_check(request: Request):
        services: RuntimeServices = request.app.state.services
        return {"status": "ok", "mode": services.language}

    return app


app = create_app()
