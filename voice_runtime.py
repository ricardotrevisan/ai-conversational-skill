import os
from dataclasses import dataclass

from dotenv import load_dotenv


VOICE_CONFIG = {
    "pt": {
        "voice_id": "Camila",
        "whisper_lang": "pt",
        "system_prompt": (
            "Você é um assistente de voz útil e rápido. "
            "Responda de forma direta e conversacional. "
            "Use frases curtas. Não use markdown ou listas."
        ),
    },
    "en": {
        "voice_id": "Joanna",
        "whisper_lang": "en",
        "system_prompt": (
            "You are a helpful and fast voice assistant. "
            "Answer directly and conversationally. "
            "Use short sentences. Do not use markdown or lists."
        ),
    },
}


@dataclass
class RuntimeConfig:
    llm_model: str
    whisper_size: str
    whisper_device: str
    whisper_compute_type: str
    aws_region: str
    language: str


class RuntimeServices:
    def __init__(self, config: RuntimeConfig):
        import boto3
        from faster_whisper import WhisperModel
        from openai import OpenAI

        self.config = config
        self.voice_config = VOICE_CONFIG
        self.language = config.language if config.language in VOICE_CONFIG else "pt"
        self.current_config = VOICE_CONFIG[self.language]

        self.openai_client = OpenAI()
        self.polly_client = boto3.client("polly", region_name=config.aws_region)
        self.whisper_model = WhisperModel(
            config.whisper_size,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )

    @classmethod
    def from_env(cls) -> "RuntimeServices":
        load_dotenv()
        config = RuntimeConfig(
            llm_model=os.getenv("LLM_MODEL", "gpt-5-mini"),
            whisper_size=os.getenv("WHISPER_SIZE", "small"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cuda"),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "float16"),
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
            language=os.getenv("LANGUAGE", "pt").lower(),
        )
        return cls(config)

    def transcribe_file(self, filename: str, language: str | None = None) -> tuple[str, str]:
        lang = language or self.current_config["whisper_lang"]
        segments, _ = self.whisper_model.transcribe(
            filename,
            language=lang,
            vad_filter=True,
            beam_size=1,
            temperature=0.0,
        )
        return " ".join(seg.text for seg in segments), lang

    def synthesize_speech(self, text: str, language: str, output_format: str) -> bytes:
        voice_id = self.voice_config[language]["voice_id"]
        response = self.polly_client.synthesize_speech(
            Text=text,
            VoiceId=voice_id,
            Engine="neural",
            OutputFormat=output_format,
            SampleRate="16000",
        )
        return response["AudioStream"].read()
