import unittest

from fastapi.testclient import TestClient

from api import create_app


class FakeServices:
    def __init__(self):
        self.language = "pt"
        self.current_config = {"whisper_lang": "pt"}
        self.voice_config = {"pt": {"voice_id": "Camila"}, "en": {"voice_id": "Joanna"}}

    def transcribe_file(self, filename: str, language: str | None = None):
        return "ola mundo", language or "pt"

    def synthesize_speech(self, text: str, language: str, output_format: str):
        return b"audio-bytes"


class ApiTests(unittest.TestCase):
    def test_health(self):
        app = create_app(services_factory=FakeServices)
        with TestClient(app) as client:
            resp = client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "ok")
        self.assertEqual(resp.json()["mode"], "pt")

    def test_tts_validation(self):
        app = create_app(services_factory=FakeServices)
        with TestClient(app) as client:
            resp = client.post("/tts", json={"text": "x", "language": "pt", "format": "wav"})
        self.assertEqual(resp.status_code, 400)

    def test_tts_success(self):
        app = create_app(services_factory=FakeServices)
        with TestClient(app) as client:
            resp = client.post("/tts", json={"text": "x", "language": "pt", "format": "mp3"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.content, b"audio-bytes")

    def test_transcribe_success(self):
        app = create_app(services_factory=FakeServices)
        with TestClient(app) as client:
            resp = client.post(
                "/transcribe",
                files={"file": ("sample.wav", b"dummy-audio", "audio/wav")},
            )
        self.assertEqual(resp.status_code, 200)
        payload = resp.json()
        self.assertEqual(payload["text"], "ola mundo")
        self.assertEqual(payload["language"], "pt")


if __name__ == "__main__":
    unittest.main()
