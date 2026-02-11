import os
import subprocess
import json
import httpx
from abc import ABC, abstractmethod
from typing import Optional
from config import settings

# --- STT Backend Abstraction ---
class STTBackend(ABC):
    @abstractmethod
    async def transcribe(self, audio_data: bytes) -> str:
        pass

class GroqSTTBackend(STTBackend):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.url = "https://api.groq.com/openai/v1/audio/transcriptions"

    async def transcribe(self, audio_data: bytes) -> str:
        if not self.api_key: return "Error: GROQ_API_KEY missing"
        
        # Groq expects a file-like object or tuple (filename, content, type)
        files = {'file': ('audio.wav', audio_data, 'audio/wav')}
        data = {
            'model': 'distil-whisper-large-v3-en',
            'response_format': 'json',
            'temperature': 0.0
        }
        headers = {'Authorization': f'Bearer {self.api_key}'}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.url, headers=headers, files=files, data=data, timeout=10.0)
                if response.status_code == 200:
                    return response.json().get('text', '')
                else:
                    return f"Groq STT Error: {response.text}"
        except Exception as e:
            return f"Groq STT Exception: {e}"

class WhisperCppBackend(STTBackend):
    def __init__(self):
        # Assuming whisper.cpp is in a 'whisper.cpp' subfolder in backend or similar
        self.executable = os.path.join(os.getcwd(), "whisper.cpp", "main.exe") 
        self.model_path = os.path.join(os.getcwd(), "whisper.cpp", "models", "ggml-base.en.bin")
        
        # Verify executable exists, else fallback or warn
        if not os.path.exists(self.executable):
            print(f"Warning: Whisper.cpp executable not found at {self.executable}")

    async def transcribe(self, audio_data: bytes) -> str:
        # Save audio to temporary wav file
        # Whisper.cpp main expects a WAV file (16kHz, 16-bit)
        # For now, just a placeholder implementation since handling binary data 
        # via subprocess requires careful file management.
        
        # TODO: Implement actual WAV file writing and subprocess call
        # This requires 'wave' module and ensuring input format is correct.
        return "Whisper STT Placeholder: Transcription not implemented yet."

class VoskBackend(STTBackend):
    def __init__(self):
        try:
            from vosk import Model, KaldiRecognizer
            self.model = Model("model") # Expects 'model' folder in CWD
            self.recognizer_cls = KaldiRecognizer
        except ImportError:
            print("Vosk not installed or model missing.")
            self.model = None

    async def transcribe(self, audio_data: bytes) -> str:
        if not self.model: return "Error: Vosk model not loaded."
        # Placeholder for actual Vosk processing
        return "Vosk STT Placeholder"

# --- STT Service ---
class STTService:
    def __init__(self):
        self.provider = settings.STT_PROVIDER
        self.backend = self._get_backend()

    def _get_backend(self) -> STTBackend:
        if self.provider == "vosk":
            print("Initializing STT Backend: Vosk")
            return VoskBackend()
        elif self.provider == "groq":
            print("Initializing STT Backend: Groq (distil-whisper)")
            return GroqSTTBackend(settings.GROQ_API_KEY)
        else:
            print("Initializing STT Backend: Whisper.cpp")
            return WhisperCppBackend()

    async def transcribe_audio(self, audio_bytes: bytes) -> str:
        if not audio_bytes: return ""
        return await self.backend.transcribe(audio_bytes)

stt_service = STTService()
