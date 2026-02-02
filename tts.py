import edge_tts
from typing import Optional
from config import language_manager

# --- TTS Service ---
class TTSService:
    async def speak(self, text: str, language_code: str = "en-US") -> Optional[bytes]:
        voice = language_manager.get_voice(language_code)
        try:
            communicate = edge_tts.Communicate(text, voice)
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            return audio_data
        except Exception as e:
            print(f"TTS Error ({voice}): {e}")
            return None

tts_service = TTSService()
