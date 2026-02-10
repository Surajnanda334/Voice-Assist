import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=False)

# --- Configuration ---
class Settings:
    def __init__(self):
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        self.GROQ_MODEL = os.getenv("GROQ_MODEL")
        self.SERPER_API_KEY = os.getenv("SERPER_API_KEY")
        self.MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.DB_NAME = os.getenv("MONGO_DB_NAME", "voice_assist_db")
        self.STT_PROVIDER = os.getenv("STT_PROVIDER", "whisper")

settings = Settings()

# --- Language Manager ---
class LanguageManager:
    def __init__(self, config_path="voice_config.json"):
        self.config_path = config_path
        self.languages = {}
        self.default_language = "en-US"
        self.load_config()

    def load_config(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.languages = json.load(f)
                print(f"LanguageManager: Loaded {len(self.languages)} languages.")
            else:
                print("LanguageManager: Config file not found, using defaults.")
                self.languages = {
                    "en-US": {"name": "English (US)", "code": "en-US", "tts_voice": "en-US-AriaNeural"}
                }
        except Exception as e:
            print(f"LanguageManager Error: {e}")

    def get_voice(self, lang_code: str):
        lang = self.languages.get(lang_code)
        if lang: return lang.get("tts_voice")
        # Fallback to English if not found
        return self.languages.get("en-US", {}).get("tts_voice", "en-US-AriaNeural")

language_manager = LanguageManager()
