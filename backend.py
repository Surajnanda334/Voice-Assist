import os
import json
import asyncio
import base64
import re
import time
from typing import AsyncGenerator, List, Dict, Optional
from abc import ABC, abstractmethod
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import httpx
import ollama
import edge_tts

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

settings = Settings()

# --- Database Service ---
from motor.motor_asyncio import AsyncIOMotorClient

class DatabaseService:
    client: AsyncIOMotorClient = None
    db = None

    def connect(self):
        try:
            self.client = AsyncIOMotorClient(settings.MONGO_URI)
            self.db = self.client[settings.DB_NAME]
            print(f"Connected to MongoDB at {settings.MONGO_URI}")
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")

    def close(self):
        if self.client:
            self.client.close()
            print("MongoDB connection closed")

    async def log_conversation(self, query: str, response: str, route: str, search_query: str = None):
        if self.db is None: return
        try:
            document = {
                "timestamp": datetime.now(),
                "query": query,
                "response": response,
                "route": route,
                "search_query": search_query
            }
            await self.db.conversations.insert_one(document)
        except Exception as e:
            print(f"Error logging to MongoDB: {e}")

db_service = DatabaseService()

# --- Search Service ---
class SearchService:
    def __init__(self):
        self.api_key = settings.SERPER_API_KEY
        self.url = "https://google.serper.dev/search"

    async def search(self, query: str):
        if not self.api_key: return "Error: SERPER_API_KEY not found."
        
        headers = {'X-API-KEY': self.api_key, 'Content-Type': 'application/json'}
        payload = json.dumps({"q": query})
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(self.url, headers=headers, data=payload)
                if response.status_code == 200:
                    results = response.json()
                    snippets = []
                    if "organic" in results:
                        for item in results["organic"][:3]:
                            snippets.append(f"{item.get('title')}: {item.get('snippet')}")
                    return "\n".join(snippets) if snippets else "No results found."
                else:
                    return f"Search Error: {response.status_code}"
        except Exception as e:
            return f"Search Exception: {e}"

search_service = SearchService()

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

# --- LLM Backend Abstraction ---
class LLMBackend(ABC):
    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        pass

class OllamaBackend(LLMBackend):
    def __init__(self, model: str = "llama3"):
        self.model = model

    async def generate(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        try:
            async for chunk in await ollama.AsyncClient().chat(model=self.model, messages=messages, stream=True):
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
        except Exception as e:
            yield f"Ollama Error: {e}"

class GroqBackend(LLMBackend):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.url = "https://api.groq.com/openai/v1/chat/completions"

    async def generate(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        if not self.api_key:
            yield "Error: GROQ_API_KEY not found."
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_completion_tokens": 4096
        }

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", self.url, headers=headers, json=payload, timeout=60.0) as response:
                    if response.status_code != 200:
                        error_body = ""
                        async for chunk in response.aiter_bytes(): error_body += chunk.decode()
                        yield f"Groq Error {response.status_code}: {error_body}"
                        return

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]": break
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    content = data["choices"][0].get("delta", {}).get("content", "")
                                    if content: yield content
                            except json.JSONDecodeError: pass
        except Exception as e:
            yield f"Groq Connection Error: {e}"

# --- LLM Service ---
class LLMService:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.backend = self._get_backend()
        self.system_prompt = (
            "You are a smart, adaptable voice assistant. "
            "TONE ADAPTATION: Match the user's tone. If they speak formally, be formal. "
            "If they use casual slang like 'bhai', 'yar', 'bro', 'mate', 'dude', 'amigo', 'hermano', be warm, witty, and use similar casual language. "
            "LANGUAGE DETECTION: The user might speak 'Hinglish' (Hindi written in English script), 'Spanglish' (Spanish mixed with English), 'Franglais' (French mixed with English), or other hybrid languages. "
            "Example: 'Mujah Bhuk Lagi' -> 'Mujhe Bhuk Lagi'. "
            "'Tengo hambre, bro' -> stay in Spanglish. "
            "'J'ai faim, mate' -> stay in Franglais. "
            "If you detect any hybrid language, reply in the SAME mixed language. "
            "Do NOT switch to pure English if the user is speaking a hybrid language. "
            "CONTEXT: If the user asks about food/hunger in a casual way (e.g. 'kya khau', 'qué como', 'qu'est-ce que je mange'), suggest culturally relevant comfort food (e.g. 'Aloo Paratha', 'Biryani', 'tacos', 'croissant') with a friendly tone. "
            "Answer questions directly. No fluff. "
            f"Current Date: {datetime.now().strftime('%Y-%m-%d')}."
        )

    def _get_backend(self) -> LLMBackend:
        if self.provider == "groq":
            print(f"Initializing LLM Backend: Groq ({settings.GROQ_MODEL})")
            return GroqBackend(settings.GROQ_API_KEY, settings.GROQ_MODEL)
        else:
            print(f"Initializing LLM Backend: Ollama (llama3)")
            return OllamaBackend("llama3")

    async def generate_response(self, prompt: str, history: list = None, user_profile: str = "", language: str = "en-US") -> AsyncGenerator[str, None]:
        current_system = self.system_prompt
        
        # Language Instruction - Allow overrides for Hinglish
        if language == "en-US":
            current_system += (
                "\nNOTE: Default language is English. "
                "BUT if the user input looks like Hindi/Hinglish, IGNORE the default. "
                "Instead, reply in Hindi/Hinglish and START your response with '~hi~'. "
                "Example: '~hi~ Aloo Paratha kha le bhai.' "
                "If replying in English, do NOT add any tag."
            )
        else:
            current_system += f"\nIMPORTANT: Reply in {language}."
        
        if user_profile:
            current_system += f"\n\nUSER KNOWLEDGE GRAPH:\n{user_profile}"
        
        messages = [{'role': 'system', 'content': current_system}]
        if history: messages.extend(history)
        messages.append({'role': 'user', 'content': prompt})
        
        async for token in self.backend.generate(messages):
            yield token

llm_service = LLMService()

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

# --- FastAPI App ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    db_service.connect()

@app.on_event("shutdown")
def shutdown_event():
    db_service.close()

@app.get("/languages")
async def get_languages():
    return language_manager.languages

async def process_audio_stream(queue: asyncio.Queue, websocket: WebSocket):
    while True:
        item = await queue.get()
        if item is None: break
        
        text, lang = item
        if not text.strip(): continue

        try:
            audio_bytes = await tts_service.speak(text, lang)
            if audio_bytes:
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                await websocket.send_text(json.dumps({"type": "audio", "data": audio_b64}))
        except Exception as e:
            print(f"Streaming TTS Error: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    tts_queue = asyncio.Queue()
    tts_task = asyncio.create_task(process_audio_stream(tts_queue, websocket))
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError: continue
            
            if message.get("type") == "query":
                prompt = message.get("text", "")
                language = message.get("language", "en-US")
                
                if not prompt: continue
                
                full_response = ""
                sentence_buffer = ""
                detected_output_language = language
                checking_language_tag = True
                tag_buffer = ""
                
                await websocket.send_text(json.dumps({"type": "status", "message": "Thinking..."}))

                async for token in llm_service.generate_response(prompt, language=language):
                    # Dynamic Language Detection
                    if checking_language_tag:
                        tag_buffer += token
                        if len(tag_buffer) > 10:
                            checking_language_tag = False
                            if "~hi~" in tag_buffer:
                                detected_output_language = "hi-IN"
                                print("Detected Hindi (~hi~)")
                                tag_buffer = tag_buffer.replace("~hi~", "")
                            
                            full_response += tag_buffer
                            sentence_buffer += tag_buffer
                            await websocket.send_text(json.dumps({"type": "llm_token", "text": tag_buffer}))
                        else:
                            continue
                    else:
                        full_response += token
                        sentence_buffer += token
                        await websocket.send_text(json.dumps({"type": "llm_token", "text": token}))
                    
                    # Sentence splitting for TTS
                    sentences = re.split(r'(?<=[.!?])\s+', sentence_buffer)
                    if len(sentences) > 1:
                        for s in sentences[:-1]:
                            if s.strip():
                                await tts_queue.put((s, detected_output_language))
                        sentence_buffer = sentences[-1]

                # Flush remaining
                if checking_language_tag and tag_buffer:
                    if "~hi~" in tag_buffer:
                        detected_output_language = "hi-IN"
                        tag_buffer = tag_buffer.replace("~hi~", "")
                    full_response += tag_buffer
                    sentence_buffer += tag_buffer
                
                if sentence_buffer.strip():
                    await tts_queue.put((sentence_buffer, detected_output_language))
                
                await websocket.send_text(json.dumps({"type": "response_complete"}))
                
                # Log conversation (simplified)
                await db_service.log_conversation(prompt, full_response, settings.LLM_PROVIDER)

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")
    finally:
        await tts_queue.put(None)
        await tts_task

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
