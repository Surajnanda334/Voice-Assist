import os
import re
import json
import time
import asyncio
import io
import base64
import logging
import tempfile
import requests
from typing import AsyncGenerator, Optional, List, Dict
from datetime import datetime
from abc import ABC, abstractmethod

# Third-party imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from motor.motor_asyncio import AsyncIOMotorClient
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Optional libraries
try:
    import ollama
except ImportError:
    ollama = None
    print("Warning: 'ollama' not installed. LLM features will fail.")

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    from kittentts import KittenTTS
except Exception:
    KittenTTS = None

try:
    import edge_tts
except ImportError:
    edge_tts = None

try:
    import vosk
except ImportError:
    vosk = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


import httpx

# -----------------------------------------------------------------------------
# 1. CONFIGURATION & LANGUAGE MANAGER
# -----------------------------------------------------------------------------
load_dotenv(override=True)

class LanguageManager:
    def __init__(self, config_path: str = "voice_config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.default_language = self.config.get("default_language", "en-US")
        self.languages = self.config.get("languages", {})
        print(f"LanguageManager initialized. Default: {self.default_language}, Loaded {len(self.languages)} languages.")

    def _load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading voice_config.json: {e}")
                return {}
        print("Warning: voice_config.json not found.")
        return {}

    def get_voice(self, language_code: str, provider: str = "edge_tts") -> str:
        # Resolve language code (handle "auto" or empty)
        lang = self.validate_language(language_code)
        
        lang_config = self.languages.get(lang)
        if not lang_config:
            # Fallback to default
            lang_config = self.languages.get(self.default_language)
            
        if not lang_config:
            return "en-US-AriaNeural" # Hard fallback
            
        if provider == "edge_tts":
            return lang_config.get("tts_voice", "en-US-AriaNeural")
        
        return ""

    def validate_language(self, language_code: str) -> str:
        if not language_code or language_code == "auto":
            return self.default_language
        if language_code in self.languages:
            return language_code
        # Try finding by prefix (e.g. "hi" -> "hi-IN")
        for code in self.languages:
            if code.startswith(language_code.split('-')[0]):
                return code
        return self.default_language

    async def save_language_preference(self, user_id: str, language_code: str):
        """Save long-term language preference to Neo4j."""
        try:
            if 'graph_service' in globals() and graph_service:
                graph_service.add_preference(user_id, "PREFERS_LANGUAGE", "Preference", language_code)
        except Exception as e:
            print(f"Failed to save language preference: {e}")

language_manager = LanguageManager()

class Settings:
    # Database (MongoDB for Logs/History)
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "voice_assist_db")
    
    # Graph Database (Neo4j for Long-Term Memory)
    NEO4J_URI: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    NEO4J_USER: str = os.getenv("NEO4J_USER", "neo4j")
    NEO4J_PASSWORD: str = os.getenv("NEO4J_PASSWORD", "password")

    # API Keys & Secrets
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY", "").strip()
    if not SERPER_API_KEY:
        print("Warning: SERPER_API_KEY is empty or missing in .env")
    else:
        print(f"SERPER_API_KEY loaded (starts with: {SERPER_API_KEY[:4]}...)")
    
    # Providers
    TTS_PROVIDER: str = os.getenv("TTS_PROVIDER", "kitten").lower().strip()
    STT_PROVIDER: str = os.getenv("STT_PROVIDER", "vosk").lower().strip()
    SEARCH_PROVIDER: str = os.getenv("SEARCH_PROVIDER", "serper").lower().strip()
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama").lower().strip()
    
    # Models
    SOPRANO_MODEL_PATH: str = os.getenv("SOPRANO_MODEL_PATH", "parler-tts/parler-tts-large-v1")
    VOSK_MODEL_PATH: str = os.getenv("VOSK_MODEL_PATH", "model").strip()
    
    # Groq Configuration
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

    def validate(self):
        missing = []
        if not self.NEO4J_URI: missing.append("NEO4J_URI")
        
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        if self.TTS_PROVIDER not in ["kitten", "soprano"]:
             print(f"Warning: Invalid TTS_PROVIDER '{self.TTS_PROVIDER}'. Defaulting to 'kitten'.")
             self.TTS_PROVIDER = "kitten"

        if self.SEARCH_PROVIDER not in ["serper"]:
             print(f"Warning: Invalid SEARCH_PROVIDER '{self.SEARCH_PROVIDER}'. Defaulting to 'serper'.")
             self.SEARCH_PROVIDER = "serper"
        
        if self.LLM_PROVIDER == "groq" and not self.GROQ_API_KEY:
            print("Warning: LLM_PROVIDER is 'groq' but GROQ_API_KEY is missing. Fallback to ollama might fail if ollama is not running.")

settings = Settings()
settings.validate()


# -----------------------------------------------------------------------------
# 2. INTERFACES
# -----------------------------------------------------------------------------

class STTProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio_data: bytes, language: str = "auto") -> dict:
        """Convert audio bytes to text. Returns {'text': str, 'language': str}"""
        pass


class VoskSTTProvider(STTProvider):
    def __init__(self):
        self.model = None
        self.recognizers = {}  # Cache recognizers per language if needed, or just one
        
        if not vosk:
            print("Warning: 'vosk' package not installed. STT will fail.")
            return

        model_path = settings.VOSK_MODEL_PATH
        # Try to load model from path, or auto-download if simple name provided
        try:
            if os.path.exists(model_path):
                print(f"Loading Vosk model from: {model_path}")
                self.model = vosk.Model(model_path)
            else:
                # If path doesn't exist, assume it's a language code (e.g. "en-us")
                # vosk.Model(lang="en-us") downloads automatically
                print(f"Vosk model path '{model_path}' not found. Attempting to load by language name...")
                self.model = vosk.Model(lang="en-us") # Default to English for now as base
                
            print("Vosk STT initialized successfully.")
        except Exception as e:
            print(f"Vosk initialization failed: {e}")

    async def transcribe(self, audio_data: bytes, language: str = "auto") -> dict:
        if not self.model:
            return {"text": "", "language": ""}
        
        # Note: Vosk usually requires a recognizer with specific sample rate.
        # We assume 16kHz for now.
        rec = vosk.KaldiRecognizer(self.model, 16000)
        
        text = ""
        # rec.AcceptWaveform accepts bytes
        if rec.AcceptWaveform(audio_data):
            result = json.loads(rec.Result())
            text = result.get("text", "")
        else:
            # Partial result?
            result = json.loads(rec.FinalResult())
            text = result.get("text", "")
            
        # Determine language code
        detected_lang = language_manager.validate_language(language)
        
        return {"text": text, "language": detected_lang}


class TTSProvider(ABC):
    @abstractmethod
    async def speak(self, text: str, language: str = "en-US", voice: str = None) -> bytes:
        """Convert text to speech audio bytes."""
        pass

    def clean_text(self, text: str) -> str:
        """Removes Markdown formatting (asterisks, links) for smoother TTS."""
        # 1. Remove bold/italic/bullets (* or **)
        text = re.sub(r'^\s*[\*]+\s+', '', text)
        text = re.sub(r'\n\s*[\*]+\s+', '\n', text)
        text = text.replace("*", "")
        
        # 2. Handle Markdown Links: [Title](URL) -> Title
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # 3. Handle Bare Links: [URL] -> "Link" or remove
        text = re.sub(r'\[http[^\]]+\]', '', text)
        text = re.sub(r'\[www\.[^\]]+\]', '', text)
        
        # 4. Remove http://... if it's in plain text
        text = re.sub(r'https?://\S+', '', text)
        
        return text.strip()

class SearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str) -> str:
        """Perform a web search and return a formatted string context."""
        pass


# -----------------------------------------------------------------------------
# 3. IMPLEMENTATIONS (TTS & Search)
# -----------------------------------------------------------------------------

class KittenProvider(TTSProvider):
    def __init__(self):
        self.kitten_model = None
        if KittenTTS:
            try:
                self.kitten_model = KittenTTS("KittenML/kitten-tts-nano-0.1")
                print("KittenTTS initialized successfully.")
            except Exception as e:
                print(f"KittenTTS initialization failed: {e}")
        
        self.use_edge_tts = True if edge_tts else False
        self.use_pyttsx3 = True if pyttsx3 else False

    async def speak(self, text: str, language: str = "en-US", voice: str = None) -> bytes:
        text = self.clean_text(text)
        if not text:
            return b""

        # 1. Try KittenTTS (Only if English and no specific voice requested)
        # KittenTTS is English-only for now (nano-0.1)
        if self.kitten_model and sf and language.startswith("en") and not voice:
            try:
                print(f"Generating TTS with KittenTTS: {text[:30]}...")
                kitten_start = time.time()
                audio = await run_in_threadpool(self.kitten_model.generate, text, voice='expr-voice-2-f')
                
                with io.BytesIO() as wav_buffer:
                    sf.write(wav_buffer, audio, 24000, format='WAV')
                    wav_buffer.seek(0)
                    data = wav_buffer.read()
                    print(f"KittenTTS generation took: {time.time() - kitten_start:.4f}s")
                    return data
            except Exception as e:
                print(f"KittenTTS error: {e}")

        # 2. Try Edge TTS (Primary for Multi-Language)
        if self.use_edge_tts:
            try:
                edge_start = time.time()
                # Determine voice: Explicit > Config > Default
                if not voice:
                    voice = language_manager.get_voice(language, "edge_tts")
                
                print(f"Generating TTS with EdgeTTS ({voice}): {text[:30]}...")
                communicate = edge_tts.Communicate(text, voice)
                audio_data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]
                print(f"EdgeTTS generation took: {time.time() - edge_start:.4f}s")
                return audio_data
            except Exception as e:
                print(f"EdgeTTS failed: {e}")

        # 3. Fallback to pyttsx3
        if self.use_pyttsx3:
            try:
                pyttsx3_start = time.time()
                data = await run_in_threadpool(self._pyttsx3_gen, text)
                print(f"pyttsx3 generation took: {time.time() - pyttsx3_start:.4f}s")
                return data
            except Exception as e:
                print(f"pyttsx3 failed: {e}")
            
        print("CRITICAL: All TTS services failed to generate audio.")
        return b""

    def _pyttsx3_gen(self, text):
        engine = pyttsx3.init()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
        engine.save_to_file(text, temp_path)
        engine.runAndWait()
        with open(temp_path, "rb") as f:
            data = f.read()
        try:
            os.remove(temp_path)
        except:
            pass
        return data


SOPRANO_DESCRIPTION = """You are a calm, clear, and natural-sounding voice assistant. 
Your speech should be conversational, confident, and human-like. 
Avoid robotic cadence. Pause naturally between thoughts. 
Emphasize clarity over speed. 
Adapt tone based on content: 
- Technical explanations: calm and precise 
- Casual responses: friendly and relaxed 
- Instructions: clear and well-paced 
Assume the listener is the same user every time."""

class SopranoProvider(TTSProvider):
    def __init__(self):
        self.model = None
        if KittenTTS:
            try:
                self.model = KittenTTS(settings.SOPRANO_MODEL_PATH)
                print(f"Soprano TTS initialized with model: {settings.SOPRANO_MODEL_PATH}")
            except Exception as e:
                print(f"Soprano TTS initialization failed: {e}")
        else:
             print("Warning: 'kittentts' package not installed. Soprano TTS unavailable.")

    async def speak(self, text: str, language: str = "en-US", voice: str = None) -> bytes:
        text = self.clean_text(text)
        if not text or not self.model or not sf:
            return b""

        try:
            print(f"Generating TTS with Soprano: {text[:30]}...")
            start = time.time()
            audio = await run_in_threadpool(self.model.generate, text, voice=SOPRANO_DESCRIPTION)
            
            with io.BytesIO() as wav_buffer:
                sf.write(wav_buffer, audio, 24000, format='WAV')
                wav_buffer.seek(0)
                data = wav_buffer.read()
                print(f"Soprano generation took: {time.time() - start:.4f}s")
                return data
        except Exception as e:
            print(f"Soprano TTS error: {e}")
            return b""


class SerperProvider(SearchProvider):
    def __init__(self):
        self.api_key = settings.SERPER_API_KEY
        self.url = "https://google.serper.dev/search"

    async def search(self, query: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._search_sync, query)

    def _search_sync(self, query: str) -> str:
        if not self.api_key:
            return "Error: SERPER_API_KEY not found."
            
        payload = json.dumps({"q": query, "num": 3})
        headers = {'X-API-KEY': self.api_key, 'Content-Type': 'application/json'}

        try:
            search_start = time.time()
            print(f"[{search_start:.3f}] Serper Search Start: {query}")
            response = requests.request("POST", self.url, headers=headers, data=payload)
            response.raise_for_status()
            data = response.json()
            
            search_duration = time.time() - search_start
            print(f"[{time.time():.3f}] Serper Search Complete. Time taken: {search_duration:.4f}s")
            
            snippets = []
            if "organic" in data:
                for result in data["organic"]:
                    title = result.get("title", "")
                    link = result.get("link", "")
                    snippet = result.get("snippet", "")
                    snippets.append(f"- Title: {title}\n  Link: {link}\n  Summary: {snippet}")
            
            if "knowledgeGraph" in data:
                kg = data["knowledgeGraph"]
                title = kg.get("title", "")
                desc = kg.get("description", "")
                snippets.insert(0, f"Knowledge Graph: {title} - {desc}")
                
            return "\n".join(snippets)
            
        except Exception as e:
            print(f"Serper Search Error: {e}")
            return f"Error performing search: {e}"


class TextExpansionService:
    def __init__(self, model: str = "llama3"):
        self.model = model

    async def expand_text(self, text: str, language: str = "en-US") -> str:
        """Expands short text to be more conversational and at least 10 words."""
        # Simple word count check
        if len(text.split()) >= 10:
            return text

        if not ollama:
            return text

        print(f"Expanding short text: '{text}' ({language})")
        prompt = (
            f"You are a helpful assistant speaking {language}. "
            f"Rewrite the following text to be more conversational and at least 10 words long, "
            f"but keep the exact same meaning. Do not add new facts. "
            f"Original text: '{text}'"
        )
        
        try:
            response = await run_in_threadpool(
                lambda: ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
            )
            expanded = response['message']['content'].replace('"', '').strip()
            print(f"Expanded to: '{expanded}'")
            return expanded
        except Exception as e:
            print(f"Text expansion failed: {e}")
            return text

text_expansion_service = TextExpansionService()

class TTSValidationService:
    def __init__(self):
        pass
        
    async def validate_and_fix(self, text: str, language: str) -> str:
        """
        Validates text for TTS:
        1. Checks length (triggers expansion if too short).
        2. Validates language support (handled by LanguageManager mostly).
        """
        # 1. Expand if too short
        text = await text_expansion_service.expand_text(text, language)
        return text

tts_validation_service = TTSValidationService()


# -----------------------------------------------------------------------------
# 4. CORE SERVICES (Database, Graph, Memory, LLM, Orchestrator)
# -----------------------------------------------------------------------------

class DatabaseService:
    """Handles MongoDB connections for chat history and logs ONLY."""
    client: AsyncIOMotorClient = None
    db = None

    def connect(self):
        try:
            self.client = AsyncIOMotorClient(settings.MONGO_URI)
            self.db = self.client[settings.MONGO_DB_NAME]
            print(f"Connected to MongoDB at {settings.MONGO_URI}")
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")

    def close(self):
        if self.client:
            self.client.close()
            print("MongoDB connection closed")

    async def log_conversation(self, query: str, response: str, route: str, search_query: str = None, serper_response: str = None):
        if self.db is None: return
        document = {
            "timestamp": datetime.now(),
            "query": query,
            "response": response,
            "route": route,
            "search_query": search_query,
            "serper_response": serper_response
        }
        try:
            await self.db.conversations.insert_one(document)
            print("Conversation logged to MongoDB")
        except Exception as e:
            print(f"Error logging to MongoDB: {e}")

    async def get_recent_conversations(self, limit: int = 10):
        if self.db is None: return []
        try:
            cursor = self.db.conversations.find().sort("timestamp", -1).limit(limit)
            conversations = await cursor.to_list(length=limit)
            return conversations[::-1] 
        except Exception as e:
            print(f"Error fetching conversations: {e}")
            return []

class GraphService:
    """Handles Neo4j connections for User Preferences (Long-Term Memory)."""
    driver = None

    def connect(self):
        try:
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI, 
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
            )
            self.driver.verify_connectivity()
            print(f"Connected to Neo4j at {settings.NEO4J_URI}")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed")

    def add_preference(self, user_id: str, relation: str, label: str, value: str):
        if not self.driver: return
        try:
            with self.driver.session() as session:
                session.execute_write(self._create_preference_tx, user_id, relation.upper(), label, value)
                print(f"Graph Update: (User)-[:{relation.upper()}]->(:{label} {{name: '{value}'}})")
        except Exception as e:
            print(f"Graph Write Error: {e}")

    @staticmethod
    def _create_preference_tx(tx, user_id, relation, label, value):
        # 1. Merge User
        tx.run("MERGE (u:User {userId: $userId})", userId=user_id)
        
        # 2. Merge Preference Node (Dynamic Label hack via APOC or explicit checks? 
        # Cypher doesn't allow dynamic labels in MERGE easily without APOC.
        # We'll use a generic way or just string interpolation if we trust the source (internal only).
        # Since 'label' comes from our controlled extraction prompt, we'll sanitize it.
        valid_labels = ["Interest", "Topic", "Entity", "Trait", "Preference"]
        safe_label = label if label in valid_labels else "Interest"
        
        # Using string formatting for Label is necessary in standard drivers for Cypher 
        # but requires strict validation to prevent injection.
        query = (
            f"MERGE (p:{safe_label} {{name: $value}}) "
            f"MERGE (u:User {{userId: $userId}})-[r:{relation}]->(p) "
            f"SET r.lastUpdated = datetime()"
        )
        
        tx.run(query, userId=user_id, value=value)

    def get_user_preferences(self, user_id: str) -> List[str]:
        if not self.driver: return []
        try:
            with self.driver.session() as session:
                return session.execute_read(self._get_preferences_tx, user_id)
        except Exception as e:
            print(f"Graph Read Error: {e}")
            return []

    @staticmethod
    def _get_preferences_tx(tx, user_id):
        query = """
        MATCH (u:User {userId: $userId})-[r]->(p)
        RETURN type(r) as relation, labels(p) as labels, p.name as value
        ORDER BY r.lastUpdated DESC LIMIT 20
        """
        result = tx.run(query, userId=user_id)
        facts = []
        for record in result:
            rel = record["relation"]
            lbl = record["labels"][0] if record["labels"] else "Thing"
            val = record["value"]
            facts.append(f"{rel} {val} ({lbl})")
        return facts

db_service = DatabaseService()
graph_service = GraphService()


class MemoryService:
    def __init__(self, model: str = "llama3"):
        self.model = model

    async def extract_and_save(self, query: str):
        if not ollama: return
        # Simple heuristic to trigger extraction
        if not re.search(r"\b(i am|my name is|i like|i love|i hate|i prefer|i want|i enjoy)\b", query, re.IGNORECASE):
            return

        # Structured prompt for Graph extraction
        prompt = (
            f"Analyze this input: '{query}'. "
            "Extract user preferences for a Knowledge Graph. "
            "Format: RELATION|LABEL|VALUE\n"
            "RELATION must be one of: LIKES, DISLIKES, PREFERS, IS, WANTS.\n"
            "LABEL must be one of: Interest, Topic, Entity, Trait.\n"
            "Example output:\n"
            "LIKES|Interest|Coding\n"
            "IS|Trait|Developer\n"
            "Return ONLY the formatted lines. If no facts, return nothing."
        )
        try:
            response = await run_in_threadpool(
                lambda: ollama.chat(model=self.model, messages=[{'role': 'user', 'content': prompt}])
            )
            content = response['message']['content']
            
            for line in content.split('\n'):
                line = line.strip()
                parts = line.split('|')
                if len(parts) == 3:
                    relation, label, value = parts[0].strip(), parts[1].strip(), parts[2].strip()
                    # Validate
                    if relation in ["LIKES", "DISLIKES", "PREFERS", "IS", "WANTS"] and \
                       label in ["Interest", "Topic", "Entity", "Trait"]:
                        # Async handoff to graph service
                        await run_in_threadpool(graph_service.add_preference, "default", relation, label, value)
                        
        except Exception as e:
            print(f"Memory Extraction Error: {e}")

memory_service = MemoryService()


class LLMBackend(ABC):
    @abstractmethod
    async def generate(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        pass

class OllamaBackend(LLMBackend):
    def __init__(self, model: str):
        self.model = model

    async def generate(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        if not ollama:
            yield "Error: Ollama not installed."
            return
        try:
            # Run blocking ollama call in threadpool if strictly needed, 
            # but ollama's stream iterator is synchronous.
            # Ideally we should run the iterator in a thread, but for now we iterate directly
            # assuming it doesn't block the loop too badly (it usually does HTTP requests).
            # To be safe, we wrap the creation in run_in_threadpool.
            
            stream = await run_in_threadpool(
                lambda: ollama.chat(model=self.model, messages=messages, stream=True, keep_alive='5m')
            )
            
            for chunk in stream:
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
        # Filter out system messages if Groq doesn't like them? No, Groq supports system messages.
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "temperature": 0.7,
            "max_completion_tokens": 4096
        }
        
        async with httpx.AsyncClient() as client:
            try:
                async with client.stream("POST", self.url, headers=headers, json=payload, timeout=60.0) as response:
                    if response.status_code != 200:
                        error_text = await response.read()
                        yield f"Error from Groq: {response.status_code} - {error_text.decode()}"
                        return
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                break
                            try:
                                data = json.loads(data_str)
                                if "choices" in data and len(data["choices"]) > 0:
                                    delta = data["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield content
                            except json.JSONDecodeError:
                                pass
            except Exception as e:
                 yield f"Error connecting to Groq: {e}"


class LLMService:
    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.backend = self._get_backend()
        self.system_prompt = (
            "You are a smart, adaptable voice assistant. "
            "TONE ADAPTATION: Match the user's tone. If they speak formally, be formal. "
            "If they use casual slang like 'bhai', 'yar', 'bro', be warm, witty, and use similar casual language. "
            "LANGUAGE DETECTION: The user might speak 'Hinglish' (Hindi written in English script). "
            "Example: 'Mujah Bhuk Lagi' -> 'Mujhe Bhuk Lagi'. "
            "If you detect Hinglish or Hindi words, reply in the SAME mixed language (Hinglish/Hindi). "
            "Do NOT switch to pure English if the user is speaking Hindi/Hinglish. "
            "CONTEXT: If the user asks about food/hunger in a casual way (e.g. 'kya khau'), suggest culturally relevant comfort food (e.g. 'Aloo Paratha', 'Biryani') with a friendly tone. "
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
        try:
            full_prompt = f"{prompt}"
            
            current_system = self.system_prompt
            
            # Language Instruction - Allow overrides for Hinglish
            if language == "en-US":
                # Weak constraint: Allow LLM to detect Hinglish
                current_system += (
                    "\nNOTE: Default language is English. "
                    "BUT if the user input looks like Hindi/Hinglish, IGNORE the default. "
                    "Instead, reply in Hindi/Hinglish and START your response with '~hi~'. "
                    "Example: '~hi~ Aloo Paratha kha le bhai.' "
                    "If replying in English, do NOT add any tag."
                )
            else:
                # Strong constraint
                current_system += f"\nIMPORTANT: Reply in {language}."
            
            if user_profile:
                current_system += f"\n\nUSER KNOWLEDGE GRAPH:\n{user_profile}"
            
            messages = [{'role': 'system', 'content': current_system}]
            if history:
                messages.extend(history)
            
            messages.append({'role': 'user', 'content': full_prompt})
            
            async for token in self.backend.generate(messages):
                yield token
            
            print(f"[{time.time():.3f}] LLM Request Complete.")
        except Exception as e:
            print(f"LLM Error: {e}")
            yield f"Error generating response: {e}"

llm_service = LLMService()


class OrchestratorService:
    def __init__(self):
        self.search_provider = self._get_provider()

    def _get_provider(self):
        print(f"Initializing Search Provider: {settings.SEARCH_PROVIDER}")
        if settings.SEARCH_PROVIDER == "serper":
            return SerperProvider()
        print(f"Warning: Unknown SEARCH_PROVIDER '{settings.SEARCH_PROVIDER}'")
        return None

    async def determine_route(self, query: str):
        # Expanded regex for better detection
        if re.search(r"\b(search|google|find|lookup|who|what|when|where|how|why|news|weather)\b", query, re.IGNORECASE):
            return "search"
        return "chat"

    async def perform_search(self, query: str) -> str:
        if not self.search_provider:
            return "Error: Search provider not available."
        
        clean_query = re.sub(r"\b(search for|google|find|lookup)\b", "", query, flags=re.IGNORECASE).strip()
        if not clean_query: clean_query = query
        
        return await self.search_provider.search(clean_query)

orchestrator_service = OrchestratorService()


# -----------------------------------------------------------------------------
# 5. FASTAPI APPLICATION
# -----------------------------------------------------------------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize TTS Provider
def get_tts_provider_instance():
    if settings.TTS_PROVIDER == "kitten":
        return KittenProvider()
    elif settings.TTS_PROVIDER == "soprano":
        return SopranoProvider()
    return None

try:
    tts_provider = get_tts_provider_instance()
    print(f"TTS Provider Initialized: {settings.TTS_PROVIDER}")
except Exception as e:
    print(f"TTS Provider Init Error: {e}")
    tts_provider = None


@app.get("/languages")
async def get_languages():
    return language_manager.languages


@app.on_event("startup")
async def startup_event():
    db_service.connect()
    graph_service.connect()

@app.on_event("shutdown")
def shutdown_event():
    db_service.close()
    graph_service.close()

async def process_audio_stream(queue: asyncio.Queue, websocket: WebSocket, tts_provider: TTSProvider):
    """Consumes (text, language) tuples from the queue and sends audio chunks to the websocket."""
    while True:
        item = await queue.get()
        if item is None:  # Sentinel value to stop
            break
        
        text, lang = item
        
        if not text.strip():
            continue

        try:
            # Validate and fix text (expand if needed)
            validated_text = await tts_validation_service.validate_and_fix(text, lang)
            
            # Generate audio for the sentence
            audio_bytes = await tts_provider.speak(validated_text, lang)
            if audio_bytes:
                audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
                await websocket.send_text(json.dumps({"type": "audio", "data": audio_b64}))
        except Exception as e:
            print(f"Streaming TTS Error: {e}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    
    session_context = [] 
    
    # Load recent history from MongoDB
    recent_history = await db_service.get_recent_conversations(limit=10)
    for conv in recent_history:
        session_context.append({"role": "user", "content": conv["query"]})
        session_context.append({"role": "assistant", "content": conv["response"]})
    
    # Load User Profile from Neo4j Graph
    user_preferences = graph_service.get_user_preferences("default")
    user_profile_str = "\n".join([f"- {pref}" for pref in user_preferences])
    if user_profile_str:
        print(f"Loaded Graph Memory:\n{user_profile_str}")

    try:
        while True:
            data_str = await websocket.receive_text()
            print(f"Received: {data_str}")
            
            # Parse incoming JSON if possible
            user_text = data_str
            language = "en-US"
            try:
                data_json = json.loads(data_str)
                if isinstance(data_json, dict):
                    if data_json.get("type") == "stop":
                        continue
                    if "text" in data_json:
                        user_text = data_json["text"]
                    if "language" in data_json:
                        language = language_manager.validate_language(data_json["language"])
                        # Save language preference asynchronously
                        asyncio.create_task(language_manager.save_language_preference("default", language))
            except json.JSONDecodeError:
                pass
            
            # 1. Memory Extraction (Graph)
            asyncio.create_task(memory_service.extract_and_save(user_text))
            
            # 2. Routing
            route = await orchestrator_service.determine_route(user_text)
            print(f"Route determined: {route} (Language: {language})")
            
            search_context = ""
            serper_response = None
            search_query = None
            
            if route == "search":
                await websocket.send_text(json.dumps({"type": "status", "message": "Searching..."}))
                search_query = user_text
                search_context = await orchestrator_service.perform_search(user_text)
                serper_response = search_context
                if search_context and not search_context.startswith("Error"):
                    search_context = f"Context from Web Search:\n{search_context}\n\n"
            
            # 3. LLM Generation & Streaming TTS
            history_for_llm = [{"role": m["role"], "content": m["content"]} for m in session_context]
            prompt = user_text
            if search_context:
                prompt = f"{search_context}User Query: {user_text}"
            
            await websocket.send_text(json.dumps({"type": "status", "message": "Thinking..."}))
            
            full_response = ""
            sentence_buffer = ""
            tts_queue = asyncio.Queue()
            tts_task = None
            
            if tts_provider:
                # Start the consumer task
                tts_task = asyncio.create_task(process_audio_stream(tts_queue, websocket, tts_provider))
                await websocket.send_text(json.dumps({"type": "status", "message": "Speaking..."}))

            # Detection State
            detected_output_language = language # Default to session language
            checking_language_tag = True
            tag_buffer = ""

            async for token in llm_service.generate_response(prompt, history=history_for_llm, user_profile=user_profile_str, language=language):
                
                # Dynamic Language Detection Logic
                if checking_language_tag:
                    tag_buffer += token
                    if len(tag_buffer) > 10: # Look at first 10 chars
                        checking_language_tag = False
                        if "~hi~" in tag_buffer:
                            detected_output_language = "hi-IN"
                            print("Output Language Switch: Detected Hindi (~hi~)")
                            # Remove tag from output
                            full_response += tag_buffer.replace("~hi~", "")
                            # Also clean buffer for TTS
                            sentence_buffer += tag_buffer.replace("~hi~", "")
                        else:
                            full_response += tag_buffer
                            sentence_buffer += tag_buffer
                    else:
                        # Buffering... don't send to TTS yet
                        continue 
                else:
                    full_response += token
                    sentence_buffer += token
                
                await websocket.send_text(json.dumps({"type": "llm_token", "text": token}))
                
                # Streaming TTS Logic
                if tts_provider:
                    # Split by sentence endings (. ! ?) followed by space or newline
                    sentences = re.split(r'(?<=[.!?])\s+', sentence_buffer)
                    if len(sentences) > 1:
                        # We have at least one complete sentence
                        for s in sentences[:-1]:
                            if s.strip():
                                await tts_queue.put((s, detected_output_language))
                        # Keep the last incomplete part
                        sentence_buffer = sentences[-1]
            
            # Process remaining buffer
            if checking_language_tag and tag_buffer:
                 # Flush buffer if response was very short
                 if "~hi~" in tag_buffer:
                     detected_output_language = "hi-IN"
                     tag_buffer = tag_buffer.replace("~hi~", "")
                 full_response += tag_buffer
                 sentence_buffer += tag_buffer

            if tts_provider:
                if sentence_buffer.strip():
                    await tts_queue.put((sentence_buffer, detected_output_language))
                # Signal end of stream
                await tts_queue.put(None)
                # Wait for all audio to be sent
                await tts_task

            # 4. Update Context
            session_context.append({"role": "user", "content": user_text})
            session_context.append({"role": "assistant", "content": full_response})
            
            # 5. Log to DB
            asyncio.create_task(db_service.log_conversation(
                query=user_text, 
                response=full_response, 
                route=route, 
                search_query=search_query,
                serper_response=serper_response
            ))
            
            await websocket.send_text(json.dumps({"type": "status", "message": "Idle", "done": True}))

            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
