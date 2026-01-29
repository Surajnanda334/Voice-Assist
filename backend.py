import os
import io
import time
import json
import wave
import base64
import asyncio
import subprocess
import requests
import re
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime
from typing import AsyncGenerator, Optional

# Third-party imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Optional imports (handle if missing)
try:
    import ollama
except ImportError:
    ollama = None
    print("Warning: 'ollama' package not installed. LLM features will fail.")

try:
    import edge_tts
except ImportError:
    edge_tts = None
    print("Warning: 'edge_tts' package not installed. EdgeTTS will fail.")

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
    print("Warning: 'pyttsx3' package not installed. Offline TTS will fail.")

# Load environment variables
load_dotenv()

# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "voice_assist_db")
SERPER_API_KEY = os.getenv("SERPER_API_KEY") # Default key from previous file

# --- Database Service ---
class DatabaseService:
    client: AsyncIOMotorClient = None
    db = None

    def connect(self):
        try:
            self.client = AsyncIOMotorClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            print(f"Connected to MongoDB at {MONGO_URI}")
        except Exception as e:
            print(f"Failed to connect to MongoDB: {e}")

    def close(self):
        if self.client:
            self.client.close()
            print("MongoDB connection closed")

    async def log_conversation(self, query: str, response: str, route: str, search_query: str = None):
        if self.db is None:
            return
        
        document = {
            "timestamp": datetime.now(),
            "query": query,
            "response": response,
            "route": route,
            "search_query": search_query
        }
        
        try:
            await self.db.conversations.insert_one(document)
            print("Conversation logged to MongoDB")
        except Exception as e:
            print(f"Error logging to MongoDB: {e}")

db_service = DatabaseService()

# --- Search Service ---
class SearchService:
    def __init__(self):
        self.api_key = SERPER_API_KEY
        self.url = "https://google.serper.dev/search"

    def search(self, query: str):
        if not self.api_key:
            return "Error: SERPER_API_KEY not found."
            
        payload = json.dumps({
            "q": query,
            "num": 3 # Fetch top 3 results
        })
        headers = {
            'X-API-KEY': self.api_key,
            'Content-Type': 'application/json'
        }

        try:
            search_start = time.time()
            print(f"[{search_start:.3f}] Serper Search Start: {query}")
            response = requests.request("POST", self.url, headers=headers, data=payload)
            response.raise_for_status()
            data = response.json()
            
            search_duration = time.time() - search_start
            print(f"[{time.time():.3f}] Serper Search Complete. Time taken: {search_duration:.4f}s")
            
            # Parse results to be LLM-friendly
            snippets = []
            if "organic" in data:
                for result in data["organic"]:
                    title = result.get("title", "")
                    snippet = result.get("snippet", "")
                    snippets.append(f"- {title}: {snippet}")
            
            if "knowledgeGraph" in data:
                kg = data["knowledgeGraph"]
                title = kg.get("title", "")
                desc = kg.get("description", "")
                snippets.insert(0, f"Knowledge Graph: {title} - {desc}")
                
            return "\n".join(snippets)
            
        except Exception as e:
            print(f"Serper Search Error: {e}")
            return f"Error performing search: {e}"

# --- Orchestrator Service ---
class OrchestratorService:
    def __init__(self, model: str = "llama3"):
        self.model = model
        print(f"Orchestrator Service Initialized with model: {self.model}")
        self.system_prompt = f"""
You are an Orchestrator Agent.
Your responsibility is to decide whether a user query should be answered by:
1) a LOCAL LLM (LLaMA-3 – fast but outdated), or
2) SERPER (web search – real-time, up-to-date data).

Current Date: {datetime.now().strftime('%Y-%m-%d')}
Model Knowledge Cutoff: ~2023

You must NOT answer the user directly.
You must ONLY decide the routing.

────────────────────────
ROUTING RULES (STRICT)
────────────────────────

Route to SERPER if the query involves:
- "Who is..." questions about living people (Politicians, CEOs, Celebrities)
- Current Government Officials (President, Prime Minister, Kings, etc.)
- Recent History (Events after 2022)
- Current affairs, news, or weather
- Company, startup, or product updates
- Stock prices, revenue, valuation
- Sports results or schedules
- Prices, rankings, trends
- "Latest", "Current", "Today", "Now", "Recently"

Route to LOCAL_LLM if the query involves:
- Logic, reasoning, or explanations
- Math or calculations
- Programming, debugging, or system design
- Algorithms, data structures
- Definitions or timeless concepts (e.g., "What is democracy?")
- Fictional or hypothetical content
- Personal advice
- General knowledge that clearly hasn't changed in 10 years

────────────────────────
DECISION HEURISTIC
────────────────────────
Ask yourself: "Is it POSSIBLE that the answer has changed since 2023?"
If YES → SERPER
If NO → LOCAL_LLM

────────────────────────
OUTPUT FORMAT (MANDATORY)
────────────────────────
Return ONLY valid JSON.
{{
  "route": "LOCAL_LLM" | "SERPER",
  "reason": "short justification",
  "serper_query": "search query OR null"
}}
"""

    def _check_heuristics(self, query: str) -> dict | None:
        """
        Fast regex-based routing to skip LLM latency for obvious cases.
        """
        query_lower = query.lower()
        
        # 1. Obvious Search Triggers
        search_patterns = [
            r"\b(who|what) (is|was|are) the (current|present|latest|new)",
            r"\b(who|what) (is|was) (president|prime minister|ceo|cto|cfo)",
            r"\b(weather|temperature|forecast)",
            r"\b(news|headline|update)",
            r"\b(stock|price|market|valuation)",
            r"\b(when) (is|was|did) (the)",
            r"\b(latest|recent|today|now)",
            r"\b(schedule|score|result|winner)",
            r"\bwho is \w+",  # "Who is X?" almost always needs search for people
        ]
        
        for pattern in search_patterns:
            if re.search(pattern, query_lower):
                print(f"Orchestrator Heuristic Match: SERPER ({pattern})")
                return {
                    "route": "SERPER",
                    "reason": "Heuristic match for real-time info",
                    "serper_query": query
                }
        
        # 2. Obvious Local Triggers (Code, Math, Definitions, Chitchat)
        local_patterns = [
            r"\b(write|create|generate|fix|debug) (code|function|script|program|app)",
            r"\b(in python|in js|in typescript|in java|in c\+\+)",
            r"\b(solve|calculate|compute|math)",
            r"\b(explain|define|meaning of) (concept|term|word)",
            r"\b(tell me a joke|write a poem|write a story)",
            r"\b(hello|hi|hey|greetings|good morning|good afternoon|good evening)",
            r"\b(thank you|thanks|cool|okay|ok|great|awesome)",
            r"\b(how are you|what's up|how is it going)",
            r"\b(who are you|what are you)",
            r"\b(summarize|paraphrase|rewrite)",
        ]
        
        for pattern in local_patterns:
            if re.search(pattern, query_lower):
                print(f"Orchestrator Heuristic Match: LOCAL_LLM ({pattern})")
                return {
                    "route": "LOCAL_LLM",
                    "reason": "Heuristic match for creative/timeless task",
                    "serper_query": None
                }
                
        return None

    def route_query(self, query: str) -> dict:
        try:
            print(f"\n--- ORCHESTRATOR LOGS ---")
            print(f"Input Query: {query}")
            
            # 1. Fast Heuristic Check
            heuristic_decision = self._check_heuristics(query)
            if heuristic_decision:
                print(f"Fast Path Decision: {heuristic_decision}")
                print(f"-------------------------\n")
                return heuristic_decision
            
            # 2. Slow LLM Check
            llm_start = time.time()
            print(f"[{llm_start:.3f}] Orchestrator LLM Routing Check Start...")
            
            if not ollama:
                return {"route": "LOCAL_LLM", "reason": "Ollama not installed", "serper_query": None}

            response = ollama.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': query}
                ],
                stream=False,
                format='json', # Force JSON mode
                keep_alive='5m'
            )
            
            llm_duration = time.time() - llm_start
            content = response['message']['content'].strip()
            print(f"[{time.time():.3f}] Orchestrator LLM Response ({llm_duration:.3f}s): {content}")
            
            decision = json.loads(content)
            print(f"Parsed Decision: {decision}")
            print(f"-------------------------\n")
            
            return decision
            
        except Exception as e:
            print(f"Orchestrator Error: {e}")
            print(f"-------------------------\n")
            # Fallback to LOCAL_LLM on error
            return {"route": "LOCAL_LLM", "reason": f"Orchestrator failed: {e}", "serper_query": None}

# --- LLM Service ---
class LLMService:
    def __init__(self, model: str = "llama3"):
        self.model = model
        self.system_prompt = (
            "You are a helpful, concise voice assistant. "
            "Provide clear, direct answers in 30-50 words. "
            f"Current Date: {datetime.now().strftime('%Y-%m-%d')}. "
            "IMPORTANT: You may receive 'Context from Web Search'. "
            "If this context is provided, you MUST use it as the source of truth, "
            "even if it contradicts your internal training data. "
            "Never say 'As of my knowledge cutoff' if you have search context. "
            "Do NOT read out long disclaimers. Be conversational."
        )
        print(f"LLM Service Initialized with model: {self.model}")

    async def generate_response(self, prompt: str, history: list = None) -> AsyncGenerator[str, None]:
        if not ollama:
             yield "Error: Ollama not installed."
             return

        try:
            start_time = time.time()
            print(f"[{start_time:.3f}] LLM Request Start: {prompt[:50]}...")

            # Enforce brevity in the user prompt as well for better adherence
            full_prompt = f"{prompt} (Answer concisely in 30-50 words)"
            
            messages = [{'role': 'system', 'content': self.system_prompt}]
            if history:
                # Add history to context
                messages.extend(history)
            
            messages.append({'role': 'user', 'content': full_prompt})
            
            # Use 'keep_alive' to keep the model loaded in memory for subsequent requests
            stream = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True,
                keep_alive='5m' 
            )
            
            first_token_time = None
            token_count = 0
            
            for chunk in stream:
                if first_token_time is None:
                    first_token_time = time.time()
                    print(f"[{first_token_time:.3f}] LLM First Token received (Latency: {first_token_time - start_time:.3f}s)")
                
                if 'message' in chunk and 'content' in chunk['message']:
                    token = chunk['message']['content']
                    token_count += 1
                    yield token
            
            end_time = time.time()
            print(f"[{end_time:.3f}] LLM Request Complete. Total Time: {end_time - start_time:.3f}s, Tokens: {token_count}")
            
        except Exception as e:
            print(f"LLM Error: {e}")
            yield f"Error generating response: {e}"

# --- TTS Service ---
class TTSService:
    def __init__(self):
        self.use_edge_tts = True if edge_tts else False
        self.use_pyttsx3 = True if pyttsx3 else False
        
        # Try loading KittenTTS
        self.kitten_model = None
        try:
            from kittentts import KittenTTS
            # Initialize with default model
            self.kitten_model = KittenTTS("KittenML/kitten-tts-nano-0.1")
            print("KittenTTS initialized successfully.")
        except Exception as e:
            print(f"KittenTTS initialization failed (using fallback): {e}")

    async def speak(self, text: str) -> bytes:
        # 1. Try KittenTTS
        if self.kitten_model:
            try:
                print(f"Generating TTS with KittenTTS: {text[:30]}...")
                kitten_start = time.time()
                # Run in threadpool because it might be blocking
                audio = await run_in_threadpool(self.kitten_model.generate, text, voice='expr-voice-2-f')
                
                # Convert float32 numpy array to int16 WAV bytes
                import io
                import soundfile as sf
                
                with io.BytesIO() as wav_buffer:
                    # KittenTTS usually returns float32 at 24000Hz (or model specific)
                    # We write it as WAV so the browser can decode it easily
                    sf.write(wav_buffer, audio, 24000, format='WAV')
                    wav_buffer.seek(0)
                    data = wav_buffer.read()
                    print(f"KittenTTS generation took: {time.time() - kitten_start:.4f}s")
                    return data
                    
            except Exception as e:
                print(f"KittenTTS error: {e}")
                # Fallthrough to next provider
        
        # 2. Try Edge TTS (Online, High Quality)
        if self.use_edge_tts:
            try:
                edge_start = time.time()
                communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
                audio_data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_data += chunk["data"]
                print(f"EdgeTTS generation took: {time.time() - edge_start:.4f}s")
                return audio_data
            except Exception as e:
                print(f"EdgeTTS failed: {e}")

        # 3. Fallback to pyttsx3 (Offline, Low Quality)
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
        # Save to temp file because getting bytes directly is hard
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

# Initialize Services
db_service.connect()
search_service = SearchService()
orchestrator_service = OrchestratorService()
llm_service = LLMService()
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

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")
    
    # Session state
    history = [] 
    current_task: asyncio.Task = None

    async def process_interaction(user_text: str):
        try:
            query_received_time = time.time()
            print(f"\n==================================================")
            print(f"[{query_received_time:.3f}] User Query Received: {user_text}")
            
            # 1. Orchestrate
            route_decision = orchestrator_service.route_query(user_text)
            route = route_decision["route"]
            
            context = ""
            if route == "SERPER":
                await websocket.send_json({"type": "status", "message": "Searching web..."})
                search_q = route_decision.get("serper_query") or user_text
                context = search_service.search(search_q)
                prompt = f"Context from Web Search:\n{context}\n\nUser Query: {user_text}"
            else:
                prompt = user_text
            
            # 2. Generate Response
            await websocket.send_json({"type": "status", "message": "Thinking..."})
            
            full_response = ""
            sentence_buffer = ""
            
            # Use history in LLM generation
            async for token in llm_service.generate_response(prompt, history):
                full_response += token
                sentence_buffer += token
                
                # Stream token to UI
                await websocket.send_json({"type": "llm_token", "text": token})
                
                # TTS Streaming
                if token in [".", "!", "?", "\n"] and len(sentence_buffer) > 10:
                    await websocket.send_json({"type": "status", "message": "Speaking..."})
                    audio_bytes = await tts_service.speak(sentence_buffer)
                    if audio_bytes:
                        b64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                        await websocket.send_json({"type": "audio", "data": b64_audio})
                    sentence_buffer = ""
            
            # Process remaining buffer
            if sentence_buffer.strip():
                audio_bytes = await tts_service.speak(sentence_buffer)
                if audio_bytes:
                    b64_audio = base64.b64encode(audio_bytes).decode('utf-8')
                    await websocket.send_json({"type": "audio", "data": b64_audio})
            
            # Update history (keep last 10 turns)
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": full_response})
            if len(history) > 20: 
                history.pop(0)
                history.pop(0)

            # Log to DB
            await db_service.log_conversation(user_text, full_response, route, route_decision.get("serper_query"))
            
            await websocket.send_json({"type": "response_complete"})
        
        except asyncio.CancelledError:
            print(f"Task cancelled for query: {user_text}")
            raise
        except Exception as e:
            print(f"Error in process_interaction: {e}")
            await websocket.send_json({"type": "error", "message": str(e)})

    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                continue
                
            msg_type = message.get("type")

            if msg_type == "query":
                # HARD AUDIO OWNERSHIP RULE (Backend Verification)
                if message.get("source") != "user":
                    print(f"SECURITY: Rejected input from non-user source: {message.get('source')}")
                    continue

                user_text = message.get("text", "").strip()
                if not user_text:
                    continue
                
                # Cancel previous task if running (Barge-in support)
                if current_task and not current_task.done():
                    print(f"Barge-in: Cancelling previous task for new query: {user_text}")
                    current_task.cancel()
                    try:
                        await current_task
                    except asyncio.CancelledError:
                        pass
                
                current_task = asyncio.create_task(process_interaction(user_text))

            elif msg_type == "stop":
                # Explicit stop (e.g., from frontend barge-in)
                if current_task and not current_task.done():
                    print("Stop received. Cancelling current task.")
                    current_task.cancel()
                    try:
                        await current_task
                    except asyncio.CancelledError:
                        pass

            elif msg_type == "reset_context":
                history = []
                print("Context reset.")
                
    except WebSocketDisconnect:
        print("Client disconnected")
        if current_task and not current_task.done():
            current_task.cancel()
    except Exception as e:
        print(f"WebSocket Error: {e}")

if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 to allow external access if needed, but localhost is fine
    uvicorn.run(app, host="0.0.0.0", port=8000)
