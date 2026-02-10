import ollama
import httpx
import json
from typing import AsyncGenerator, List, Dict
from abc import ABC, abstractmethod
from datetime import datetime
from config import settings

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
            "You are an AI voice assistant running in continuous ambient mode. Follow these rules strictly:\n\n"
            "Plain text output only\n"
            "NEVER use markdown, asterisks (*), bold, italics, bullet points, emojis, or special formatting.\n"
            "Output must be clean, natural spoken Hinglish/Hindi-English sentences suitable for TTS.\n\n"
            "Single unified response\n"
            "Generate the entire reply as ONE continuous paragraph.\n"
            "Do NOT split thoughts into separate sentences that could be treated as separate threads.\n"
            "Ensure the response is sent as a single TTS chunk so only ONE voice speaks.\n\n"
            "Speech length & language\n"
            "Minimum 10–15 spoken words per response.\n"
            "Must correctly detect and respond in the SAME language as the user (Hindi, English, or Hinglish).\n"
            "Do NOT switch languages unless explicitly asked.\n\n"
            "TTS safety\n"
            "Avoid symbols, formatting characters, or punctuation patterns that could cause multiple TTS invocations.\n"
            "Use simple commas and full stops only.\n\n"
            "Ambient mode behavior\n"
            "Once ambient mode is enabled, remain in listening mode continuously.\n"
            "NEVER ask the user to tap or enable the mic again.\n"
            "Continue listening until the user explicitly exits ambient mode.\n"
            "If the user mutes and unmutes, resume listening silently without asking for permission or confirmation.\n\n"
            "No meta talk\n"
            "Do not mention permissions, microphones, ambient mode, system behavior, or internal state.\n"
            "Do not ask follow-up questions unless the user explicitly asks for suggestions.\n\n"
            "User experience\n"
            "Sound calm, friendly, and natural.\n"
            "Responses should feel like a single human speaking smoothly without interruptions.\n"
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
        
        # Language Instruction
        if language == "en-US":
            current_system += (
                "\n\nINTERNAL TTS INSTRUCTION (Silent): "
                "Detect the language of the user's prompt. "
                "If the user speaks English, reply in English. "
                "If the user speaks Hindi or Hinglish, reply in Hinglish/Hindi. "
                "If responding in Hindi or Hinglish, you MUST start your response with '~hi~' so the voice engine switches to Hindi. "
                "Example: '~hi~ Haan bhai, batao kya haal hai.' "
                "This tag '~hi~' will be removed before speaking. "
                "If responding in pure English, do NOT use the tag."
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
