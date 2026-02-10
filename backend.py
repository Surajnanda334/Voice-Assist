import asyncio
import base64
import re
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

# Import modules
from config import settings, language_manager
from database import db_service
from search import search_service
from llm import llm_service
from tts import tts_service
from stt import stt_service

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

@app.get("/")
async def get():
    with open("frontend.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

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
            # Send the full sentence audio as one chunk to ensure smooth playback
            # (Streaming individual bytes can cause decoding artifacts/gaps in frontend)
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
                    
                    # Stream-aware Sentence Splitting for Faster TTS
                    # 1. Check for standard sentence delimiters
                    # Aggressive split for first chunk (latency optimization)
                    if len(sentence_buffer) > 0: 
                        parts = re.split(r'(?<=[.!?|])\s+', sentence_buffer)
                        if len(parts) > 1:
                            for s in parts[:-1]:
                                s = s.strip()
                                if s: await tts_queue.put((s, detected_output_language))
                            sentence_buffer = parts[-1]
                    
                    # 2. Safety Valve: Force split if buffer gets too long (latency protection)
                    # If > 100 chars (approx 15-20 words) and no sentence end, try splitting on comma
                    if len(sentence_buffer) > 100:
                        comma_parts = sentence_buffer.split(', ')
                        if len(comma_parts) > 1:
                            to_speak = comma_parts[0] + ","
                            await tts_queue.put((to_speak, detected_output_language))
                            sentence_buffer = ", ".join(comma_parts[1:])
                        else:
                            # Last resort: split on space to avoid indefinite silence
                            space_parts = sentence_buffer.split(' ')
                            if len(space_parts) > 1:
                                # Keep the last word in buffer to avoid splitting mid-word
                                to_speak = " ".join(space_parts[:-1])
                                await tts_queue.put((to_speak, detected_output_language))
                                sentence_buffer = space_parts[-1]

                # Flush remaining
                if checking_language_tag and tag_buffer:
                    if "~hi~" in tag_buffer:
                        detected_output_language = "hi-IN"
                        tag_buffer = tag_buffer.replace("~hi~", "")
                    full_response += tag_buffer
                    sentence_buffer += tag_buffer
                
                # Send the entire response as one chunk
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
