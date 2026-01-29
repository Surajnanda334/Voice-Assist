# Local Voice Assistant

A fully local, voice-controlled AI assistant using:
- **Frontend**: React + TypeScript + Vite + Tailwind CSS
- **Backend**: FastAPI + WebSockets
- **STT**: Whisper.cpp (Standalone binary, Python 3.14 compatible)
- **LLM**: Ollama (local Llama 3 / Mistral)
- **TTS**: Piper TTS (Standalone binary, Fast, CPU-optimized)

## Prerequisites

1. **Python 3.10+** (Works with 3.14 via standalone binaries).
2. **Node.js** (for frontend).
3. **Ollama**: [Download & Install](https://ollama.com/).
   - Run `ollama pull llama3` (or your preferred model).

## Setup

1. **Backend**:
   ```bash
   cd backend
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   
   # Install/Setup Standalone Binaries (Whisper.cpp & Piper)
   python setup_whisper.py
   python setup_piper.py
   ```
   *Note: This downloads `whisper.cpp` (STT) and `piper` (TTS) binaries to the `backend` folder, ensuring compatibility with newer Python versions where native libraries might fail.*

2. **Frontend**:
   ```bash
   cd frontend
   npm install
   ```

## Running

1. **Start Backend**:
   - **Manual**:
     ```bash
     cd backend
     .\venv\Scripts\Activate.ps1
     uvicorn app.main:app --reload --port 8000
     ```

2. **Start Frontend**:
   - **Manual**:
     ```bash
     cd frontend
     npm run dev
     ```

3. Open `http://localhost:5173` in your browser.


4. Future improvements:
   - Persistent Long-Term Personalized Memory via Knowledge Graph✅
   - Expressive Emotional TTS with Dynamic Prosody Control
   - Advanced Agentic Multi-Step Task Execution
   - Language Understanding and Interpretation
   - Have laptop context aware of all application all files search for files for application and open them
   - Super tonic tts vs soprano TTS 