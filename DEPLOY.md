# Deployment Guide

This project is containerized using Docker, making it easy to deploy on any system (Linux, Windows, macOS) or cloud provider (AWS, GCP, Azure, DigitalOcean, etc.).

## Prerequisites

- [Docker](https://www.docker.com/get-started)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Quick Start

1.  **Configure Environment**:
    Create a `.env` file in the root directory (or rename `.env.example` if available) and set your API keys:
    ```ini
    GROQ_API_KEY=your_key_here
    GROQ_MODEL=llama3-70b-8192
    SERPER_API_KEY=your_key_here
    LLM_PROVIDER=groq  # or 'ollama'
    STT_PROVIDER=vosk  # or 'whisper'
    ```

2.  **Build and Run**:
    ```bash
    docker-compose up --build
    ```

3.  **Access the App**:
    Open your browser and navigate to: `http://localhost:8000`

## Configuration Details

### 1. Database (MongoDB)
The `docker-compose.yml` includes a MongoDB container. The backend automatically connects to it via `mongodb://mongo:27017`. Data is persisted in a docker volume `mongo_data`.

### 2. LLM Provider (Groq vs. Ollama)
- **Groq (Recommended for Cloud)**: Set `LLM_PROVIDER=groq` and provide `GROQ_API_KEY`. This is lightweight and requires no local GPU.
- **Ollama (Local)**:
    - Uncomment the `ollama` service in `docker-compose.yml`.
    - Set `LLM_PROVIDER=ollama`.
    - Ensure your deployment server has enough RAM/CPU (or GPU).

### 3. Speech-to-Text (STT)
The project supports multiple STT backends.
- **Vosk (Offline)**: Requires a model. Download a Vosk model (e.g., `vosk-model-small-en-us-0.15`) and extract it to a folder named `model` in the project root. Uncomment the volume mapping in `docker-compose.yml`:
  ```yaml
  - ./model:/app/model
  ```
- **Whisper (Offline)**: Requires `ffmpeg`. The Dockerfile includes `ffmpeg`. However, `whisper.cpp` (if used) requires a compiled binary compatible with Linux.

### 4. Text-to-Speech (TTS)
The project uses `edge-tts` (online, high quality, free) by default. No extra configuration is needed.

## Production Deployment Tips

- **Reverse Proxy**: For a public server, run Nginx or Traefik in front of the backend to handle SSL/HTTPS.
- **HTTPS**: Browsers require HTTPS for microphone access on non-localhost domains. **This is critical.** You cannot use the microphone on `http://your-server-ip:8000`. You MUST set up HTTPS (e.g., using Let's Encrypt).
