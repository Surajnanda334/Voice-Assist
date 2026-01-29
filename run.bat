@echo off
echo Starting Voice Assistant...

:: Read LLM_PROVIDER from .env
set "LLM_PROVIDER=ollama"
if exist .env (
    for /f "usebackq tokens=1* delims==" %%a in (`type .env ^| findstr /b "LLM_PROVIDER="`) do (
        set "LLM_PROVIDER=%%b"
    )
)

:: Trim whitespace just in case
set "LLM_PROVIDER=%LLM_PROVIDER: =%"

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Running with global python...
)

echo Opening Frontend...
start "" "frontend.html"

if /i "%LLM_PROVIDER%"=="groq" (
    echo LLM Provider is Groq. Skipping Ollama startup.
) else (
    echo Starting Ollama...
    start "Ollama Server" cmd /c "ollama serve"
)

echo Starting Backend...
python backend.py
pause
