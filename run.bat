@echo off
echo Starting Voice Assistant...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo Virtual environment not found. Running with global python...
)

echo Opening Frontend...
start "" "frontend.html"

echo Starting Ollama...
start "Ollama Server" cmd /c "ollama serve"

echo Starting Backend...
python backend.py
pause
