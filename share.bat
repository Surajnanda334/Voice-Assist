@echo off
title Voice Assistant - Share (Ngrok)
echo.
echo --- Voice Assistant Sharing Tool ---
echo.

:: 1. Check if backend is running (simple check, user should ensure run.bat is active)
echo NOTE: Ensure 'run.bat' is running in another terminal!
echo.

:: 2. Check for Ngrok in PATH
where ngrok >nul 2>nul
if %errorlevel% EQU 0 (
    echo [SUCCESS] Ngrok found in PATH.
    echo Starting tunnel on port 8000...
    ngrok http 8000
    goto :EOF
)

:: 3. Check for Ngrok in current folder
if exist "ngrok.exe" (
    echo [SUCCESS] Ngrok found in current folder.
    echo Starting tunnel on port 8000...
    ngrok.exe http 8000
    goto :EOF
)

:: 4. Failed
echo [ERROR] Ngrok command not found!
echo.
echo Possible reasons:
echo 1. You added it to PATH but haven't restarted VS Code/Terminal.
echo 2. The path was added incorrectly.
echo.
echo QUICK FIX:
echo -> Copy 'ngrok.exe' into this folder:
echo    %CD%
echo.
echo Once copied, run this script again.
echo.
pause
