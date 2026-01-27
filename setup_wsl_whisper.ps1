# Check if running on Windows (compatible with PS 5.1 and Core)
if ($env:OS -notlike "*Windows*") {
    # If $env:OS is not available or doesn't match, fall back to $IsWindows (Core) or assume Windows if neither checks fail explicitly
    if ($IsWindows -eq $false) {
         Write-Error "This script is intended for Windows."
         exit 1
    }
}

Write-Host "Checking for WSL..." -ForegroundColor Cyan
$wslStatus = wsl --status 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Error "WSL is NOT detected or not running."
    Write-Host "Please install WSL by running the following command in PowerShell as Administrator:" -ForegroundColor Yellow
    Write-Host "    wsl --install" -ForegroundColor White
    Write-Host "After installation, restart your computer and run this script again." -ForegroundColor Yellow
    exit 1
}
Write-Host "WSL is installed and active." -ForegroundColor Green

$backendDir = Join-Path $PSScriptRoot "backend"
$whisperDir = Join-Path $backendDir "whisper.cpp"

# Clone whisper.cpp
if (-not (Test-Path $whisperDir)) {
    Write-Host "Cloning whisper.cpp..." -ForegroundColor Cyan
    # We use git from Windows, assuming it's available.
    git clone https://github.com/ggerganov/whisper.cpp.git $whisperDir
} else {
    Write-Host "whisper.cpp directory already exists." -ForegroundColor Yellow
}

# Build using make via WSL
Write-Host "Building whisper.cpp using WSL..." -ForegroundColor Cyan
Write-Host "Note: This requires 'make', 'g++', and 'build-essential' in your WSL distro." -ForegroundColor Gray

# We need to run the command inside the directory.
# PowerShell's Set-Location affects the CWD for the wsl command.
Push-Location $whisperDir

try {
    # Check if make is available
    wsl which make | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "It seems 'make' is not installed in WSL."
        Write-Host "Attempting to install dependencies (you may be prompted for password)..." -ForegroundColor Yellow
        wsl sudo apt-get update `&`& sudo apt-get install -y build-essential
    }

    # Run make
    # We use wsl to run make in the current directory (which is mounted)
    wsl make
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed."
    }
    Write-Host "Build successful." -ForegroundColor Green

    # Download model
    Write-Host "Downloading base.en model..." -ForegroundColor Cyan
    # The script is in models/download-ggml-model.sh
    # Fix line endings using sed (more common than dos2unix)
    wsl sed -i 's/\r$//' models/download-ggml-model.sh
    wsl bash ./models/download-ggml-model.sh base.en

    # Verify build
    Write-Host "Verifying build with sample transcription..." -ForegroundColor Cyan
    if (Test-Path "samples/jfk.wav") {
        wsl ./main -m models/ggml-base.en.bin -f samples/jfk.wav --no-timestamps
    } else {
        Write-Warning "Sample file not found, skipping verification."
    }

} catch {
    Write-Error "An error occurred during the WSL setup: $_"
} finally {
    Pop-Location
}

Write-Host "WSL Setup Complete." -ForegroundColor Green
Write-Host "You can now use the 'wsl' mode in the backend." -ForegroundColor Gray
