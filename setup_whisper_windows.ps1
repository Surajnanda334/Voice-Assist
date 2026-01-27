$ErrorActionPreference = "Stop"

$backendDir = Join-Path $PSScriptRoot "backend"
$whisperDir = Join-Path $backendDir "whisper.cpp"
$modelsDir = Join-Path $whisperDir "models"

if (-not (Test-Path $whisperDir)) {
    New-Item -ItemType Directory -Path $whisperDir -Force | Out-Null
}
if (-not (Test-Path $modelsDir)) {
    New-Item -ItemType Directory -Path $modelsDir -Force | Out-Null
}

# Download Binary
# Using v1.6.0 as it is known to have windows binaries (newer versions might not).
$zipUrl = "https://github.com/ggerganov/whisper.cpp/releases/download/v1.6.0/whisper-bin-x64.zip"
$zipPath = Join-Path $whisperDir "whisper-bin.zip"

Write-Host "Downloading whisper.cpp Windows binary..." -ForegroundColor Cyan
try {
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath
} catch {
    Write-Error "Failed to download binary from $zipUrl. Error: $_"
    exit 1
}

# Extract
Write-Host "Extracting binary..." -ForegroundColor Cyan
Expand-Archive -Path $zipPath -DestinationPath $whisperDir -Force
Remove-Item $zipPath

# Download Model
$modelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
$modelPath = Join-Path $modelsDir "ggml-base.en.bin"

if (-not (Test-Path $modelPath)) {
    Write-Host "Downloading base.en model..." -ForegroundColor Cyan
    Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath
} else {
    Write-Host "Model already exists." -ForegroundColor Yellow
}

Write-Host "Setup Complete. Executables should be in $whisperDir" -ForegroundColor Green
Get-ChildItem $whisperDir *.exe | Select-Object Name
