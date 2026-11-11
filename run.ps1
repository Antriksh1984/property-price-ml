<#
.SYNOPSIS
    One-command local startup for the Bengaluru property price app (Windows).

.DESCRIPTION
    1. Creates .venv if missing and installs requirements (re-installs only when requirements change).
    2. Trains the model if artifacts/ is missing (or when -Retrain is passed).
    3. Starts the FastAPI backend (port 8000) and the Vite frontend (port 5173) in separate windows.

.EXAMPLE
    .\run.ps1              # set up if needed, then start both servers
    .\run.ps1 -Retrain     # force re-training first
    .\run.ps1 -SetupOnly   # venv + install + train, but don't start servers

    If scripts are blocked:  powershell -ExecutionPolicy Bypass -File .\run.ps1
#>
[CmdletBinding()]
param(
    [switch]$Retrain,
    [switch]$SetupOnly
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
Set-Location $Root

function Write-Step([string]$Message) { Write-Host "`n==> $Message" -ForegroundColor Cyan }

# --- Prerequisites ---------------------------------------------------------
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    throw 'Python 3.10+ was not found on PATH. Install it from https://www.python.org/downloads/'
}
if (-not $SetupOnly -and -not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw 'Node.js 18+ (npm) was not found on PATH. Install it from https://nodejs.org/'
}

$DataFile = Join-Path $Root 'data\raw\bengaluru_house_data.csv'
$VenvPython = Join-Path $Root '.venv\Scripts\python.exe'

# --- Virtual environment ---------------------------------------------------
if (-not (Test-Path $VenvPython)) {
    Write-Step 'Creating virtual environment (.venv)'
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Failed to create the virtual environment.' }
}

# Re-install only when requirements files changed since the last successful install.
$Marker = Join-Path $Root '.venv\.requirements.hash'
$ReqHash = (Get-FileHash requirements.txt, requirements-api.txt -Algorithm SHA256 |
    ForEach-Object Hash) -join ''
if (-not (Test-Path $Marker) -or (Get-Content $Marker -Raw).Trim() -ne $ReqHash) {
    Write-Step 'Installing Python requirements'
    & $VenvPython -m pip install --upgrade pip --quiet
    & $VenvPython -m pip install -r requirements.txt --quiet
    if ($LASTEXITCODE -ne 0) { throw 'pip install failed.' }
    Set-Content -Path $Marker -Value $ReqHash -Encoding ascii
} else {
    Write-Step 'Python requirements already installed'
}

# --- Model -----------------------------------------------------------------
$Artifacts = 'artifacts\model.joblib', 'artifacts\locations.json', 'artifacts\metrics.json'
$Missing = $Artifacts | Where-Object { -not (Test-Path (Join-Path $Root $_)) }
if ($Retrain -or $Missing) {
    if (-not (Test-Path $DataFile)) {
        throw "Dataset not found. Download 'Bengaluru House Price Data' from Kaggle and save it as:`n  $DataFile"
    }
    Write-Step 'Training the model (takes ~2 minutes)'
    $env:PYTHONIOENCODING = 'utf-8'
    & $VenvPython -m ml.train
    if ($LASTEXITCODE -ne 0) { throw 'Training failed.' }
} else {
    Write-Step 'Model artifacts found (use -Retrain to rebuild)'
}

# --- Frontend dependencies -------------------------------------------------
if (-not (Test-Path (Join-Path $Root 'frontend\node_modules'))) {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Write-Step 'Installing frontend dependencies (npm install)'
        Push-Location frontend
        try {
            npm install
            if ($LASTEXITCODE -ne 0) { throw 'npm install failed.' }
        } finally { Pop-Location }
    }
}

if ($SetupOnly) {
    Write-Step 'Setup complete (servers not started because of -SetupOnly)'
    exit 0
}

# --- Start servers in separate windows ------------------------------------
Write-Step 'Starting backend on http://localhost:8000 (new window)'
$BackendCmd = "Set-Location '$Root'; & '$VenvPython' -m uvicorn backend.main:app --reload --port 8000"
Start-Process powershell -ArgumentList '-NoExit', '-Command', $BackendCmd | Out-Null

Write-Step 'Starting frontend on http://localhost:5173 (new window)'
$FrontendCmd = "Set-Location '$Root\frontend'; npm run dev"
Start-Process powershell -ArgumentList '-NoExit', '-Command', $FrontendCmd | Out-Null

# Wait for the API, then open the app in the default browser.
Write-Step 'Waiting for the API to come up'
$ready = $false
for ($i = 0; $i -lt 40; $i++) {
    try {
        Invoke-RestMethod -Uri 'http://localhost:8000/health' -TimeoutSec 2 | Out-Null
        $ready = $true
        break
    } catch { Start-Sleep -Milliseconds 750 }
}
if ($ready) {
    Write-Host 'API is up. Opening http://localhost:5173 ...' -ForegroundColor Green
    Start-Sleep -Seconds 2
    Start-Process 'http://localhost:5173'
} else {
    Write-Warning 'The API did not respond within 30s - check the backend window for errors.'
}
Write-Host "`nClose the two server windows to stop the app." -ForegroundColor Yellow
