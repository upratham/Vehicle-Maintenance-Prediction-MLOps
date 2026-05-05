# Windows / PowerShell equivalent of dev.sh
# Run from the repo root:  .\.dev\dev.ps1

$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path "$PSScriptRoot\..").Path
$FeDir = Join-Path $RootDir "frontend"
$VenvDir = Join-Path $RootDir "venv"
$Py = Join-Path $VenvDir "Scripts\python.exe"

Set-Location $RootDir

if (-not (Test-Path $VenvDir)) {
    Write-Host "  creating venv..."
    python -m venv venv
}

# pip install if fastapi missing
& $Py -c "import fastapi" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  installing python deps..."
    & $Py -m pip install -q -r requirements.txt
}

if (-not (Test-Path (Join-Path $FeDir "node_modules"))) {
    Write-Host "  installing frontend deps..."
    Push-Location $FeDir
    npm install --silent
    Pop-Location
}

# train models if any are missing
$NeedTrain = $false
foreach ($p in @("vehicle_maintenance", "cars_hyundai", "engine_data")) {
    $hits = Get-ChildItem -Path "artifact\$p" -Recurse -Filter "model.pkl" -ErrorAction SilentlyContinue
    if (-not $hits) { $NeedTrain = $true; break }
}
if ($NeedTrain) {
    Write-Host "  training models (one-time, ~5-15 min)..."
    & $Py demo.py
}

if (-not (Test-Path (Join-Path $FeDir "dist"))) {
    Write-Host "  building frontend..."
    Push-Location $FeDir
    npm run build --silent
    Pop-Location
}

Write-Host ""
Write-Host "  Starting Vehicle-Maintenance..."
Write-Host ""

$BeProc = Start-Process -PassThru -NoNewWindow -FilePath $Py `
    -ArgumentList "-m","uvicorn","app:app","--host","0.0.0.0","--port","8000","--reload","--log-level","error"

$FeProc = Start-Process -PassThru -NoNewWindow -FilePath "npx" `
    -WorkingDirectory $FeDir -ArgumentList "vite","--port","3000"

Start-Sleep -Seconds 4
Write-Host "  Backend   -> http://localhost:8000"
Write-Host "  Frontend  -> http://localhost:3000"
Write-Host ""
Write-Host "  press Ctrl+C to stop"

try {
    Wait-Process -Id $BeProc.Id, $FeProc.Id
}
finally {
    if ($BeProc -and -not $BeProc.HasExited) { Stop-Process -Id $BeProc.Id -Force -ErrorAction SilentlyContinue }
    if ($FeProc -and -not $FeProc.HasExited) { Stop-Process -Id $FeProc.Id -Force -ErrorAction SilentlyContinue }
}
