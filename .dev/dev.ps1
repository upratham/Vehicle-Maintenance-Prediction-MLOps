# Windows / PowerShell equivalent of dev.sh
# Run from the repo root:  .\.dev\dev.ps1

$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path "$PSScriptRoot\..").Path
$FeDir = Join-Path $RootDir "frontend"
$VenvDir = Join-Path $RootDir "venv"
$Py = Join-Path $VenvDir "Scripts\python.exe"

Set-Location $RootDir

# Run a native command, swallow its output, and return its exit code without
# tripping $ErrorActionPreference="Stop" on stderr noise.
function Invoke-Native {
    param([string]$File, [string[]]$Args)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $File @Args *>$null
        return $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
}

if (-not (Test-Path $VenvDir)) {
    Write-Host "  creating venv..."
    python -m venv venv
}

# pip install if fastapi missing
$probe = Invoke-Native $Py @("-c", "import fastapi")
if ($probe -ne 0) {
    Write-Host "  installing python deps..."
    & $Py -m pip install -q -r requirements.txt
}

if (-not (Test-Path (Join-Path $FeDir "node_modules"))) {
    Write-Host "  installing frontend deps..."
    Push-Location $FeDir
    try {
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        npm install --silent 2>&1 | Out-Null
        $ErrorActionPreference = $prev
        if ($LASTEXITCODE -ne 0) { throw "npm install failed (exit $LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
}

# train models if any are missing (match dev.sh path: artifact/$p/*/model_trainer/trained_model/model.pkl)
$NeedTrain = $false
foreach ($p in @("vehicle_maintenance", "cars_hyundai", "engine_data")) {
    $hits = Get-ChildItem -Path (Join-Path $RootDir "artifact\$p") -Recurse -Filter "model.pkl" -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match "model_trainer\\trained_model\\model\.pkl$" }
    if (-not $hits) { $NeedTrain = $true; break }
}
if ($NeedTrain) {
    Write-Host "  training models (one-time, ~5-15 min)..."
    & $Py demo.py
}

if (-not (Test-Path (Join-Path $FeDir "dist"))) {
    Write-Host "  building frontend..."
    Push-Location $FeDir
    try {
        $prev = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        npm run build --silent 2>&1 | Out-Null
        $ErrorActionPreference = $prev
        if ($LASTEXITCODE -ne 0) { throw "npm run build failed (exit $LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
}

Write-Host ""
Write-Host "  Starting Vehicle-Maintenance..."
Write-Host ""

# Resolve npx.cmd specifically -- Start-Process needs a real .exe/.cmd, not npx.ps1
$NpxCmd = (Get-Command npx.cmd -ErrorAction SilentlyContinue).Source
if (-not $NpxCmd) {
    $NpxCmd = (Get-Command npx -All -ErrorAction SilentlyContinue |
               Where-Object { $_.Source -like "*.cmd" -or $_.Source -like "*.exe" } |
               Select-Object -First 1).Source
}
if (-not $NpxCmd) { throw "npx.cmd not found on PATH (only found npx.ps1, which Start-Process can't launch)" }

$BeProc = Start-Process -PassThru -NoNewWindow -FilePath $Py `
    -ArgumentList "-m","uvicorn","app:app","--host","0.0.0.0","--port","8000","--reload","--log-level","error"

$FeProc = Start-Process -PassThru -NoNewWindow -FilePath $NpxCmd `
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
