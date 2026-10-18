<#
.SYNOPSIS
    Lanzador de 1 comando para el MVP Integrado Indra Smart Pick & Pack (Reto 4 — Entregable #10).
.DESCRIPTION
    Inicia:
      1. Servicio de Inferencia de Visión Clásica COIL-100 en http://localhost:8000
      2. Copiloto Agéntico Operativo + Consola Web Unificada en http://localhost:8001
    Si se pasa el parámetro -UseDocker, ejecuta `docker compose up --build`.
#>
param(
    [switch]$UseDocker = $false
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " INDRA SMART PICK & PACK — MVP INTEGRADO (VISION + AGENTE + CONSOLA)" -ForegroundColor Cyan
Write-Host "=====================================================================" -ForegroundColor Cyan

if ($UseDocker) {
    Write-Host "[Docker Mode] Construyendo y levantando servicios con docker compose..." -ForegroundColor Yellow
    docker compose up --build
    exit 0
}

$PythonExe = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

Write-Host "[1/2] Iniciando Servicio de Inferencia de Visión en http://localhost:8000 ..." -ForegroundColor Green
$VisionJob = Start-Process -FilePath $PythonExe -ArgumentList "-m uvicorn src.vision_service.app:app --host 0.0.0.0 --port 8000" -PassThru -NoNewWindow

Start-Sleep -Seconds 2

Write-Host "[2/2] Iniciando Copiloto Agéntico + Consola Web en http://localhost:8001 ..." -ForegroundColor Green
Write-Host "      -> Consola Operativa Interactiva: http://localhost:8001/" -ForegroundColor Cyan
Write-Host "      -> Swagger API Copiloto Agéntico: http://localhost:8001/docs" -ForegroundColor Cyan
Write-Host "      -> Swagger API Motor de Visión  : http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "Presiona Ctrl+C para detener ambos servicios." -ForegroundColor Yellow

try {
    & $PythonExe -m uvicorn src.agent_service.app:app --host 0.0.0.0 --port 8001
}
finally {
    if ($VisionJob -and -not $VisionJob.HasExited) {
        Stop-Process -Id $VisionJob.Id -Force -ErrorAction SilentlyContinue
    }
}
