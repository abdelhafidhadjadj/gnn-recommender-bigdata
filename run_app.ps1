#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Lance l'interface Streamlit du GNN Recommender.

.DESCRIPTION
    Utilise python3.13 (qui contient torch, streamlit, etc.)
    Tue l'instance existante sur le port si necessaire.

.PARAMETER Port
    Port Streamlit (defaut: 8501)

.PARAMETER Stop
    Arrete Streamlit si il tourne.

.EXAMPLE
    .\run_app.ps1          # Lance sur http://localhost:8501
    .\run_app.ps1 -Port 8502
    .\run_app.ps1 -Stop
#>

param(
    [int]$Port = 8501,
    [switch]$Stop
)

$PY = "python3.13"
$APP = "demo\app.py"

function Kill-Port {
    param([int]$p)
    $pids = Get-NetTCPConnection -LocalPort $p -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($id in $pids) {
        Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
    }
}

if ($Stop) {
    Write-Host "Arret de Streamlit (port $Port)..." -ForegroundColor Cyan
    Kill-Port $Port
    Write-Host "  [OK] Arrete" -ForegroundColor Green
    exit 0
}

# Tuer l'instance existante si presente
$existing = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "  Port $Port occupe - arret de l'ancienne instance..." -ForegroundColor Yellow
    Kill-Port $Port
    Start-Sleep -Seconds 2
}

# Verifier que python3.13 a torch
$check = & $PY -c "import torch, streamlit" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [X] torch ou streamlit manquant dans python3.13" -ForegroundColor Red
    Write-Host "  Installer avec: python3.13 -m pip install torch streamlit" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "Lancement Streamlit..." -ForegroundColor Cyan
Write-Host "  Python  : $PY" -ForegroundColor Gray
Write-Host "  App     : $APP" -ForegroundColor Gray
Write-Host "  URL     : http://localhost:$Port" -ForegroundColor Gray
Write-Host ""

Start-Process $PY -ArgumentList "-m streamlit run $APP --server.port $Port" -WindowStyle Normal

Start-Sleep -Seconds 5
if (Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue) {
    Write-Host "  [OK] Streamlit demarre -> http://localhost:$Port" -ForegroundColor Green
    # Ouvrir dans le navigateur par defaut
    Start-Process "http://localhost:$Port"
} else {
    Write-Host "  Demarrage en cours - attendre quelques secondes puis aller sur http://localhost:$Port" -ForegroundColor Yellow
}
