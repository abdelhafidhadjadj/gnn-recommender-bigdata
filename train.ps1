# train.ps1 — Lance le training Docker, génère le rapport HTML et l'ouvre
# Usage :
#   .\train.ps1
#   .\train.ps1 -Model gat -Workers 4 -Size full

param(
    [string]$Model   = "sage",
    [int]   $Workers = 4,
    [string]$Size    = "full",
    [string]$Mode    = "scratch"
)

$ROOT     = $PSScriptRoot
$DataDir  = "data/raw/$Size"
$EnvFile  = "$ROOT\docker\.env"

# ── Écrire le .env ────────────────────────────────────────────────────────────
@"
WORLD_SIZE=$Workers
MODEL_TYPE=$Model
DATA_DIR=$DataDir
SIZE=$Size
TRAINING_MODE=$Mode
"@ | Out-File -Encoding utf8 $EnvFile

Write-Host ""
Write-Host "============================================================"
Write-Host "  GNN Recommender — Training"
Write-Host "  Modele  : $Model"
Write-Host "  Dataset : $Size ($DataDir)"
Write-Host "  Workers : $Workers"
Write-Host "============================================================"
Write-Host ""

# ── Lancer Docker training ────────────────────────────────────────────────────
$t0 = Get-Date
docker compose --env-file $EnvFile -f "$ROOT\docker\docker-compose.yml" up --abort-on-container-exit
$elapsed = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)

Write-Host ""
Write-Host "Training termine en ${elapsed}s"

# ── Générer le rapport HTML ───────────────────────────────────────────────────
Write-Host "Generation du rapport HTML..."
python "$ROOT\generate_report.py"
python "$ROOT\generate_charts.py" 2>$null

# ── Ouvrir le rapport dans le navigateur ─────────────────────────────────────
$htmlPath = "$ROOT\results_final\report.html"
if (Test-Path $htmlPath) {
    Write-Host "Ouverture du rapport : $htmlPath"
    Start-Process $htmlPath
} else {
    Write-Host "Rapport non trouve : $htmlPath"
}
