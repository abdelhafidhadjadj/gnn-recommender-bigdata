#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Lance le pipeline complet GNN Recommender.

.DESCRIPTION
    Enchaîne automatiquement :
      1. Génération des données medium (si absentes)
      2. Entraînement des 3 modèles (SAGE, GAT, LightGCN)
      3. Évaluation + rapport de comparaison
      4. Génération du batch incrémental
      5. Apprentissage incrémental sur le meilleur modèle
      6. Lancement de l'interface Streamlit

.PARAMETER Model
    Entraîner un seul modèle : sage | gat | lightgcn | all (défaut: all)

.PARAMETER SkipTrain
    Sauter l'entraînement (utilise les checkpoints existants).

.PARAMETER SkipIncremental
    Sauter l'apprentissage incrémental.

.PARAMETER SkipApp
    Ne pas lancer l'interface Streamlit à la fin.

.PARAMETER Debug
    Mode debug rapide (3 epochs, emb_dim=16, CPU). Idéal pour tester.

.EXAMPLE
    .\run_all.ps1                        # Pipeline complet
    .\run_all.ps1 -Debug                 # Test rapide
    .\run_all.ps1 -Model sage            # Un seul modèle
    .\run_all.ps1 -SkipTrain             # Juste l'app (modèles déjà entraînés)
    .\run_all.ps1 -SkipIncremental       # Sans apprentissage incrémental
#>

param(
    [ValidateSet("all","sage","gat","lightgcn")]
    [string]$Model = "all",
    [switch]$SkipTrain,
    [switch]$SkipIncremental,
    [switch]$SkipApp,
    [switch]$Debug
)

$PY       = "python3.13"
$ROOT     = $PSScriptRoot
$DataDir  = "data/medium"
$IncrDir  = "data/incremental"

Set-Location $ROOT

# ─── Couleurs ────────────────────────────────────────────────────────────────
function Write-Step  { param($n,$msg); Write-Host ""; Write-Host "[$n] $msg" -ForegroundColor Cyan }
function Write-OK    { param($msg);    Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn  { param($msg);    Write-Host "  [!]  $msg" -ForegroundColor Yellow }
function Write-Err   { param($msg);    Write-Host "  [X]  $msg" -ForegroundColor Red; exit 1 }
function Write-Info  { param($msg);    Write-Host "       $msg" -ForegroundColor Gray }

# ─── Bannière ────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor DarkCyan
Write-Host "║       GNN Recommender — Pipeline Complet             ║" -ForegroundColor DarkCyan
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor DarkCyan
Write-Host "  Modèle(s)    : $Model"      -ForegroundColor Gray
Write-Host "  Debug        : $Debug"      -ForegroundColor Gray
Write-Host "  SkipTrain    : $SkipTrain"  -ForegroundColor Gray
Write-Host "  SkipIncr.    : $SkipIncremental" -ForegroundColor Gray

# ─── Vérifier Python ─────────────────────────────────────────────────────────
$pyCheck = & $PY -c "import torch; print('ok')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Err "python3.13 ou torch introuvable. Vérifiez l'installation."
}

$Errors = 0
$Times  = @{}

function Run-Step {
    param([string]$Label, [string]$Cmd)
    Write-Info "Commande : $Cmd"
    $t0 = Get-Date
    Invoke-Expression $Cmd
    $elapsed = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Échec : $Label"
        $script:Errors++
    } else {
        Write-OK "$Label terminé en ${elapsed}s"
        $script:Times[$Label] = $elapsed
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Données medium
# ═══════════════════════════════════════════════════════════════════════════════
Write-Step "1/6" "Vérification des données medium"

$reviewFile = "$DataDir/yelp_academic_dataset_review_healthandmedical.csv"
if (Test-Path $reviewFile) {
    Write-OK "Dataset medium déjà présent — skip génération"
} else {
    Write-Info "Fichier absent — génération en cours..."
    Run-Step "Génération medium" "$PY scripts/generate_medium_data.py"
}

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Entraînement des modèles
# ═══════════════════════════════════════════════════════════════════════════════
if (-not $SkipTrain) {

    $debugFlag = if ($Debug) { "--debug" } else { "" }

    # Liste des modèles à entraîner
    $modelsToTrain = if ($Model -eq "all") { @("sage","gat","lightgcn") } else { @($Model) }

    foreach ($mdl in $modelsToTrain) {
        Write-Step "2/6" "Entraînement — $($mdl.ToUpper())"

        if ($Debug) {
            # Mode debug : pas de config YAML (auto-config)
            Run-Step "Train $mdl (debug)" "$PY main.py --model $mdl --mode scratch --data-dir $DataDir --debug"
        } else {
            $configArg = if (Test-Path "configs/medium_$mdl.yaml") { "--config configs/medium_$mdl.yaml" } else { "" }
            Run-Step "Train $mdl" "$PY main.py --model $mdl --mode scratch --data-dir $DataDir $configArg"
        }
    }

} else {
    Write-Step "2/6" "Entraînement — IGNORÉ (--SkipTrain)"
    # Vérifier qu'au moins un checkpoint existe
    $anyChkpt = Get-ChildItem "checkpoints" -Recurse -Filter "model_best.pt" -ErrorAction SilentlyContinue
    if (-not $anyChkpt) {
        Write-Warn "Aucun checkpoint trouvé dans checkpoints/. Lancez sans -SkipTrain d'abord."
    } else {
        Write-OK "$(@($anyChkpt).Count) checkpoint(s) trouvé(s)"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Évaluation + rapport
# ═══════════════════════════════════════════════════════════════════════════════
Write-Step "3/6" "Évaluation des modèles"

$modelsToEval = if ($Model -eq "all") { @("sage","gat","lightgcn") } else { @($Model) }

foreach ($mdl in $modelsToEval) {
    $ckptPath = "checkpoints/$mdl/model_best.pt"
    if (Test-Path $ckptPath) {
        Run-Step "Eval $mdl" "$PY main.py --model $mdl --mode evaluate --ckpt $ckptPath --data-dir $DataDir"
    } else {
        Write-Warn "Checkpoint $ckptPath introuvable — skip évaluation $mdl"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Génération du batch incrémental
# ═══════════════════════════════════════════════════════════════════════════════
if (-not $SkipIncremental) {
    Write-Step "4/6" "Génération du batch incrémental"

    $incrFile = "$IncrDir/incremental_reviews.csv"
    if (Test-Path $incrFile) {
        Write-OK "Batch incrémental déjà présent — skip génération"
    } else {
        Run-Step "Génération incrémental" "$PY generate_incremental_dataset.py"
    }

    # ═══════════════════════════════════════════════════════════════════════════
    # ÉTAPE 5 — Apprentissage incrémental
    # ═══════════════════════════════════════════════════════════════════════════
    Write-Step "5/6" "Apprentissage incrémental"

    # Choisir le modèle pour l'incrémental (sage par défaut, ou le modèle sélectionné)
    $incrModel = if ($Model -eq "all") { "sage" } else { $Model }
    $ckptPath  = "checkpoints/$incrModel/model_best.pt"
    $incrData  = "$IncrDir/incremental_reviews.csv"

    if (Test-Path $ckptPath) {
        if (Test-Path $incrData) {
            $debugFlag2 = if ($Debug) { "--finetune-epochs 2" } else { "--finetune-epochs 20 --finetune-lr-scale 0.1 --replay-ratio 0.3" }
            Run-Step "Incrémental $incrModel" "$PY main.py --model $incrModel --mode incremental --ckpt $ckptPath --new-data $incrData --data-dir $DataDir $debugFlag2"
        } else {
            Write-Warn "Fichier incrémental introuvable ($incrData) — skip"
        }
    } else {
        Write-Warn "Checkpoint $ckptPath introuvable — skip incrémental"
    }
} else {
    Write-Step "4/6" "Données incrémentales — IGNORÉ (--SkipIncremental)"
    Write-Step "5/6" "Apprentissage incrémental — IGNORÉ (--SkipIncremental)"
}

# ═══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 6 — Interface Streamlit
# ═══════════════════════════════════════════════════════════════════════════════
if (-not $SkipApp) {
    Write-Step "6/6" "Lancement de l'interface Streamlit"

    $stlCheck = & $PY -c "import streamlit" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "Streamlit non installé. Installer avec : $PY -m pip install streamlit"
    } else {
        & "$ROOT\run_app.ps1"
    }
} else {
    Write-Step "6/6" "Interface Streamlit — IGNORÉ (--SkipApp)"
}

# ═══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ FINAL
# ═══════════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "══════════════════════════════════════════════════" -ForegroundColor DarkCyan
Write-Host "  RÉSUMÉ DU PIPELINE" -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════════════" -ForegroundColor DarkCyan

if ($Times.Count -gt 0) {
    $Times.GetEnumerator() | Sort-Object Value | ForEach-Object {
        Write-Host ("  {0,-35} {1,6}s" -f $_.Key, $_.Value) -ForegroundColor White
    }
    $total = ($Times.Values | Measure-Object -Sum).Sum
    Write-Host ""
    Write-Host ("  {'Total',-35} {0,6}s" -f [math]::Round($total,1)) -ForegroundColor Cyan
}

Write-Host ""
if ($Errors -eq 0) {
    Write-Host "  Résultat : SUCCÈS" -ForegroundColor Green
    if (-not $SkipApp) {
        Write-Host "  Interface : http://localhost:8501" -ForegroundColor Green
    }
    Write-Host "  Métriques : outputs/metrics/" -ForegroundColor Gray
    Write-Host "  Rapports  : outputs/reports/model_comparison.md" -ForegroundColor Gray
} else {
    Write-Host "  Résultat : $Errors erreur(s) détectée(s)" -ForegroundColor Red
    Write-Host "  Relancez les étapes en erreur manuellement." -ForegroundColor Yellow
}
Write-Host ""
