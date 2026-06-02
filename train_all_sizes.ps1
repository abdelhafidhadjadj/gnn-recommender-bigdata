#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Entraîne les 3 modèles GNN sur 6 tailles de dataset progressives.

.DESCRIPTION
    Tailles : 1k → 5k → 10k → 50k → 100k → full (188k)
    Modèles : GraphSAGE, GAT, LightGCN
    Checkpoints sauvegardés par taille : checkpoints/<model>_<size>/model_best.pt
    Métriques sauvegardées             : outputs/metrics/<model>_<size>_metrics.json

.PARAMETER Models
    Modèles à entraîner : "all" | "sage" | "gat" | "lightgcn"  (défaut: all)

.PARAMETER Sizes
    Tailles à entraîner (virgule-séparées) : "1k,5k,10k,50k,100k,full"  (défaut: all)

.PARAMETER DataDir
    Répertoire des données brutes  (défaut: data/raw)

.PARAMETER Debug
    Mode debug rapide (3 epochs, emb_dim=16).

.PARAMETER SkipExisting
    Sauter si le checkpoint existe déjà pour ce (modèle, taille).

.EXAMPLE
    .\train_all_sizes.ps1                              # Tout entraîner
    .\train_all_sizes.ps1 -Models sage                 # SAGE uniquement
    .\train_all_sizes.ps1 -Sizes "1k,5k,10k"          # 3 premières tailles
    .\train_all_sizes.ps1 -SkipExisting                # Reprendre où on s'est arrêté
    .\train_all_sizes.ps1 -Sizes "50k,100k,full" -Models gat   # GAT grandes tailles
#>

param(
    [ValidateSet("all","sage","gat","lightgcn")]
    [string]$Models = "all",

    [string]$Sizes = "1k,5k,10k,50k,100k,full",

    [string]$DataBase = "data/raw",   # dossier parent contenant les sous-dossiers 1k/ 5k/ ...

    [switch]$Debug,

    [switch]$SkipExisting,

    [switch]$Partition    # génère les partitions avant d'entraîner
)

$PY   = "python3.13"
$ROOT = $PSScriptRoot
Set-Location $ROOT

# ─── Helpers ──────────────────────────────────────────────────────────────────
function Write-Banner { param($msg)
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor DarkCyan
    Write-Host "║  $($msg.PadRight(51))║" -ForegroundColor DarkCyan
    Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor DarkCyan
}
function Write-Step  { param($msg); Write-Host ""; Write-Host "▶ $msg" -ForegroundColor Cyan }
function Write-OK    { param($msg); Write-Host "  ✔  $msg" -ForegroundColor Green }
function Write-Skip  { param($msg); Write-Host "  ⏭  $msg" -ForegroundColor DarkGray }
function Write-Warn  { param($msg); Write-Host "  ⚠  $msg" -ForegroundColor Yellow }
function Write-Err   { param($msg); Write-Host "  ✘  $msg" -ForegroundColor Red }
function Write-Info  { param($msg); Write-Host "     $msg" -ForegroundColor Gray }

# ─── Tailles valides ─────────────────────────────────────────────────────────
$ValidSizes = @("1k","5k","10k","50k","100k","full")

# ─── Résolution des paramètres ────────────────────────────────────────────────
$sizeList  = $Sizes -split "," | ForEach-Object { $_.Trim() }
$modelList = if ($Models -eq "all") { @("sage","gat","lightgcn") } else { @($Models) }

Write-Banner "GNN Recommender — Multi-Scale Training"
Write-Host "  Modèles  : $($modelList -join ', ')"  -ForegroundColor Gray
Write-Host "  Tailles  : $($sizeList  -join ', ')"  -ForegroundColor Gray
Write-Host "  DataBase : $DataBase"                 -ForegroundColor Gray
Write-Host "  Debug    : $Debug"                    -ForegroundColor Gray
Write-Host "  Skip     : $SkipExisting"             -ForegroundColor Gray
Write-Host "  Partition: $Partition"                -ForegroundColor Gray

# ─── Vérifier Python ─────────────────────────────────────────────────────────
$pyCheck = & $PY -c "import torch; print('ok')" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Err "python3.13 ou torch introuvable."
    exit 1
}

# ─── Étape optionnelle : générer les partitions ───────────────────────────────
if ($Partition) {
    Write-Step "Génération des partitions ($Sizes)..."
    $sizesArg = $Sizes -replace " ",""
    & $PY scripts/partition_dataset.py --input-dir $DataBase --output-base $DataBase --sizes $sizesArg
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Partition échouée."
        exit 1
    }
    Write-OK "Partitions générées dans $DataBase/{$sizesArg}"
}

# ─── Vérifier que les sous-dossiers existent ─────────────────────────────────
foreach ($size in ($Sizes -split "," | ForEach-Object { $_.Trim() })) {
    $reviewFile = "$DataBase/$size/yelp_academic_dataset_review_healthandmedical.csv"
    if (-not (Test-Path $reviewFile)) {
        Write-Warn "Partition manquante : $reviewFile"
        Write-Info "Lancez d'abord : python3.13 scripts/partition_dataset.py"
        Write-Info "Ou relancez avec le flag -Partition"
    }
}

# ─── Compteurs ────────────────────────────────────────────────────────────────
$Times   = @{}
$Errors  = @()
$Skipped = 0
$total   = $sizeList.Count * $modelList.Count
$current = 0

# ══════════════════════════════════════════════════════════════════════════════
# BOUCLE PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════
foreach ($size in $sizeList) {

    if ($size -notin $ValidSizes) {
        Write-Warn "Taille inconnue '$size' — valeurs valides : $($ValidSizes -join ', ')"
        continue
    }

    foreach ($mdl in $modelList) {
        $current++
        $ckptDir  = "checkpoints/${mdl}_${size}"
        $ckptFile = "$ckptDir/model_best.pt"
        $label    = "$($mdl.ToUpper()) @ $size"

        Write-Step "[$current/$total] $label"

        # ── Skip si checkpoint existe déjà ────────────────────────────────────
        if ($SkipExisting -and (Test-Path $ckptFile)) {
            Write-Skip "$label — checkpoint déjà présent ($ckptFile)"
            $Skipped++
            continue
        }

        # ── Vérifier que la partition existe ─────────────────────────────────
        $partitionDir = "$DataBase/$size"
        $reviewPath   = "$partitionDir/yelp_academic_dataset_review_healthandmedical.csv"
        if (-not (Test-Path $reviewPath)) {
            Write-Warn "$label — partition introuvable ($partitionDir). Skipping."
            $Errors += "$label (partition manquante)"
            continue
        }

        # ── Construire la commande ────────────────────────────────────────────
        $cmd = "$PY main.py --model $mdl --mode scratch --data-dir $partitionDir --ckpt-dir $ckptDir --output-dir outputs/${mdl}_${size}"

        if ($Debug) {
            $cmd += " --debug"
        } else {
            # Adapter les epochs selon la taille
            switch ($size) {
                "1k"   { $cmd += " --epochs 50" }
                "5k"   { $cmd += " --epochs 80" }
                "10k"  { $cmd += " --epochs 100" }
                "50k"  { $cmd += " --epochs 150" }
                "100k" { $cmd += " --epochs 150" }
                "full" { }   # valeur par défaut du config
            }
        }

        Write-Info "Commande : $cmd"

        # ── Exécuter ──────────────────────────────────────────────────────────
        $t0 = Get-Date
        Invoke-Expression $cmd
        $elapsed = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)

        if ($LASTEXITCODE -ne 0) {
            Write-Err "$label échoué après ${elapsed}s"
            $Errors += "$label"
        } else {
            Write-OK  "$label terminé en ${elapsed}s"
            $Times["$label"] = $elapsed
        }
    }
}

# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ
# ══════════════════════════════════════════════════════════════════════════════
Write-Host ""
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor DarkCyan
Write-Host "  RÉSUMÉ MULTI-SCALE" -ForegroundColor Cyan
Write-Host "══════════════════════════════════════════════════════" -ForegroundColor DarkCyan

if ($Times.Count -gt 0) {
    Write-Host ""
    Write-Host ("  {0,-28} {1,8}" -f "Modèle @ Taille", "Durée") -ForegroundColor White
    Write-Host "  $("-" * 38)" -ForegroundColor DarkGray
    # Afficher dans l'ordre : sage/gat/lightgcn × tailles
    foreach ($mdl in $modelList) {
        foreach ($size in $sizeList) {
            $key = "$($mdl.ToUpper()) @ $size"
            if ($Times.ContainsKey($key)) {
                Write-Host ("  {0,-28} {1,6}s" -f $key, $Times[$key]) -ForegroundColor White
            }
        }
    }
    $total_time = ($Times.Values | Measure-Object -Sum).Sum
    Write-Host ""
    Write-Host ("  {0,-28} {1,6}s" -f "TOTAL", [math]::Round($total_time,1)) -ForegroundColor Cyan
}

Write-Host ""
if ($Skipped -gt 0) {
    Write-Host "  Ignorés (déjà présents) : $Skipped" -ForegroundColor DarkGray
}

if ($Errors.Count -eq 0) {
    Write-Host "  Résultat : SUCCÈS ✔" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Checkpoints :" -ForegroundColor Gray
    foreach ($mdl in $modelList) {
        foreach ($size in $sizeList) {
            $ckpt = "checkpoints/${mdl}_${size}/model_best.pt"
            if (Test-Path $ckpt) {
                Write-Host "    $ckpt" -ForegroundColor Gray
            }
        }
    }
    Write-Host ""
    Write-Host "  Pour évaluer un modèle :" -ForegroundColor Gray
    Write-Host "    $PY main.py --model sage --mode evaluate --ckpt checkpoints/sage_full/model_best.pt --data-dir $DataBase/full" -ForegroundColor DarkGray
} else {
    Write-Host "  Résultat : $($Errors.Count) erreur(s)" -ForegroundColor Red
    foreach ($e in $Errors) {
        Write-Host "    ✘ $e" -ForegroundColor Red
    }
}
Write-Host ""
