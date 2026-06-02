#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Lance le Spark Structured Streaming Consumer (Kafka -> HDFS + Parquet).

.DESCRIPTION
    Sur Windows, PySpark necessite winutils.exe + HADOOP_HOME.
    Ce script contourne le probleme en executant le consumer DANS le container
    spark-master via docker exec.

.PARAMETER Mode
    hdfs   -> ecrit en Parquet sur HDFS (defaut, mode production)
    local  -> ecrit en Parquet en local dans le container

.PARAMETER Console
    Mode debug : affiche les messages dans la console sans ecrire sur disque.

.PARAMETER Stop
    Arrete le consumer s'il tourne.

.PARAMETER Status
    Affiche l'etat du consumer + stats HDFS.

.EXAMPLE
    .\run_consumer.ps1                      # Mode HDFS (production)
    .\run_consumer.ps1 -Console             # Debug console uniquement
    .\run_consumer.ps1 -Stop                # Arret propre
    .\run_consumer.ps1 -Status              # Verifier l'etat
#>

param(
    [ValidateSet("hdfs","local")]
    [string]$Mode = "hdfs",
    [switch]$Console,
    [switch]$Stop,
    [switch]$Status
)

$ErrorActionPreference = "Stop"

function Write-Title { param($msg); Write-Host "" ; Write-Host $msg -ForegroundColor Cyan }
function Write-OK    { param($msg); Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn  { param($msg); Write-Host "  [!]  $msg" -ForegroundColor Yellow }
function Write-Err   { param($msg); Write-Host "  [X]  $msg" -ForegroundColor Red }

# ── Verifier que Docker est actif ─────────────────────────────────────────────
$dockerOk = docker ps 2>$null
if (-not $dockerOk) {
    Write-Err "Docker n'est pas demarre. Lance Docker Desktop d'abord."
    exit 1
}

# ── Verifier que spark-master tourne ──────────────────────────────────────────
$sparkRunning = docker ps --filter "name=spark-master" --filter "status=running" -q
if (-not $sparkRunning) {
    Write-Err "Container spark-master non demarre. Lance d'abord: docker compose up -d"
    exit 1
}

# ── STATUS ────────────────────────────────────────────────────────────────────
if ($Status) {
    Write-Title "=== Consumer Status ==="

    $pidVal = docker exec spark-master bash -c "pgrep -f spark_consumer || echo ''" 2>$null
    if ($pidVal) {
        Write-OK "Consumer en cours (PID $pidVal)"
        Write-Host ""
        Write-Host "  Dernieres lignes du log:" -ForegroundColor Gray
        docker exec spark-master bash -c "tail -10 /tmp/spark_consumer.log 2>/dev/null || echo 'Pas de log'" |
            ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
    } else {
        Write-Warn "Consumer arrete"
    }

    Write-Title "=== HDFS ==="
    docker exec namenode bash -c "hdfs dfs -du -h /data/scientific/ 2>/dev/null" |
        ForEach-Object { Write-OK "$_" }

    Write-Title "=== Kafka Offsets ==="
    docker exec kafka kafka-run-class kafka.tools.GetOffsetShell --bootstrap-server localhost:9092 --time -1 2>$null |
        Where-Object { $_ -match "articles" } |
        ForEach-Object {
            $parts = $_ -split ":"
            Write-OK "$($parts[0]) : $($parts[2]) messages"
        }

    Write-Title "=== Elasticsearch ==="
    try {
        $es = Invoke-RestMethod -Uri "http://localhost:9200/_cat/indices?v&h=index,docs.count,store.size" -Method Get
        $es -split "`n" | Where-Object { $_ -match "articles" } |
            ForEach-Object { Write-OK "$_" }
    } catch {
        Write-Warn "Elasticsearch inaccessible"
    }
    exit 0
}

# ── STOP ──────────────────────────────────────────────────────────────────────
if ($Stop) {
    Write-Title "Arret du Spark Consumer..."
    docker exec spark-master bash -c "pkill -SIGTERM -f spark_consumer 2>/dev/null || true; pkill -SIGTERM -f SparkSubmit 2>/dev/null || true"
    Start-Sleep -Seconds 3
    $pidVal = docker exec spark-master bash -c "pgrep -f spark_consumer || echo ''" 2>$null
    if ($pidVal) {
        docker exec spark-master bash -c "pkill -9 -f spark_consumer 2>/dev/null || true"
        Write-Warn "Force (SIGKILL)"
    } else {
        Write-OK "Consumer arrete proprement"
    }
    exit 0
}

# ── Verifier si deja en cours ─────────────────────────────────────────────────
$existingPid = docker exec spark-master bash -c "pgrep -f spark_consumer || echo ''" 2>$null
if ($existingPid) {
    Write-Warn "Consumer deja en cours (PID $existingPid) - redemarrage..."
    docker exec spark-master bash -c "pkill -9 -f spark_consumer 2>/dev/null || true"
    Start-Sleep -Seconds 2
}

# ── Copier les fichiers a jour dans le container ──────────────────────────────
Write-Title "Copie des fichiers dans spark-master..."
docker cp consumers/spark_consumer.py spark-master:/opt/spark_consumer.py
docker cp config/settings.py spark-master:/opt/settings.py
docker exec -u root spark-master bash -c "mkdir -p /opt/config && cp /opt/settings.py /opt/config/settings.py && touch /opt/config/__init__.py"
Write-OK "Fichiers copies"

# ── Construire la commande spark-submit ───────────────────────────────────────
$consoleFlag = if ($Console) { "--console" } else { "" }
$hdfsConf    = if ($Mode -eq "hdfs") {
    "--conf spark.hadoop.fs.defaultFS=hdfs://namenode:9000"
} else { "" }

$submitCmd = "/opt/spark/bin/spark-submit " +
    "--master spark://spark-master:7077 " +
    "--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0 " +
    "$hdfsConf " +
    "/opt/spark_consumer.py --mode $Mode $consoleFlag"

# ── Lancement ─────────────────────────────────────────────────────────────────
Write-Title "Lancement du Spark Consumer (mode=$Mode, console=$Console)..."
docker exec -u root spark-master bash -c "rm -f /tmp/spark_consumer.log && nohup $submitCmd > /tmp/spark_consumer.log 2>&1 &"

Write-Host "  Attente du demarrage..." -ForegroundColor Gray

$started = $false
for ($i = 0; $i -lt 8; $i++) {
    Start-Sleep -Seconds 5
    $log = docker exec spark-master bash -c "grep -E 'Stream started|Traceback|Exception in thread' /tmp/spark_consumer.log 2>/dev/null" 2>$null
    if ($log -match "Stream started") { $started = $true; break }
    if ($log -match "Traceback|Exception in thread") {
        Write-Err "Erreur au demarrage !"
        docker exec spark-master bash -c "tail -25 /tmp/spark_consumer.log"
        exit 1
    }
    Write-Host "  . ($([int]($i+1)*5)s)" -ForegroundColor Gray
}

Write-Host ""
if ($started) {
    Write-OK "Consumer demarre !"
    docker exec spark-master bash -c "grep 'Stream started' /tmp/spark_consumer.log" |
        ForEach-Object { Write-OK $_ }
} else {
    Write-Warn 'Demarrage long - verifie avec: .\run_consumer.ps1 -Status'
}

Write-Host ''
Write-Host '  Logs en direct :' -ForegroundColor Gray
Write-Host '    docker exec spark-master tail -f /tmp/spark_consumer.log' -ForegroundColor DarkGray
Write-Host ''
Write-Host '  Arret :' -ForegroundColor Gray
Write-Host '    .\run_consumer.ps1 -Stop' -ForegroundColor DarkGray
