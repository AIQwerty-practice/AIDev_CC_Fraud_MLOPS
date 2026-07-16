# =============================================================
# Demarrage complet de la stack AutoML Credit Card Fraud (Docker Desktop)
# Usage :  clic droit > "Executer avec PowerShell"
#   ou    :  .\start.ps1
#   options:  .\start.ps1 -Rebuild     (force le rebuild des images)
#             .\start.ps1 -Down         (arrete tout et nettoie)
# =============================================================
param(
    [switch]$Rebuild,   # Force "docker compose build" avant le up
    [switch]$Down       # Arrete et supprime les conteneurs puis quitte
)

$ErrorActionPreference = "Stop"

# Se placer dans le dossier du script (pour que docker compose trouve le yml)
Set-Location -Path $PSScriptRoot

function Write-Step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }

# --- Option d'arret ---
if ($Down) {
    Write-Step "Arret de la stack"
    docker compose down
    Write-Host "Stack arretee." -ForegroundColor Green
    return
}

# --- 1) Verifier que Docker tourne ---
Write-Step "Verification de Docker Desktop"
try {
    docker info *> $null
    if ($LASTEXITCODE -ne 0) { throw }
    Write-Host "Docker est demarre." -ForegroundColor Green
} catch {
    Write-Host "Docker Desktop n'est pas demarre. Lance Docker Desktop puis relance ce script." -ForegroundColor Red
    Read-Host "Appuie sur Entree pour quitter"
    return
}

# --- 2) Build (optionnel) + Up ---
if ($Rebuild) {
    Write-Step "Build des images (force)"
    docker compose build
}

Write-Step "Demarrage de la stack (mlflow -> trainer -> backend -> frontend)"
Write-Host "Le 1er demarrage entraine le modele : cela prend quelques minutes..." -ForegroundColor Yellow
docker compose up -d --build
if ($LASTEXITCODE -ne 0) {
    Write-Host "Echec du demarrage. Consulte les logs : docker compose logs" -ForegroundColor Red
    Read-Host "Appuie sur Entree pour quitter"
    return
}

# --- 3) Attendre que le backend soit healthy ---
Write-Step "Attente que le backend soit pret"
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        $cid = (docker compose ps -q backend) 2>$null
        if ($cid) {
            $status = (docker inspect --format '{{.State.Health.Status}}' $cid) 2>$null
        } else { $status = "" }
    } catch { $status = "" }
    if ($status -eq "healthy") { $ready = $true; break }
    Start-Sleep -Seconds 5
    Write-Host "  ...en cours ($status)" -ForegroundColor DarkGray
}

if (-not $ready) {
    Write-Host "Le backend n'est pas encore pret. Verifie : docker compose logs -f" -ForegroundColor Yellow
} else {
    Write-Host "Backend pret !" -ForegroundColor Green
}

# --- 4) Recap + ouverture des interfaces ---
Write-Step "Stack demarree"
docker compose ps
Write-Host ""
Write-Host "  UI Streamlit : http://localhost:8501" -ForegroundColor Green
Write-Host "  API FastAPI  : http://localhost:8000/docs" -ForegroundColor Green
Write-Host "  MLflow UI    : http://localhost:5000" -ForegroundColor Green
Write-Host "  H2O Flow     : http://localhost:54321" -ForegroundColor Green
Write-Host ""
Write-Host "Astuce : dans l'UI, uploade backend\data\sample_test.csv ou sample_test_labeled.csv puis clique 'Start Prediction'." -ForegroundColor DarkGray
Write-Host "Pour tout arreter : .\start.ps1 -Down" -ForegroundColor DarkGray

if ($ready) {
    Start-Process "http://localhost:8501"
    Start-Process "http://localhost:5000"
}
