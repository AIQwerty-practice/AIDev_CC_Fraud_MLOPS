#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

[[ -f .env ]] || { echo "Missing .env; copy .env.example and configure it." >&2; exit 1; }
[[ -f backend/data/processed/train.csv ]] || { echo "Missing processed training data." >&2; exit 1; }
[[ -f /etc/letsencrypt/live/fraud.barbaraplascencia.com/fullchain.pem ]] || {
  echo "TLS certificate missing; run deploy/aws/bootstrap-certificates.sh first." >&2
  exit 1
}

docker compose --env-file .env -f docker-compose.yml -f docker-compose.prod.yml pull
docker compose --env-file .env -f docker-compose.yml -f docker-compose.prod.yml up -d --build
docker compose --env-file .env -f docker-compose.yml -f docker-compose.prod.yml ps
