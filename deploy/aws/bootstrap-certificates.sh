#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f .env ]]; then
  echo "Missing .env. Copy .env.example to .env and set CERTBOT_EMAIL first." >&2
  exit 1
fi

set -a
source .env
set +a

if [[ -z "${CERTBOT_EMAIL:-}" || "$CERTBOT_EMAIL" == replace-* ]]; then
  echo "Set a real CERTBOT_EMAIL in .env." >&2
  exit 1
fi

docker run --rm -p 80:80 \
  -v /etc/letsencrypt:/etc/letsencrypt \
  -v /var/lib/letsencrypt:/var/lib/letsencrypt \
  certbot/certbot certonly --standalone --non-interactive --agree-tos \
  --email "$CERTBOT_EMAIL" \
  -d fraud.barbaraplascencia.com \
  -d app.barbaraplascencia.com \
  -d api.barbaraplascencia.com \
  -d mlflow.barbaraplascencia.com \
  -d h2o.barbaraplascencia.com
