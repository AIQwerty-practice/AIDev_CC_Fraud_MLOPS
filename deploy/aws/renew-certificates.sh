#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Standalone renewal requires port 80 briefly; only the proxy is stopped.
docker compose --env-file .env -f docker-compose.yml -f docker-compose.prod.yml stop proxy
trap 'docker compose --env-file .env -f docker-compose.yml -f docker-compose.prod.yml start proxy' EXIT
docker run --rm -p 80:80 \
  -v /etc/letsencrypt:/etc/letsencrypt \
  -v /var/lib/letsencrypt:/var/lib/letsencrypt \
  certbot/certbot renew --standalone
docker compose --env-file .env -f docker-compose.yml -f docker-compose.prod.yml start proxy
trap - EXIT
