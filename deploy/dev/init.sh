#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p config

if [ -f deploy/.livekit-channel.env ] && [ ! -f config/.env ]; then
  cp deploy/.livekit-channel.env config/.env
  echo "[INFO] migrated deploy/.livekit-channel.env -> config/.env"
elif [ ! -f config/.env ]; then
  cp deploy/livekit-channel.env.template config/.env
  echo "[INFO] created config/.env from template"
fi
if [ ! -f config/settings.yaml ]; then
  cp config/settings.example.yaml config/settings.yaml
  echo "[INFO] created config/settings.yaml"
fi
echo "[INFO] done."
