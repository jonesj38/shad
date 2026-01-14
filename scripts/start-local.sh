#!/bin/bash
# Start Shad API locally (uses Claude Code CLI from host)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SHAD_API_DIR="$PROJECT_ROOT/services/shad-api"

echo "Starting Docker services (Redis, Open Notebook, SurrealDB)..."
cd "$PROJECT_ROOT"
docker compose up -d

echo "Waiting for services to be ready..."
sleep 5

echo "Starting Shad API locally..."
cd "$SHAD_API_DIR"

# Activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Start the API server
uvicorn shad.api.main:app --host 0.0.0.0 --port 8000 --reload
