#!/bin/bash
#
# Ingest all repos from a GitHub organization
#
# Usage: ./ingest-org.sh <org-name> <vault-path> [preset]
# Example: ./ingest-org.sh bsv-blockchain ~/MyVault docs
#

set -euo pipefail

# Find shad command
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SHAD_CMD="${SCRIPT_DIR}/../services/shad-api/.venv/bin/shad"

if [ ! -x "$SHAD_CMD" ]; then
  # Try ~/.shad/bin/shad
  SHAD_CMD="$HOME/.shad/bin/shad"
fi

if [ ! -x "$SHAD_CMD" ]; then
  # Fall back to PATH
  SHAD_CMD="shad"
fi

# Verify shad is available
if ! command -v "$SHAD_CMD" &> /dev/null && [ ! -x "$SHAD_CMD" ]; then
  echo "Error: shad command not found"
  echo "Run the installer first: ./install.sh"
  exit 1
fi

if [ $# -lt 2 ]; then
  echo "Usage: $0 <org-name> <vault-path> [preset]"
  echo "Example: $0 bsv-blockchain ~/MyVault docs"
  exit 1
fi

ORG="$1"
VAULT="$2"
PRESET="${3:-docs}"

echo "Fetching repos from github.com/$ORG..."

# Fetch all repos from the org (handles pagination up to 100)
REPOS=$(curl -s "https://api.github.com/orgs/$ORG/repos?per_page=100" | grep -o '"clone_url": "[^"]*"' | cut -d'"' -f4)

if [ -z "$REPOS" ]; then
  echo "No repos found or API error"
  exit 1
fi

COUNT=$(echo "$REPOS" | wc -l)
echo "Found $COUNT repos"
echo ""

CURRENT=0
for repo in $REPOS; do
  CURRENT=$((CURRENT + 1))
  REPO_NAME=$(basename "$repo" .git)
  echo "[$CURRENT/$COUNT] Ingesting $REPO_NAME..."

  if "$SHAD_CMD" ingest github "$repo" --preset "$PRESET" --vault "$VAULT"; then
    echo "  ✓ Done"
  else
    echo "  ✗ Failed (continuing...)"
  fi
  echo ""
done

echo "Ingestion complete!"
