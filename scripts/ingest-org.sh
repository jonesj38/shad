#!/bin/bash
#
# Ingest all repos from a GitHub organization
#
# Usage: ./ingest-org.sh <org-name> <vault-path> [preset]
# Example: ./ingest-org.sh bsv-blockchain ~/MyVault docs
#

set -euo pipefail

ORG="${1:-bsv-blockchain}"
VAULT="${2:-$HOME/Desktop/test_vault}"
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

  if shad ingest github "$repo" --preset "$PRESET" --vault "$VAULT" 2>/dev/null; then
    echo "  ✓ Done"
  else
    echo "  ✗ Failed (continuing...)"
  fi
  echo ""
done

echo "Ingestion complete!"
