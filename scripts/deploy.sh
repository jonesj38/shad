#!/bin/bash
# Deploy script for Shad

set -e

echo "🚀 Deploying Shad..."

# Check for .env file
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Copying from .env.example..."
    cp .env.example .env
    echo "📝 Please edit .env with your API keys before running."
    exit 1
fi

# Build and start services
echo "📦 Building containers..."
docker compose build

echo "🔄 Starting services..."
docker compose up -d

echo "⏳ Waiting for services to be healthy..."
sleep 5

# Check health
if curl -s http://localhost:8000/v1/health | grep -q "healthy"; then
    echo "✅ Shad API is healthy!"
else
    echo "❌ Shad API health check failed"
    docker compose logs shad-api
    exit 1
fi

echo ""
echo "🎉 Deployment complete!"
echo ""
echo "API: http://localhost:8000"
echo "Health: http://localhost:8000/v1/health"
echo ""
echo "CLI usage:"
echo "  shad run \"Your goal here\" --max-depth 2"
