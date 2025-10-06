#!/bin/bash
set -e

echo "🚀 Starting development environment..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv is not installed. Please install it first:"
    echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Start services with Docker
echo "🐳 Starting Docker services (MongoDB, Redis)..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    uv venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
uv pip install -e .

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Copying from .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "📝 Please edit .env file with your API keys"
    else
        echo "❌ .env.example not found. Please create .env file manually"
    fi
fi

# Run the application
echo "🏃 Starting FastAPI application with hot reload..."
echo "🌐 Application will be available at: http://localhost:5000"
echo "📚 API docs will be available at: http://localhost:5000/docs"
echo ""
echo "Press Ctrl+C to stop the application"

uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 5000
