@echo off
setlocal enabledelayedexpansion

echo 🚀 Starting development environment...

REM Check if uv is installed
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo ❌ uv is not installed. Please install it first:
    echo powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    exit /b 1
)

REM Start services with Docker
echo 🐳 Starting Docker services (MongoDB, Redis)...
docker-compose up -d mongodb redis

REM Wait for services to be ready
echo ⏳ Waiting for services to be ready...
timeout /t 5 /nobreak >nul

REM Check if virtual environment exists
if not exist ".venv" (
    echo 📦 Creating virtual environment...
    uv venv
)

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Install dependencies
echo 📥 Installing dependencies...
uv pip install -e .

REM Check if .env file exists
if not exist ".env" (
    echo ⚠️  .env file not found. Copying from .env.example...
    if exist ".env.example" (
        copy .env.example .env
        echo 📝 Please edit .env file with your API keys
    ) else (
        echo ❌ .env.example not found. Please create .env file manually
    )
)

REM Run the application
echo 🏃 Starting FastAPI application with hot reload...
echo 🌐 Application will be available at: http://localhost:5000
echo 📚 API docs will be available at: http://localhost:5000/docs
echo.
echo Press Ctrl+C to stop the application

uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 5000
