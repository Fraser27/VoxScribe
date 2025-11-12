#!/bin/bash

# VoxScribe Docker Quick Start Script

set -e

echo "=========================================="
echo "VoxScribe Docker Setup"
echo "=========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check Docker Compose version
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
    echo "✓ Using docker-compose"
elif docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
    echo "✓ Using docker compose (plugin)"
else
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available (optional)
if docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null 2>&1; then
    echo "✓ NVIDIA Docker runtime detected"
    GPU_AVAILABLE=true
else
    echo "⚠ NVIDIA Docker runtime not detected. Running in CPU mode."
    echo "  For GPU support, install nvidia-docker2"
    GPU_AVAILABLE=false
fi

echo ""
echo "Creating necessary directories..."
mkdir -p models/{whisper,voxtral,nvidia,granite,parler,higgs,huggingface}
mkdir -p transcriptions logs

echo ""
echo "Building Docker images..."
echo "This may take 10-15 minutes on first run..."

# Build images one by one for better compatibility
echo ""
echo "Building frontend service..."
$COMPOSE_CMD build frontend

echo ""
echo "Building STT service..."
$COMPOSE_CMD build stt

echo ""
echo "Building TTS service..."
$COMPOSE_CMD build tts

echo ""
echo "Starting services..."
$COMPOSE_CMD up -d

echo ""
echo "Waiting for services to start..."
sleep 10

# Check if services are running
if $COMPOSE_CMD ps | grep -q "Up"; then
    echo ""
    echo "=========================================="
    echo "✓ VoxScribe is running!"
    echo "=========================================="
    echo ""
    echo "Access the application:"
    echo "  Web Interface: http://localhost:3000"
    echo "  STT API: http://localhost:8001"
    echo "  TTS API: http://localhost:8002"
    echo ""
    echo "Useful commands:"
    echo "  View logs: $COMPOSE_CMD logs -f"
    echo "  View frontend logs: $COMPOSE_CMD logs -f frontend"
    echo "  View STT logs: $COMPOSE_CMD logs -f stt"
    echo "  View TTS logs: $COMPOSE_CMD logs -f tts"
    echo "  Stop services: $COMPOSE_CMD down"
    echo "  Restart: $COMPOSE_CMD restart"
    echo ""
    echo "Or use the Makefile:"
    echo "  make logs    - View all logs"
    echo "  make ps      - Show service status"
    echo "  make down    - Stop services"
    echo ""
else
    echo ""
    echo "❌ Services failed to start. Check logs:"
    echo "  $COMPOSE_CMD logs"
    exit 1
fi
