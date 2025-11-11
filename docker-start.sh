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

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
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
docker-compose build

echo ""
echo "Starting services..."
docker-compose up -d

echo ""
echo "Waiting for services to start..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
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
    echo "  make logs          - View all logs"
    echo "  make logs-frontend - View frontend logs"
    echo "  make logs-stt      - View STT logs"
    echo "  make logs-tts      - View TTS logs"
    echo "  make ps            - Show service status"
    echo "  make down          - Stop services"
    echo "  make test          - Test all services"
    echo ""
else
    echo ""
    echo "❌ Services failed to start. Check logs:"
    echo "  docker-compose logs"
    exit 1
fi
