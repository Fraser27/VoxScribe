#!/bin/bash

# Manual Docker Build Script for Older Docker Versions
# Use this if docker-compose build fails

set -e

echo "=========================================="
echo "VoxScribe Manual Docker Build"
echo "=========================================="
echo ""

# Create directories
echo "Creating necessary directories..."
mkdir -p models/{whisper,voxtral,nvidia,granite,parler,higgs,huggingface}
mkdir -p transcriptions logs

# Build frontend
echo ""
echo "Building frontend service..."
docker build -t voxscribe-frontend:latest ./frontend-service

# Build STT
echo ""
echo "Building STT service..."
docker build -t voxscribe-stt:latest ./stt-service

# Build TTS
echo ""
echo "Building TTS service..."
docker build -t voxscribe-tts:latest ./tts-service

echo ""
echo "=========================================="
echo "✓ All images built successfully!"
echo "=========================================="
echo ""
echo "Now start the services with:"
echo "  docker-compose up -d"
echo "  or"
echo "  docker compose up -d"
echo ""
