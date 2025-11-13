#!/bin/bash

echo "=========================================="
echo "VoxScribe Voice Agent - Quick Start"
echo "=========================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found!"
    echo "Creating .env from template..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and add your AWS credentials:"
    echo "   - AWS_ACCESS_KEY_ID"
    echo "   - AWS_SECRET_ACCESS_KEY"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Check if AWS credentials are set
source .env
if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "⚠️  AWS credentials not set in .env file!"
    echo "Please edit .env and add:"
    echo "   - AWS_ACCESS_KEY_ID"
    echo "   - AWS_SECRET_ACCESS_KEY"
    exit 1
fi

echo "✓ AWS credentials found"
echo ""

# Build and start services
echo "Building and starting services..."
echo ""
docker-compose up --build -d

echo ""
echo "=========================================="
echo "Services Starting..."
echo "=========================================="
echo ""
echo "Waiting for services to be ready..."
sleep 10

# Check service health
echo ""
echo "Checking service health..."
echo ""

# Check STT
if curl -s http://localhost:8001/health > /dev/null; then
    echo "✓ STT Service: Running (http://localhost:8001)"
else
    echo "✗ STT Service: Not responding"
fi

# Check TTS
if curl -s http://localhost:8002/health > /dev/null; then
    echo "✓ TTS Service: Running (http://localhost:8002)"
else
    echo "✗ TTS Service: Not responding"
fi

# Check Agent
if curl -s http://localhost:8003/health > /dev/null; then
    echo "✓ Agent Service: Running (http://localhost:8003)"
else
    echo "✗ Agent Service: Not responding"
fi

# Check Frontend
if curl -s http://localhost:3000 > /dev/null; then
    echo "✓ Frontend: Running (http://localhost:3000)"
else
    echo "✗ Frontend: Not responding (may still be starting...)"
fi

echo ""
echo "=========================================="
echo "Next Steps"
echo "=========================================="
echo ""
echo "1. Open http://localhost:3000 in your browser"
echo "2. Go to 'Model Catalog' and download:"
echo "   - At least one STT model (e.g., Whisper Large V3)"
echo "   - At least one TTS model (e.g., Parler TTS Mini)"
echo "3. Go to 'Voice Agent' page"
echo "4. Click 'Start Speaking' and talk!"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f agent"
echo ""
echo "To stop services:"
echo "  docker-compose down"
echo ""
echo "For more info, see VOICE-AGENT-README.md"
echo ""
