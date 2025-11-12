const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Get service URLs from environment
const STT_SERVICE_URL = process.env.STT_SERVICE_URL || 'http://localhost:8001';
const TTS_SERVICE_URL = process.env.TTS_SERVICE_URL || 'http://localhost:8002';

// Enable CORS
app.use(cors());

// Proxy API requests to STT service
app.use('/api/stt', createProxyMiddleware({
  target: STT_SERVICE_URL,
  changeOrigin: true,
  pathRewrite: {
    '^/api/stt': '/api/stt'
  },
  onError: (err, req, res) => {
    console.error('STT Proxy Error:', err);
    res.status(503).json({ error: 'STT service unavailable' });
  }
}));

// Proxy API requests to TTS service
app.use('/api/tts', createProxyMiddleware({
  target: TTS_SERVICE_URL,
  changeOrigin: true,
  pathRewrite: {
    '^/api/tts': '/api/tts'
  },
  onError: (err, req, res) => {
    console.error('TTS Proxy Error:', err);
    res.status(503).json({ error: 'TTS service unavailable' });
  }
}));

// WebSocket proxy for STT service
app.use('/ws', createProxyMiddleware({
  target: STT_SERVICE_URL,
  ws: true,
  changeOrigin: true,
  onError: (err, req, res) => {
    console.error('WebSocket Proxy Error:', err);
  }
}));

// Serve static files from dist directory
app.use(express.static(path.join(__dirname, 'dist')));

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', service: 'frontend' });
});

// Serve index.html for all other routes (SPA)
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

const server = app.listen(PORT, '0.0.0.0', () => {
  console.log('='.repeat(50));
  console.log('VoxScribe Frontend Service (React + Cloudscape)');
  console.log('='.repeat(50));
  console.log(`Server running on: http://0.0.0.0:${PORT}`);
  console.log(`STT Service: ${STT_SERVICE_URL}`);
  console.log(`TTS Service: ${TTS_SERVICE_URL}`);
  console.log('='.repeat(50));
});

// Handle WebSocket upgrade
server.on('upgrade', (request, socket, head) => {
  console.log('WebSocket upgrade request received');
});
