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

// Proxy /api/logs to STT service
app.use('/api/logs', createProxyMiddleware({
  target: STT_SERVICE_URL,
  changeOrigin: true,
  onError: (err, req, res) => {
    console.error('Logs Proxy Error:', err);
    res.status(503).json({ error: 'STT service unavailable' });
  }
}));

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
const wsSttProxy = createProxyMiddleware({
  target: STT_SERVICE_URL,
  ws: true,
  changeOrigin: true,
  logLevel: 'debug',
  onError: (err) => {
    console.error('STT WebSocket Proxy Error:', err);
  },
  onProxyReqWs: (proxyReq, req, socket) => {
    console.log('STT WebSocket proxying to:', STT_SERVICE_URL);
  }
});

// WebSocket proxy for TTS service
const wsTtsProxy = createProxyMiddleware({
  target: TTS_SERVICE_URL,
  ws: true,
  changeOrigin: true,
  logLevel: 'debug',
  onError: (err) => {
    console.error('TTS WebSocket Proxy Error:', err);
  },
  onProxyReqWs: (proxyReq, req, socket) => {
    console.log('TTS WebSocket proxying to:', TTS_SERVICE_URL);
  }
});

app.use('/ws/stt', wsSttProxy);
app.use('/ws/tts', wsTtsProxy);

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

// Handle WebSocket upgrade - forward to appropriate proxy
server.on('upgrade', (req, socket, head) => {
  console.log('WebSocket upgrade request received for:', req.url);
  if (req.url.startsWith('/ws/stt')) {
    wsSttProxy.upgrade(req, socket, head);
  } else if (req.url.startsWith('/ws/tts')) {
    wsTtsProxy.upgrade(req, socket, head);
  } else {
    socket.destroy();
  }
});
