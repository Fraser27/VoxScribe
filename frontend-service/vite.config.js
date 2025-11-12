import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api/stt': {
        target: process.env.STT_SERVICE_URL || 'http://localhost:8001',
        changeOrigin: true
      },
      '/api/tts': {
        target: process.env.TTS_SERVICE_URL || 'http://localhost:8002',
        changeOrigin: true
      },
      '/ws': {
        target: process.env.STT_SERVICE_URL || 'http://localhost:8001',
        ws: true,
        changeOrigin: true
      }
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
});
