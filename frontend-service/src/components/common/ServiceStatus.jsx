import React from 'react';
import Alert from '@cloudscape-design/components/alert';
import { useWebSocket } from '../../contexts/WebSocketContext';

const ServiceStatus = ({ models, status }) => {
  const { connected } = useWebSocket();

  // Check if backend services are responding
  const hasModels = models && models.length > 0;
  const hasStatus = status && status.device;

  if (!connected && !hasModels && !hasStatus) {
    return (
      <Alert
        type="warning"
        header="Backend services not available"
      >
        The STT/TTS backend services are not responding. Please ensure the services are running.
        <br />
        <br />
        <strong>To start services:</strong>
        <ul>
          <li>Docker: <code>docker-compose up -d</code></li>
          <li>Local: Start stt-service and tts-service manually</li>
        </ul>
      </Alert>
    );
  }

  if (!connected) {
    return (
      <Alert
        type="info"
        header="WebSocket disconnected"
      >
        Real-time updates are not available. Reconnecting...
      </Alert>
    );
  }

  return null;
};

export default ServiceStatus;
