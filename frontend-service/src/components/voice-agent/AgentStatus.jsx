import React from 'react';
import {
  Box,
  ColumnLayout,
  Container,
  StatusIndicator,
  Header
} from '@cloudscape-design/components';

const AgentStatus = ({ isConnected, status, sessionId }) => {
  const getStatusIndicator = () => {
    if (!isConnected) {
      return <StatusIndicator type="error">Disconnected</StatusIndicator>;
    }
    
    switch (status) {
      case 'listening':
        return <StatusIndicator type="in-progress">Listening</StatusIndicator>;
      case 'transcribing':
        return <StatusIndicator type="in-progress">Transcribing</StatusIndicator>;
      case 'thinking':
        return <StatusIndicator type="in-progress">Thinking</StatusIndicator>;
      case 'speaking':
        return <StatusIndicator type="in-progress">Speaking</StatusIndicator>;
      case 'processing':
        return <StatusIndicator type="in-progress">Processing</StatusIndicator>;
      case 'idle':
      default:
        return <StatusIndicator type="success">Ready</StatusIndicator>;
    }
  };
  
  return (
    <Container>
      <ColumnLayout columns={3} variant="text-grid">
        <div>
          <Box variant="awsui-key-label">Connection</Box>
          <div>{getStatusIndicator()}</div>
        </div>
        <div>
          <Box variant="awsui-key-label">Session ID</Box>
          <div>
            <Box fontSize="body-s" color="text-body-secondary">
              {sessionId ? sessionId.substring(0, 8) : 'N/A'}
            </Box>
          </div>
        </div>
        <div>
          <Box variant="awsui-key-label">Status</Box>
          <div>
            <Box fontSize="body-s">
              {status.charAt(0).toUpperCase() + status.slice(1)}
            </Box>
          </div>
        </div>
      </ColumnLayout>
    </Container>
  );
};

export default AgentStatus;
