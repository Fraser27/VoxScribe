import React from 'react';
import Container from '@cloudscape-design/components/container';
import ColumnLayout from '@cloudscape-design/components/column-layout';
import Box from '@cloudscape-design/components/box';
import StatusIndicator from '@cloudscape-design/components/status-indicator';

const StatusBar = ({ status }) => {
  if (!status) return null;

  const getDeviceStatus = () => {
    if (status.device === 'cuda') return 'success';
    if (status.device === 'mps') return 'success';
    return 'info';
  };

  return (
    <Container>
      <ColumnLayout columns={3} variant="text-grid">
        <div>
          <Box variant="awsui-key-label">Device</Box>
          <StatusIndicator type={getDeviceStatus()}>
            {status.device?.toUpperCase() || 'CPU'}
          </StatusIndicator>
        </div>
        <div>
          <Box variant="awsui-key-label">Dependencies</Box>
          <StatusIndicator type={status.dependencies_ready ? 'success' : 'warning'}>
            {status.dependencies_ready ? 'Ready' : 'Checking'}
          </StatusIndicator>
        </div>
        <div>
          <Box variant="awsui-key-label">Models Loaded</Box>
          <Box>{status.models_loaded || 0}</Box>
        </div>
      </ColumnLayout>
    </Container>
  );
};

export default StatusBar;
