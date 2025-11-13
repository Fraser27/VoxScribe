import React from 'react';
import Container from '@cloudscape-design/components/container';
import ProgressBar from '@cloudscape-design/components/progress-bar';
import Box from '@cloudscape-design/components/box';

const DownloadProgress = ({ progress }, {message}) => {
  return (
    <Container>
      <Box textAlign="center">
        <ProgressBar
          value={progress}
          label="Download progress..."
          description={message}
        />
      </Box>
    </Container>
  );
};

export default DownloadProgress;