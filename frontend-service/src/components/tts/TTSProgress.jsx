import React from 'react';
import Container from '@cloudscape-design/components/container';
import ProgressBar from '@cloudscape-design/components/progress-bar';
import Box from '@cloudscape-design/components/box';

const TTSProgress = ({ progress }) => {
  return (
    <Container>
      <Box textAlign="center">
        <ProgressBar
          value={progress}
          label="Synthesis progress"
          description="Generating speech..."
        />
      </Box>
    </Container>
  );
};

export default TTSProgress;
