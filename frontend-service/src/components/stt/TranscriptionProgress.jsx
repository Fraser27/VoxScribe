import React from 'react';
import Container from '@cloudscape-design/components/container';
import ProgressBar from '@cloudscape-design/components/progress-bar';
import Box from '@cloudscape-design/components/box';

const TranscriptionProgress = ({ progress }) => {
  return (
    <Container>
      <Box textAlign="center">
        <ProgressBar
          value={progress}
          label="Transcription progress"
          description="Processing audio file..."
        />
      </Box>
    </Container>
  );
};

export default TranscriptionProgress;
