import React from 'react';
import Container from '@cloudscape-design/components/container';
import Header from '@cloudscape-design/components/header';
import Box from '@cloudscape-design/components/box';
import ColumnLayout from '@cloudscape-design/components/column-layout';

const TTSResults = ({ results }) => {
  if (!results) return null;

  return (
    <Container header={<Header variant="h2">Synthesized Audio</Header>}>
      <ColumnLayout columns={2} variant="text-grid">
        <div>
          <Box variant="awsui-key-label">Processing Time</Box>
          <Box>{results.processing_time?.toFixed(2)}s</Box>
        </div>
        <div>
          <Box variant="awsui-key-label">Audio Duration</Box>
          <Box>{results.duration?.toFixed(2)}s</Box>
        </div>
      </ColumnLayout>
      
      {results.audio_url && (
        <Box margin={{ top: 'l' }}>
          <audio controls src={results.audio_url} style={{ width: '100%' }} />
        </Box>
      )}
    </Container>
  );
};

export default TTSResults;
