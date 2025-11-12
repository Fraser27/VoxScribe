import React from 'react';
import Container from '@cloudscape-design/components/container';
import Header from '@cloudscape-design/components/header';
import Box from '@cloudscape-design/components/box';
import ColumnLayout from '@cloudscape-design/components/column-layout';
import Table from '@cloudscape-design/components/table';

const TranscriptionResults = ({ results, mode }) => {
  if (!results) return null;

  if (mode === 'single') {
    return (
      <Container header={<Header variant="h2">Transcription Result</Header>}>
        <ColumnLayout columns={3} variant="text-grid">
          <div>
            <Box variant="awsui-key-label">Duration</Box>
            <Box>{results.duration?.toFixed(2)}s</Box>
          </div>
          <div>
            <Box variant="awsui-key-label">Processing Time</Box>
            <Box>{results.processing_time?.toFixed(2)}s</Box>
          </div>
          <div>
            <Box variant="awsui-key-label">RTFx</Box>
            <Box>{results.rtfx?.toFixed(2)}x</Box>
          </div>
        </ColumnLayout>
        <Box margin={{ top: 'l' }}>
          <Box variant="awsui-key-label">Transcription</Box>
          <Box padding={{ top: 's' }}>{results.text}</Box>
        </Box>
      </Container>
    );
  }

  return (
    <Container header={<Header variant="h2">Comparison Results</Header>}>
      <Table
        columnDefinitions={[
          { header: 'Model', cell: item => `${item.engine} - ${item.model_id}` },
          { header: 'Duration', cell: item => `${item.duration?.toFixed(2)}s` },
          { header: 'Processing Time', cell: item => `${item.processing_time?.toFixed(2)}s` },
          { header: 'RTFx', cell: item => `${item.rtfx?.toFixed(2)}x` },
          { header: 'Transcription', cell: item => item.text }
        ]}
        items={results.results || []}
        variant="embedded"
      />
    </Container>
  );
};

export default TranscriptionResults;
