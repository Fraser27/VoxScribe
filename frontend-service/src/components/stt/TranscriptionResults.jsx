import React, { useMemo } from 'react';
import Container from '@cloudscape-design/components/container';
import Header from '@cloudscape-design/components/header';
import Box from '@cloudscape-design/components/box';
import ColumnLayout from '@cloudscape-design/components/column-layout';
import Table from '@cloudscape-design/components/table';

const TranscriptionResults = ({ results, mode }) => {
  if (!results) return null;

  // Extract text from csv_data
  const extractText = (csvData) => {
    if (!csvData || csvData.length <= 1) return '';
    // Skip header row and concatenate all transcription text
    return csvData.slice(1).map(row => row[row.length - 1]).join(' ');
  };

  if (mode === 'single') {
    const transcriptionText = results.text || extractText(results.csv_data);
    
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
          <Box padding={{ top: 's' }}>{transcriptionText}</Box>
        </Box>
      </Container>
    );
  }

  // For comparison mode
  const comparisonItems = useMemo(() => {
    if (!results.results) return [];
    return Object.entries(results.results).map(([key, result]) => ({
      key,
      ...result,
      text: result.text || extractText(result.csv_data)
    }));
  }, [results]);

  return (
    <Container header={<Header variant="h2">Comparison Results</Header>}>
      <Table
        columnDefinitions={[
          { header: 'Model', cell: item => item.key },
          { header: 'Duration', cell: item => `${item.duration?.toFixed(2)}s` },
          { header: 'Processing Time', cell: item => `${item.processing_time?.toFixed(2)}s` },
          { header: 'RTFx', cell: item => `${item.rtfx?.toFixed(2)}x` },
          { header: 'Transcription', cell: item => item.text }
        ]}
        items={comparisonItems}
        variant="embedded"
      />
    </Container>
  );
};

export default TranscriptionResults;
