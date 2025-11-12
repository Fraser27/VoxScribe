import React from 'react';
import Button from '@cloudscape-design/components/button';
import SpaceBetween from '@cloudscape-design/components/space-between';

const TranscriptionActions = ({ mode, disabled, onTranscribe }) => {
  return (
    <SpaceBetween direction="horizontal" size="xs">
      <Button variant="primary" disabled={disabled} onClick={onTranscribe}>
        {mode === 'single' ? 'Transcribe' : 'Compare Models'}
      </Button>
    </SpaceBetween>
  );
};

export default TranscriptionActions;
