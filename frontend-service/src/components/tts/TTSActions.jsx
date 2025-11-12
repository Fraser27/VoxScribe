import React from 'react';
import Button from '@cloudscape-design/components/button';
import SpaceBetween from '@cloudscape-design/components/space-between';

const TTSActions = ({ disabled, onSynthesize }) => {
  return (
    <SpaceBetween direction="horizontal" size="xs">
      <Button variant="primary" disabled={disabled} onClick={onSynthesize}>
        Synthesize
      </Button>
    </SpaceBetween>
  );
};

export default TTSActions;
