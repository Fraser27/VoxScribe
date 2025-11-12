import React, { useMemo } from 'react';
import Button from '@cloudscape-design/components/button';
import SpaceBetween from '@cloudscape-design/components/space-between';

const TranscriptionActions = ({ mode, models, selectedEngine, selectedModel, selectedModels, disabled, onTranscribe }) => {
  const isModelCached = useMemo(() => {
    if (mode === 'single') {
      if (!selectedEngine || !selectedModel) return false;
      const model = models.find(m => m.engine === selectedEngine && m.model_id === selectedModel);
      return model?.cached || false;
    } else {
      // For compare mode, all selected models must be cached
      return selectedModels.every(sm => {
        const model = models.find(m => m.engine === sm.engine && m.model_id === sm.model_id);
        return model?.cached || false;
      });
    }
  }, [mode, models, selectedEngine, selectedModel, selectedModels]);

  return (
    <SpaceBetween direction="horizontal" size="xs">
      <Button 
        variant="primary" 
        disabled={disabled || !isModelCached} 
        onClick={onTranscribe}
      >
        {mode === 'single' ? 'Transcribe' : 'Compare Models'}
      </Button>
    </SpaceBetween>
  );
};

export default TranscriptionActions;
