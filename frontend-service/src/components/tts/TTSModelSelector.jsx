import React, { useMemo, useState } from 'react';
import FormField from '@cloudscape-design/components/form-field';
import Select from '@cloudscape-design/components/select';
import SpaceBetween from '@cloudscape-design/components/space-between';
import Button from '@cloudscape-design/components/button';
import Alert from '@cloudscape-design/components/alert';

const TTSModelSelector = ({
  mode,
  models,
  selectedEngine,
  selectedModel,
  onEngineChange,
  onModelChange
}) => {
  const [downloading, setDownloading] = useState({});

  const engines = useMemo(() => {
    if (!models || models.length === 0) return [];
    const uniqueEngines = [...new Set(models.map(m => m.engine))];
    return uniqueEngines.map(engine => ({ label: engine, value: engine }));
  }, [models]);

  const availableModels = useMemo(() => {
    if (!models || models.length === 0) return [];
    return models
      .filter(m => m.engine === selectedEngine)
      .map(m => ({ 
        label: `${m.model_id} ${m.cached ? '✓' : '(not downloaded)'}`, 
        value: m.model_id,
        cached: m.cached
      }));
  }, [models, selectedEngine]);

  const selectedModelData = useMemo(() => {
    if (!selectedEngine || !selectedModel) return null;
    return models.find(m => m.engine === selectedEngine && m.model_id === selectedModel);
  }, [models, selectedEngine, selectedModel]);

  const handleDownload = async (engine, modelId) => {
    const key = `${engine}-${modelId}`;
    setDownloading(prev => ({ ...prev, [key]: true }));

    try {
      const formData = new FormData();
      formData.append('engine', engine);
      formData.append('model_id', modelId);

      const response = await fetch('/api/tts/download-model', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Download failed');
      }

      alert(`Download started for ${engine}/${modelId}. Check the progress in the UI.`);
    } catch (error) {
      console.error('Download error:', error);
      alert(`Failed to start download: ${error.message}`);
    } finally {
      setDownloading(prev => ({ ...prev, [key]: false }));
    }
  };

  return (
    <SpaceBetween size="m">
      <FormField label="Engine">
        <Select
          selectedOption={engines.find(e => e.value === selectedEngine) || null}
          onChange={({ detail }) => onEngineChange(detail.selectedOption.value)}
          options={engines}
          placeholder="Select engine"
        />
      </FormField>
      
      <FormField label="Model">
        <Select
          selectedOption={availableModels.find(m => m.value === selectedModel) || null}
          onChange={({ detail }) => onModelChange(detail.selectedOption.value)}
          options={availableModels}
          placeholder="Select model"
          disabled={!selectedEngine}
        />
      </FormField>

      {selectedModelData && !selectedModelData.cached && (
        <Alert
          type="warning"
          action={
            <Button
              onClick={() => handleDownload(selectedEngine, selectedModel)}
              loading={downloading[`${selectedEngine}-${selectedModel}`]}
            >
              Download Model
            </Button>
          }
        >
          This model is not downloaded. Download it before synthesizing.
        </Alert>
      )}
    </SpaceBetween>
  );
};

export default TTSModelSelector;
