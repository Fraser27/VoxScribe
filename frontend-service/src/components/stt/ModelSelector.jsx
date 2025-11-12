import React, { useMemo, useState } from 'react';
import FormField from '@cloudscape-design/components/form-field';
import Select from '@cloudscape-design/components/select';
import SpaceBetween from '@cloudscape-design/components/space-between';
import Checkbox from '@cloudscape-design/components/checkbox';
import Grid from '@cloudscape-design/components/grid';
import Box from '@cloudscape-design/components/box';
import Button from '@cloudscape-design/components/button';
import Alert from '@cloudscape-design/components/alert';

const ModelSelector = ({
  mode,
  models,
  selectedEngine,
  selectedModel,
  selectedModels,
  onEngineChange,
  onModelChange,
  onModelsChange,
  onRefreshModels
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

  const handleDownload = async (engine, modelId, onRefresh) => {
    const key = `${engine}-${modelId}`;
    setDownloading(prev => ({ ...prev, [key]: true }));

    try {
      const formData = new FormData();
      formData.append('engine', engine);
      formData.append('model_id', modelId);

      const response = await fetch('/api/stt/download-model', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error('Download failed');
      }

      alert(`Download started for ${engine}/${modelId}. The model will be available once download completes.`);
      
      // Refresh models list after a delay to check download status
      if (onRefresh) {
        setTimeout(() => onRefresh(), 2000);
      }
    } catch (error) {
      console.error('Download error:', error);
      alert(`Failed to start download: ${error.message}`);
    } finally {
      setDownloading(prev => ({ ...prev, [key]: false }));
    }
  };

  if (mode === 'single') {
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
                onClick={() => handleDownload(selectedEngine, selectedModel, onRefreshModels)}
                loading={downloading[`${selectedEngine}-${selectedModel}`]}
              >
                Download Model
              </Button>
            }
          >
            This model is not downloaded. Download it before transcribing.
          </Alert>
        )}
      </SpaceBetween>
    );
  }

  return (
    <FormField label="Select Models to Compare (minimum 2)">
      <Grid gridDefinition={[{ colspan: 6 }, { colspan: 6 }]}>
        {(!models || models.length === 0) ? (
          <Box color="text-status-inactive">No models available. Please check backend services.</Box>
        ) : (
          models.map((model) => (
          <SpaceBetween key={`${model.engine}-${model.model_id}`} size="xs" direction="horizontal">
            <Checkbox
              checked={selectedModels.some(
                m => m.engine === model.engine && m.model_id === model.model_id
              )}
              onChange={({ detail }) => {
                if (detail.checked) {
                  onModelsChange([...selectedModels, model]);
                } else {
                  onModelsChange(
                    selectedModels.filter(
                      m => !(m.engine === model.engine && m.model_id === model.model_id)
                    )
                  );
                }
              }}
              disabled={!model.cached}
            >
              {model.engine} - {model.model_id} {model.cached ? '✓' : ''}
            </Checkbox>
            {!model.cached && (
              <Button
                variant="inline-link"
                onClick={() => handleDownload(model.engine, model.model_id, onRefreshModels)}
                loading={downloading[`${model.engine}-${model.model_id}`]}
              >
                Download
              </Button>
            )}
          </SpaceBetween>
          ))
        )}
      </Grid>
    </FormField>
  );
};

export default ModelSelector;
