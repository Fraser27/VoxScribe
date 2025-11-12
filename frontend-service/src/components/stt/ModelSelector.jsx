import React, { useMemo } from 'react';
import FormField from '@cloudscape-design/components/form-field';
import Select from '@cloudscape-design/components/select';
import SpaceBetween from '@cloudscape-design/components/space-between';
import Checkbox from '@cloudscape-design/components/checkbox';
import Grid from '@cloudscape-design/components/grid';
import Box from '@cloudscape-design/components/box';

const ModelSelector = ({
  mode,
  models,
  selectedEngine,
  selectedModel,
  selectedModels,
  onEngineChange,
  onModelChange,
  onModelsChange
}) => {
  const engines = useMemo(() => {
    if (!models || models.length === 0) return [];
    const uniqueEngines = [...new Set(models.map(m => m.engine))];
    return uniqueEngines.map(engine => ({ label: engine, value: engine }));
  }, [models]);

  const availableModels = useMemo(() => {
    if (!models || models.length === 0) return [];
    return models
      .filter(m => m.engine === selectedEngine)
      .map(m => ({ label: m.model_id, value: m.model_id }));
  }, [models, selectedEngine]);

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
          <Checkbox
            key={`${model.engine}-${model.model_id}`}
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
          >
            {model.engine} - {model.model_id}
          </Checkbox>
          ))
        )}
      </Grid>
    </FormField>
  );
};

export default ModelSelector;
