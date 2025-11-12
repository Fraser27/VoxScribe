import React, { useMemo } from 'react';
import FormField from '@cloudscape-design/components/form-field';
import Select from '@cloudscape-design/components/select';
import SpaceBetween from '@cloudscape-design/components/space-between';

const TTSModelSelector = ({
  mode,
  models,
  selectedEngine,
  selectedModel,
  onEngineChange,
  onModelChange
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
};

export default TTSModelSelector;
