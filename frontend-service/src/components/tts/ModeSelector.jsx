import React from 'react';
import FormField from '@cloudscape-design/components/form-field';
import RadioGroup from '@cloudscape-design/components/radio-group';

const ModeSelector = ({ mode, onChange }) => {
  return (
    <FormField label="Mode Selection">
      <RadioGroup
        value={mode}
        onChange={({ detail }) => onChange(detail.value)}
        items={[
          { value: 'single', label: 'Single Model' },
          { value: 'compare', label: 'Compare Models' }
        ]}
      />
    </FormField>
  );
};

export default ModeSelector;
