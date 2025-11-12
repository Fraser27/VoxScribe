import React from 'react';
import FormField from '@cloudscape-design/components/form-field';
import Select from '@cloudscape-design/components/select';
import Textarea from '@cloudscape-design/components/textarea';
import SpaceBetween from '@cloudscape-design/components/space-between';

const VoiceConfig = ({
  language,
  speaker,
  description,
  onLanguageChange,
  onSpeakerChange,
  onDescriptionChange
}) => {
  const languageOptions = [
    { label: 'Auto-detect', value: '' },
    { label: 'English', value: 'en' },
    { label: 'Spanish', value: 'es' },
    { label: 'French', value: 'fr' },
    { label: 'German', value: 'de' }
  ];

  const speakerOptions = [
    { label: 'Default', value: '' },
    { label: 'Speaker 1', value: 'speaker_1' },
    { label: 'Speaker 2', value: 'speaker_2' }
  ];

  return (
    <SpaceBetween size="m">
      <FormField label="Language (optional)">
        <Select
          selectedOption={languageOptions.find(l => l.value === language) || languageOptions[0]}
          onChange={({ detail }) => onLanguageChange(detail.selectedOption.value)}
          options={languageOptions}
        />
      </FormField>

      <FormField label="Speaker (optional)">
        <Select
          selectedOption={speakerOptions.find(s => s.value === speaker) || speakerOptions[0]}
          onChange={({ detail }) => onSpeakerChange(detail.selectedOption.value)}
          options={speakerOptions}
        />
      </FormField>

      <FormField
        label="Voice Description"
        description="Describe the desired voice characteristics, speed, pitch, and style"
      >
        <Textarea
          value={description}
          onChange={({ detail }) => onDescriptionChange(detail.value)}
          placeholder="e.g., A female speaker with a calm and clear voice"
          rows={3}
        />
      </FormField>
    </SpaceBetween>
  );
};

export default VoiceConfig;
