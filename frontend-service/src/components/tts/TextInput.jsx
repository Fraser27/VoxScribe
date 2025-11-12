import React from 'react';
import FormField from '@cloudscape-design/components/form-field';
import Textarea from '@cloudscape-design/components/textarea';
import Box from '@cloudscape-design/components/box';

const TextInput = ({ text, onChange }) => {
  return (
    <FormField
      label="Text Input"
      description={
        <Box float="right" color="text-body-secondary">
          {text.length} characters
        </Box>
      }
    >
      <Textarea
        value={text}
        onChange={({ detail }) => onChange(detail.value)}
        placeholder="Enter the text you want to convert to speech..."
        rows={6}
      />
    </FormField>
  );
};

export default TextInput;
