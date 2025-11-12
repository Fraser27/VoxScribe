import React, { useRef } from 'react';
import FormField from '@cloudscape-design/components/form-field';
import Button from '@cloudscape-design/components/button';
import Box from '@cloudscape-design/components/box';
import SpaceBetween from '@cloudscape-design/components/space-between';

const FileUpload = ({ audioFile, onFileSelect }) => {
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      onFileSelect(file);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  };

  return (
    <FormField label="Audio File" description="Supported: WAV, MP3, FLAC, M4A, OGG">
      <SpaceBetween size="s">
        <input
          ref={fileInputRef}
          type="file"
          accept=".wav,.mp3,.flac,.m4a,.ogg"
          onChange={handleFileChange}
          style={{ display: 'none' }}
        />
        
        <Button onClick={() => fileInputRef.current?.click()}>
          Choose file
        </Button>

        {audioFile && (
          <Box>
            <SpaceBetween size="xs">
              <Box variant="strong">{audioFile.name}</Box>
              <Box variant="small">{formatFileSize(audioFile.size)}</Box>
              <audio controls src={URL.createObjectURL(audioFile)} style={{ width: '100%' }} />
            </SpaceBetween>
          </Box>
        )}
      </SpaceBetween>
    </FormField>
  );
};

export default FileUpload;
