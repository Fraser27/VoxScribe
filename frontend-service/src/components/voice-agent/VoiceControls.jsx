import React from 'react';
import { Box, Button, SpaceBetween, Icon } from '@cloudscape-design/components';

const VoiceControls = ({
  isConnected,
  isRecording,
  isSpeaking,
  onStartRecording,
  onStopRecording,
  onClearConversation
}) => {
  return (
    <Box textAlign="center" padding="l">
      <SpaceBetween size="m" direction="horizontal" alignItems="center">
        {!isRecording ? (
          <Button
            variant="primary"
            iconName="microphone"
            disabled={!isConnected || isSpeaking}
            onClick={onStartRecording}
          >
            Start Speaking
          </Button>
        ) : (
          <Button
            variant="normal"
            iconName="close"
            onClick={onStopRecording}
          >
            Stop Speaking
          </Button>
        )}
        
        <Button
          variant="normal"
          iconName="remove"
          disabled={!isConnected}
          onClick={onClearConversation}
        >
          Clear Conversation
        </Button>
      </SpaceBetween>
      
      {isRecording && (
        <Box margin={{ top: 'm' }} color="text-status-info">
          <Icon name="status-in-progress" variant="link" /> Recording... Click "Stop Speaking" when done
        </Box>
      )}
      
      {isSpeaking && (
        <Box margin={{ top: 'm' }} color="text-status-info">
          <Icon name="status-in-progress" variant="link" /> Agent is speaking...
        </Box>
      )}
    </Box>
  );
};

export default VoiceControls;
