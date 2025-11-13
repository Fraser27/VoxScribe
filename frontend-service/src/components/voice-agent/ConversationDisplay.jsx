import React, { useEffect, useRef } from 'react';
import {
  Box,
  Container,
  SpaceBetween,
  Header,
  StatusIndicator
} from '@cloudscape-design/components';

const ConversationDisplay = ({ conversation, isProcessing }) => {
  const endOfMessagesRef = useRef(null);
  
  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [conversation]);
  
  return (
    <Container
      header={
        <Header
          variant="h2"
          counter={`(${conversation.length})`}
        >
          Conversation
        </Header>
      }
    >
      <Box padding="s">
        {conversation.length === 0 ? (
          <Box textAlign="center" color="text-body-secondary" padding="xxl">
            No messages yet. Click "Start Speaking" to begin a conversation.
          </Box>
        ) : (
          <SpaceBetween size="m">
            {conversation.map((message, index) => (
              <ConversationMessage
                key={index}
                role={message.role}
                text={message.text}
                timestamp={message.timestamp}
              />
            ))}
            {isProcessing && (
              <Box textAlign="center" padding="s">
                <StatusIndicator type="loading">Processing...</StatusIndicator>
              </Box>
            )}
            <div ref={endOfMessagesRef} />
          </SpaceBetween>
        )}
      </Box>
    </Container>
  );
};

const ConversationMessage = ({ role, text, timestamp }) => {
  const isUser = role === 'user';
  
  return (
    <Box
      padding="m"
      backgroundColor={isUser ? 'color-background-container-content' : 'color-background-layout-main'}
      borderRadius="8px"
    >
      <SpaceBetween size="xs">
        <Box fontSize="body-s" fontWeight="bold" color={isUser ? 'text-status-info' : 'text-status-success'}>
          {isUser ? '👤 You' : '🤖 Assistant'}
        </Box>
        <Box>{text}</Box>
        <Box fontSize="body-xs" color="text-body-secondary">
          {new Date(timestamp).toLocaleTimeString()}
        </Box>
      </SpaceBetween>
    </Box>
  );
};

export default ConversationDisplay;
