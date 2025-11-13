import React, { useState, useEffect, useRef } from 'react';
import {
  Container,
  Header,
  SpaceBetween,
  Button,
  Box,
  Alert,
  StatusIndicator,
  ColumnLayout,
  Cards,
  Badge
} from '@cloudscape-design/components';
import VoiceControls from '../components/voice-agent/VoiceControls';
import ConversationDisplay from '../components/voice-agent/ConversationDisplay';
import AgentStatus from '../components/voice-agent/AgentStatus';

const VoiceAgentPage = () => {
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [status, setStatus] = useState('idle');
  const [conversation, setConversation] = useState([]);
  const [error, setError] = useState(null);
  const [sessionId, setSessionId] = useState(null);
  
  const wsRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const audioQueueRef = useRef([]);
  const isPlayingRef = useRef(false);
  
  // WebSocket connection
  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      stopRecording();
    };
  }, []);
  
  const connectWebSocket = () => {
    const agentUrl = import.meta.env.VITE_AGENT_SERVICE_URL || 'ws://localhost:8003';
    const ws = new WebSocket(`${agentUrl}/ws/agent`);
    
    ws.onopen = () => {
      console.log('Connected to agent service');
      setIsConnected(true);
      setError(null);
    };
    
    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      handleWebSocketMessage(message);
    };
    
    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Connection error');
      setIsConnected(false);
    };
    
    ws.onclose = () => {
      console.log('Disconnected from agent service');
      setIsConnected(false);
      // Attempt reconnect after 3 seconds
      setTimeout(connectWebSocket, 3000);
    };
    
    wsRef.current = ws;
  };
  
  const handleWebSocketMessage = (message) => {
    console.log('Received message:', message.type);
    
    switch (message.type) {
      case 'session_created':
        setSessionId(message.session_id);
        break;
      
      case 'status':
        setStatus(message.status);
        break;
      
      case 'transcription_final':
        addToConversation('user', message.text);
        break;
      
      case 'agent_response_chunk':
        updateLastAssistantMessage(message.text);
        break;
      
      case 'audio_chunk':
        queueAudioChunk(message.data);
        break;
      
      case 'audio_complete':
        setIsSpeaking(false);
        setStatus('idle');
        break;
      
      case 'response_complete':
        setStatus('idle');
        break;
      
      case 'error':
        setError(message.message);
        setStatus('idle');
        break;
      
      default:
        console.log('Unknown message type:', message.type);
    }
  };
  
  const addToConversation = (role, text) => {
    setConversation(prev => [...prev, {
      role,
      text,
      timestamp: new Date().toISOString()
    }]);
  };
  
  const updateLastAssistantMessage = (chunk) => {
    setConversation(prev => {
      const newConv = [...prev];
      const lastMsg = newConv[newConv.length - 1];
      
      if (lastMsg && lastMsg.role === 'assistant') {
        lastMsg.text += chunk;
      } else {
        newConv.push({
          role: 'assistant',
          text: chunk,
          timestamp: new Date().toISOString()
        });
      }
      
      return newConv;
    });
    
    setIsSpeaking(true);
  };
  
  const queueAudioChunk = (base64Data) => {
    audioQueueRef.current.push(base64Data);
    if (!isPlayingRef.current) {
      playNextAudioChunk();
    }
  };
  
  const playNextAudioChunk = async () => {
    if (audioQueueRef.current.length === 0) {
      isPlayingRef.current = false;
      return;
    }
    
    isPlayingRef.current = true;
    const base64Data = audioQueueRef.current.shift();
    
    try {
      // Decode base64 to array buffer
      const binaryString = atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      // Create audio context if needed
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      
      // Decode and play audio
      const audioBuffer = await audioContextRef.current.decodeAudioData(bytes.buffer);
      const source = audioContextRef.current.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioContextRef.current.destination);
      
      source.onended = () => {
        playNextAudioChunk();
      };
      
      source.start(0);
    } catch (error) {
      console.error('Error playing audio chunk:', error);
      playNextAudioChunk();
    }
  };
  
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm'
      });
      
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && wsRef.current?.readyState === WebSocket.OPEN) {
          // Convert blob to base64 and send
          const reader = new FileReader();
          reader.onloadend = () => {
            const base64Data = reader.result.split(',')[1];
            wsRef.current.send(JSON.stringify({
              type: 'audio_chunk',
              data: base64Data,
              session_id: sessionId
            }));
          };
          reader.readAsDataURL(event.data);
        }
      };
      
      mediaRecorder.start(100); // Capture in 100ms chunks
      mediaRecorderRef.current = mediaRecorder;
      setIsRecording(true);
      setStatus('listening');
      setError(null);
    } catch (error) {
      console.error('Error starting recording:', error);
      setError('Microphone access denied');
    }
  };
  
  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      mediaRecorderRef.current = null;
    }
    
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'end_speech',
        session_id: sessionId
      }));
    }
    
    setIsRecording(false);
    setStatus('processing');
  };
  
  const clearConversation = () => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'clear_history',
        session_id: sessionId
      }));
    }
    setConversation([]);
  };
  
  return (
    <Container
      header={
        <Header
          variant="h1"
          description="Real-time voice conversation with AI"
        >
          Voice Agent
        </Header>
      }
    >
      <SpaceBetween size="l">
        {error && (
          <Alert type="error" dismissible onDismiss={() => setError(null)}>
            {error}
          </Alert>
        )}
        
        <AgentStatus
          isConnected={isConnected}
          status={status}
          sessionId={sessionId}
        />
        
        <VoiceControls
          isConnected={isConnected}
          isRecording={isRecording}
          isSpeaking={isSpeaking}
          onStartRecording={startRecording}
          onStopRecording={stopRecording}
          onClearConversation={clearConversation}
        />
        
        <ConversationDisplay
          conversation={conversation}
          isProcessing={status !== 'idle'}
        />
      </SpaceBetween>
    </Container>
  );
};

export default VoiceAgentPage;
