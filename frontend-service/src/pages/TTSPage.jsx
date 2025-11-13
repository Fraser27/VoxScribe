import React, { useState, useEffect } from 'react';
import Container from '@cloudscape-design/components/container';
import Header from '@cloudscape-design/components/header';
import SpaceBetween from '@cloudscape-design/components/space-between';
import ContentLayout from '@cloudscape-design/components/content-layout';
import ModeSelector from '../components/tts/ModeSelector';
import TTSModelSelector from '../components/tts/TTSModelSelector';
import VoiceConfig from '../components/tts/VoiceConfig';
import TextInput from '../components/tts/TextInput';
import TTSActions from '../components/tts/TTSActions';
import TTSProgress from '../components/tts/TTSProgress';
import TTSResults from '../components/tts/TTSResults';
import ServiceStatus from '../components/common/ServiceStatus';
import { useWebSocket } from '../contexts/WebSocketContext';

const TTSPage = () => {
  const [mode, setMode] = useState('single');
  const [selectedEngine, setSelectedEngine] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [text, setText] = useState('');
  const [language, setLanguage] = useState('');
  const [speaker, setSpeaker] = useState('');
  const [description, setDescription] = useState('');
  const [models, setModels] = useState([]);
  const [synthesizing, setSynthesizing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [downloading, setDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [downloadMessage, setDownloadMessage] = useState('');
  const { messages } = useWebSocket();

  useEffect(() => {
    loadModels();
  }, []);

  // Listen for WebSocket messages
  useEffect(() => {
    messages.forEach((msg) => {
      // Handle download progress for TTS models
      if (msg.type === 'download_progress') {
        console.log('TTS download progress:', msg);
        setDownloading(true);
        setDownloadProgress(msg.progress || 0);
        setDownloadMessage(msg.message || `Downloading... ${msg.engine}/${msg.model_id}...`);
        
        // Clear download state when complete
        if (msg.status === 'complete' || msg.status === 'error') {
          setTimeout(() => {
            setDownloading(false);
            setDownloadProgress(0);
            setDownloadMessage('');
            loadModels();
          }, 2000);
        }
      } else if (msg.type === 'download_complete') {
        console.log('TTS download complete:', msg);
        loadModels();
        setDownloading(false);
        setDownloadProgress(0);
        setDownloadMessage('');
      } else if (msg.type === 'synthesis_progress') {
        setProgress(msg.progress || 0);
      } else if (msg.type === 'synthesis_complete') {
        setSynthesizing(false);
        if (msg.success) {
          console.log('Synthesis completed via WebSocket');
        }
      }
    });
  }, [messages]);

  const loadModels = async () => {
    try {
      const response = await fetch('/api/tts/models');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setModels(data.models || []);
    } catch (error) {
      console.error('Failed to load TTS models:', error.message);
      // Set empty models array so UI doesn't break
      setModels([]);
    }
  };

  const handleSynthesize = async () => {
    if (!text) return;

    setSynthesizing(true);
    setProgress(0);
    setResults(null);

    const formData = new FormData();
    formData.append('text', text);
    formData.append('engine', selectedEngine);
    formData.append('model_id', selectedModel);
    
    if (language) formData.append('language', language);
    if (speaker) formData.append('speaker', speaker);
    if (description) formData.append('description', description);

    try {
      const response = await fetch('/api/tts/synthesize', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      // The response is an audio file, create a blob URL
      const blob = await response.blob();
      const audioUrl = URL.createObjectURL(blob);
      
      // Get metadata from headers
      const duration = parseFloat(response.headers.get('X-Audio-Duration') || '0');
      const processingTime = parseFloat(response.headers.get('X-Processing-Time') || '0');
      const sampleRate = parseInt(response.headers.get('X-Sample-Rate') || '0');
      
      setResults({
        audio_url: audioUrl,
        duration,
        processing_time: processingTime,
        sample_rate: sampleRate
      });
      setSynthesizing(false);
    } catch (error) {
      console.error('Synthesis error:', error);
      alert(`Synthesis failed: ${error.message}`);
      setSynthesizing(false);
    }
  };

  return (
    <ContentLayout
      header={
        <Header variant="h1" description="Convert text to natural speech">
          Text-to-Speech
        </Header>
      }
    >
      <SpaceBetween size="l">
        <ServiceStatus models={models} status={null} />
        
        <Container>
          <SpaceBetween size="l">
            <ModeSelector mode={mode} onChange={setMode} />
            
            <TTSModelSelector
              mode={mode}
              models={models}
              selectedEngine={selectedEngine}
              selectedModel={selectedModel}
              onEngineChange={setSelectedEngine}
              onModelChange={setSelectedModel}
            />
            
            <VoiceConfig
              language={language}
              speaker={speaker}
              description={description}
              onLanguageChange={setLanguage}
              onSpeakerChange={setSpeaker}
              onDescriptionChange={setDescription}
            />
            
            <TextInput text={text} onChange={setText} />
            
            <TTSActions
              disabled={!text || synthesizing}
              onSynthesize={handleSynthesize}
            />
          </SpaceBetween>
        </Container>

        {downloading && (
          <Container header={<Header variant="h2">Downloading Model</Header>}>
            <SpaceBetween size="s">
              <TTSProgress progress={downloadProgress} />
              {downloadMessage && <div>{downloadMessage}</div>}
            </SpaceBetween>
          </Container>
        )}

        {synthesizing && (
          <TTSProgress progress={progress} />
        )}

        {results && (
          <TTSResults results={results} />
        )}
      </SpaceBetween>
    </ContentLayout>
  );
};

export default TTSPage;
