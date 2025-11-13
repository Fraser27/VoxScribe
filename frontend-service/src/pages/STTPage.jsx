import React, { useState, useEffect } from 'react';
import Container from '@cloudscape-design/components/container';
import Header from '@cloudscape-design/components/header';
import SpaceBetween from '@cloudscape-design/components/space-between';
import ContentLayout from '@cloudscape-design/components/content-layout';
import ModeSelector from '../components/stt/ModeSelector';
import ModelSelector from '../components/stt/ModelSelector';
import FileUpload from '../components/stt/FileUpload';
import TranscriptionActions from '../components/stt/TranscriptionActions';
import TranscriptionProgress from '../components/stt/TranscriptionProgress';
import TranscriptionResults from '../components/stt/TranscriptionResults';
import StatusBar from '../components/common/StatusBar';
import ServiceStatus from '../components/common/ServiceStatus';
import { useWebSocket } from '../contexts/WebSocketContext';

const STTPage = () => {
  const [mode, setMode] = useState('single');
  const [selectedEngine, setSelectedEngine] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [selectedModels, setSelectedModels] = useState([]);
  const [audioFile, setAudioFile] = useState(null);
  const [models, setModels] = useState([]);
  const [status, setStatus] = useState(null);
  const [transcribing, setTranscribing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [downloading, setDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState(0);
  const [downloadMessage, setDownloadMessage] = useState('');
  const { messages } = useWebSocket();

  useEffect(() => {
    loadModels();
    loadStatus();
  }, []);

  useEffect(() => {
    messages.forEach((msg) => {
      // Handle download progress for STT models
      if (msg.type === 'download_progress') {
        console.log('STT download progress:', msg);
        setDownloading(true);
        setDownloadProgress(msg.progress || 0);
        setDownloadMessage(msg.message || `Downloading ${msg.engine}/${msg.model_id}...`);
        
        // Clear download state when complete
        if (msg.status === 'complete' || msg.status === 'error') {
          setTimeout(() => {
            setDownloading(false);
            setDownloadProgress(0);
            setDownloadMessage('');
          }, 2000);
        }
      } else if (msg.type === 'download_complete') {
        console.log('STT download complete:', msg);
        loadModels();
        setDownloading(false);
        setDownloadProgress(0);
        setDownloadMessage('');
      } else if (msg.type === 'transcription_progress') {
        setProgress(msg.progress || 0);
      } else if (msg.type === 'transcription_complete') {
        setTimeout(() => {
            setTranscribing(false);  
        }, 2000);
      }
    });
  }, [messages]);

  const loadModels = async () => {
    try {
      const response = await fetch('/api/stt/models');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setModels(data.models || []);
    } catch (error) {
      console.error('Failed to load models:', error.message);
      // Set empty models array so UI doesn't break
      setModels([]);
    }
  };

  const loadStatus = async () => {
    try {
      const response = await fetch('/api/stt/status');
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      const data = await response.json();
      setStatus(data);
    } catch (error) {
      console.error('Failed to load status:', error.message);
      // Set default status so UI doesn't break
      setStatus({ device: 'unknown', dependencies_ready: false, models_loaded: 0 });
    }
  };

  const handleTranscribe = async () => {
    if (!audioFile) return;

    setTranscribing(true);
    setProgress(0);
    setResults(null);

    const formData = new FormData();
    formData.append('file', audioFile);
    
    if (mode === 'single') {
      formData.append('engine', selectedEngine);
      formData.append('model_id', selectedModel);
    } else {
      formData.append('engines', JSON.stringify(selectedModels));
    }

    try {
      const endpoint = mode === 'single' ? '/api/stt/transcribe' : '/api/stt/compare';
      const response = await fetch(endpoint, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setResults(data);
      setTranscribing(false);
    } catch (error) {
      console.error('Transcription error:', error);
      alert(`Transcription failed: ${error.message}`);
      setTranscribing(false);
    }
  };

  return (
    <ContentLayout
      header={
        <Header variant="h1" description="Compare and evaluate Speech-to-Text models">
          Speech-to-Text
        </Header>
      }
    >
      <SpaceBetween size="l">
        <ServiceStatus models={models} status={status} />
        
        <StatusBar status={status} />
        
        <Container>
          <SpaceBetween size="l">
            <ModeSelector mode={mode} onChange={setMode} />
            
            <ModelSelector
              mode={mode}
              models={models}
              selectedEngine={selectedEngine}
              selectedModel={selectedModel}
              selectedModels={selectedModels}
              onEngineChange={setSelectedEngine}
              onModelChange={setSelectedModel}
              onModelsChange={setSelectedModels}
              onRefreshModels={loadModels}
            />
            
            <FileUpload
              audioFile={audioFile}
              onFileSelect={setAudioFile}
            />
            
            <TranscriptionActions
              mode={mode}
              models={models}
              selectedEngine={selectedEngine}
              selectedModel={selectedModel}
              selectedModels={selectedModels}
              disabled={!audioFile || transcribing}
              onTranscribe={handleTranscribe}
            />
          </SpaceBetween>
        </Container>

        {downloading && (
          <Container header={<Header variant="h2">Downloading Model</Header>}>
            <SpaceBetween size="s">
              <TranscriptionProgress progress={downloadProgress} />
              {downloadMessage && <div>{downloadMessage}</div>}
            </SpaceBetween>
          </Container>
        )}

        {transcribing && (
          <TranscriptionProgress progress={progress} />
        )}

        {results && (
          <TranscriptionResults results={results} mode={mode} />
        )}
      </SpaceBetween>
    </ContentLayout>
  );
};

export default STTPage;
