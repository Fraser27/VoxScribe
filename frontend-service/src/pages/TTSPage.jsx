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

  useEffect(() => {
    loadModels();
  }, []);

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

    try {
      const response = await fetch('/api/tts/synthesize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          engine: selectedEngine,
          model_id: selectedModel,
          language: language || undefined,
          speaker: speaker || undefined,
          description: description || undefined
        })
      });

      if (!response.ok) {
        throw new Error('Synthesis failed');
      }

      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error('Synthesis error:', error);
    } finally {
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
