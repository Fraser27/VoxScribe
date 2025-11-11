// TTS-specific functionality for VoxScribe

class TTSManager {
    constructor(voxscribe) {
        this.voxscribe = voxscribe;
        this.currentMode = 'single';
        this.availableModels = [];
        this.selectedText = '';
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadTTSModels();
    }
    
    setupEventListeners() {
        // Tab switching is handled by main app.js, we just listen for tab changes
        
        // TTS mode selection
        document.querySelectorAll('input[name="tts-mode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.currentMode = e.target.value;
                this.updateTTSUI();
            });
        });
        
        // TTS model selection
        document.getElementById('ttsEngineSelect').addEventListener('change', () => {
            this.updateTTSModelSelect();
        });
        
        document.getElementById('ttsModelSelect').addEventListener('change', () => {
            this.updateTTSDownloadButton();
            this.updateLanguagesAndSpeakers();
            this.updateTTSUI();
        });
        
        // Download TTS model button
        document.getElementById('downloadTTSModelBtn').addEventListener('click', () => {
            this.handleDownloadTTSModelClick();
        });
        
        // Text input
        const textInput = document.getElementById('ttsTextInput');
        textInput.addEventListener('input', () => {
            this.selectedText = textInput.value;
            this.updateCharCount();
            this.updateTTSUI();
        });
        
        // Synthesize button
        document.getElementById('synthesizeBtn').addEventListener('click', () => {
            this.handleSynthesizeClick();
        });
    }
    

    async loadTTSModels() {
        try {
            const response = await fetch('/api/tts/models');
            const data = await response.json();
            
            this.availableModels = data.models;
            this.updateTTSModelSelects();
            
        } catch (error) {
            console.error('Failed to load TTS models:', error);
            this.voxscribe.showToast('Failed to load TTS models', 'error');
        }
    }
    
    updateTTSModelSelects() {
        const engineSelect = document.getElementById('ttsEngineSelect');
        const engines = [...new Set(this.availableModels.map(m => m.engine))];
        
        engineSelect.innerHTML = '<option value="">Select Engine...</option>';
        engines.forEach(engine => {
            const option = document.createElement('option');
            option.value = engine;
            option.textContent = engine.charAt(0).toUpperCase() + engine.slice(1);
            engineSelect.appendChild(option);
        });
        
        this.updateTTSModelSelect();
    }
    
    updateTTSModelSelect() {
        const engineSelect = document.getElementById('ttsEngineSelect');
        const modelSelect = document.getElementById('ttsModelSelect');
        const selectedEngine = engineSelect.value;
        
        modelSelect.innerHTML = '<option value="">Select Model...</option>';
        
        if (!selectedEngine) return;
        
        const models = this.availableModels.filter(m => m.engine === selectedEngine);
        models.forEach(model => {
            const option = document.createElement('option');
            option.value = model.model_id;
            option.textContent = `${model.display_name} (${model.size})`;
            if (!model.cached) {
                option.textContent += ' - Not Downloaded';
            }
            modelSelect.appendChild(option);
        });
        
        this.updateTTSDownloadButton();
        this.updateLanguagesAndSpeakers();
    }
    
    updateTTSDownloadButton() {
        const engineSelect = document.getElementById('ttsEngineSelect');
        const modelSelect = document.getElementById('ttsModelSelect');
        const downloadBtn = document.getElementById('downloadTTSModelBtn');
        
        const selectedEngine = engineSelect.value;
        const selectedModelId = modelSelect.value;
        
        if (!selectedEngine || !selectedModelId) {
            downloadBtn.style.display = 'none';
            return;
        }
        
        const selectedModel = this.availableModels.find(
            m => m.engine === selectedEngine && m.model_id === selectedModelId
        );
        
        if (!selectedModel) {
            downloadBtn.style.display = 'none';
            return;
        }
        
        if (!selectedModel.cached && !selectedModel.downloading) {
            downloadBtn.style.display = 'inline-flex';
            downloadBtn.disabled = false;
            downloadBtn.innerHTML = '<i class="fas fa-download"></i> Download Model';
        } else if (selectedModel.downloading) {
            downloadBtn.style.display = 'inline-flex';
            downloadBtn.disabled = true;
            downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Downloading...';
        } else {
            downloadBtn.style.display = 'none';
        }
    }
    
    updateLanguagesAndSpeakers() {
        const engineSelect = document.getElementById('ttsEngineSelect');
        const modelSelect = document.getElementById('ttsModelSelect');
        const languageSelect = document.getElementById('ttsLanguage');
        const speakerSelect = document.getElementById('ttsSpeaker');
        
        const selectedEngine = engineSelect.value;
        const selectedModelId = modelSelect.value;
        
        // Reset
        languageSelect.innerHTML = '<option value="">Auto-detect</option>';
        speakerSelect.innerHTML = '<option value="">Default</option>';
        
        if (!selectedEngine || !selectedModelId) return;
        
        const selectedModel = this.availableModels.find(
            m => m.engine === selectedEngine && m.model_id === selectedModelId
        );
        
        if (!selectedModel) return;
        
        // Populate languages
        if (selectedModel.languages && selectedModel.languages.length > 0) {
            selectedModel.languages.forEach(lang => {
                const option = document.createElement('option');
                option.value = lang;
                option.textContent = lang;
                languageSelect.appendChild(option);
            });
        }
        
        // Populate speakers based on selected language
        languageSelect.addEventListener('change', () => {
            this.updateSpeakersForLanguage(selectedModel, languageSelect.value);
        });
    }
    
    updateSpeakersForLanguage(model, language) {
        const speakerSelect = document.getElementById('ttsSpeaker');
        speakerSelect.innerHTML = '<option value="">Default</option>';
        
        if (!language || !model.speakers || !model.speakers[language]) return;
        
        model.speakers[language].forEach(speaker => {
            const option = document.createElement('option');
            option.value = speaker;
            option.textContent = speaker;
            speakerSelect.appendChild(option);
        });
    }
    
    updateCharCount() {
        const charCount = document.getElementById('ttsCharCount');
        charCount.textContent = `${this.selectedText.length} characters`;
    }
    
    updateTTSUI() {
        const synthesizeBtn = document.getElementById('synthesizeBtn');
        const engineSelect = document.getElementById('ttsEngineSelect');
        const modelSelect = document.getElementById('ttsModelSelect');
        
        const hasText = this.selectedText.trim().length > 0;
        const hasModel = engineSelect.value && modelSelect.value;
        
        const selectedModel = this.availableModels.find(
            m => m.engine === engineSelect.value && m.model_id === modelSelect.value
        );
        
        const modelCached = selectedModel && selectedModel.cached;
        
        synthesizeBtn.disabled = !(hasText && hasModel && modelCached);
    }
    
    async handleDownloadTTSModelClick() {
        const engineSelect = document.getElementById('ttsEngineSelect');
        const modelSelect = document.getElementById('ttsModelSelect');
        
        const selectedEngine = engineSelect.value;
        const selectedModelId = modelSelect.value;
        
        if (!selectedEngine || !selectedModelId) {
            this.voxscribe.showToast('Please select an engine and model first', 'warning');
            return;
        }
        
        const selectedModel = this.availableModels.find(
            m => m.engine === selectedEngine && m.model_id === selectedModelId
        );
        
        if (!selectedModel) {
            this.voxscribe.showToast('Selected model not found', 'error');
            return;
        }
        
        if (selectedModel.cached) {
            this.voxscribe.showToast('Model is already cached', 'info');
            return;
        }
        
        try {
            const formData = new FormData();
            formData.append('engine', selectedEngine);
            formData.append('model_id', selectedModelId);
            
            const response = await fetch('/api/tts/download-model', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (!response.ok) {
                throw new Error(result.detail || 'Download request failed');
            }
            
            if (result.cached) {
                this.voxscribe.showToast('Model is already cached!', 'info');
                await this.loadTTSModels();
            } else {
                this.voxscribe.showToast(`Download started for ${selectedEngine}/${selectedModelId}`, 'info');
                await this.loadTTSModels();
            }
            
        } catch (error) {
            console.error('Download failed:', error);
            this.voxscribe.showToast(`Download failed: ${error.message}`, 'error');
        }
    }
    
    async handleSynthesizeClick() {
        const engineSelect = document.getElementById('ttsEngineSelect');
        const modelSelect = document.getElementById('ttsModelSelect');
        const textInput = document.getElementById('ttsTextInput');
        const description = document.getElementById('ttsDescription').value;
        const language = document.getElementById('ttsLanguage').value;
        const speaker = document.getElementById('ttsSpeaker').value;
        
        if (!textInput.value.trim()) {
            this.voxscribe.showToast('Please enter some text', 'warning');
            return;
        }
        
        const engine = engineSelect.value;
        const modelId = modelSelect.value;
        
        if (!engine || !modelId) {
            this.voxscribe.showToast('Please select an engine and model', 'warning');
            return;
        }
        
        this.showTTSProgress('Synthesizing speech...');
        
        const formData = new FormData();
        formData.append('text', textInput.value);
        formData.append('engine', engine);
        formData.append('model_id', modelId);
        if (description) formData.append('description', description);
        if (language) formData.append('language', language);
        if (speaker) formData.append('speaker', speaker);
        
        try {
            const response = await fetch('/api/tts/synthesize', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Synthesis failed');
            }
            
            // Get audio blob
            const audioBlob = await response.blob();
            const audioUrl = URL.createObjectURL(audioBlob);
            
            // Get metadata from headers
            const duration = response.headers.get('X-Audio-Duration');
            const processingTime = response.headers.get('X-Processing-Time');
            const sampleRate = response.headers.get('X-Sample-Rate');
            
            this.hideTTSProgress();
            this.displayTTSResults(audioUrl, {
                engine,
                modelId,
                text: textInput.value,
                duration,
                processingTime,
                sampleRate
            });
            
            this.voxscribe.showToast('Speech synthesized successfully!', 'success');
            
        } catch (error) {
            this.hideTTSProgress();
            this.voxscribe.showToast(`Synthesis failed: ${error.message}`, 'error');
            console.error('Synthesis error:', error);
        }
    }
    
    showTTSProgress(message) {
        const progressSection = document.getElementById('ttsProgressSection');
        const progressText = document.getElementById('ttsProgressText');
        
        progressSection.style.display = 'block';
        progressText.textContent = message;
    }
    
    hideTTSProgress() {
        const progressSection = document.getElementById('ttsProgressSection');
        progressSection.style.display = 'none';
    }
    
    displayTTSResults(audioUrl, metadata) {
        const resultsSection = document.getElementById('ttsResultsSection');
        const resultsContent = document.getElementById('ttsResultsContent');
        
        resultsContent.innerHTML = `
            <div class="tts-result-card">
                <div class="result-header">
                    <h4>${metadata.engine} - ${metadata.modelId}</h4>
                </div>
                <div class="result-audio">
                    <audio controls src="${audioUrl}"></audio>
                </div>
                <div class="result-metadata">
                    <div class="metadata-item">
                        <span class="metadata-label">Duration:</span>
                        <span class="metadata-value">${parseFloat(metadata.duration).toFixed(2)}s</span>
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">Processing Time:</span>
                        <span class="metadata-value">${parseFloat(metadata.processingTime).toFixed(2)}s</span>
                    </div>
                    <div class="metadata-item">
                        <span class="metadata-label">Sample Rate:</span>
                        <span class="metadata-value">${metadata.sampleRate} Hz</span>
                    </div>
                </div>
                <div class="result-text">
                    <strong>Text:</strong>
                    <p>${metadata.text}</p>
                </div>
                <div class="result-actions">
                    <a href="${audioUrl}" download="synthesized_speech.wav" class="btn btn-secondary">
                        <i class="fas fa-download"></i> Download
                    </a>
                </div>
            </div>
        `;
        
        resultsSection.style.display = 'block';
    }
}

// Initialize TTS manager when VoxScribe is ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait for VoxScribe to be initialized
    const checkVoxScribe = setInterval(() => {
        if (window.voxscribe) {
            window.ttsManager = new TTSManager(window.voxscribe);
            clearInterval(checkVoxScribe);
        }
    }, 100);
});
