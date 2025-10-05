// SpeechHub Frontend JavaScript

class SpeechHub {
    constructor() {
        this.currentMode = 'single';
        this.selectedFile = null;
        this.availableModels = [];
        this.selectedModels = [];
        this.websocket = null;
        this.pendingDownload = null;

        this.init();
    }

    async init() {
        this.setupEventListeners();
        this.setupTheme();
        this.setupWebSocket();
        await this.loadStatus();
        await this.loadModels();
        await this.loadRecentLogs();
        await this.loadTranscriptionHistory();
        this.updateUI();
    }

    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        console.log('Attempting to connect to WebSocket:', wsUrl);
        this.websocket = new WebSocket(wsUrl);

        this.websocket.onopen = () => {
            console.log('WebSocket connected successfully');
            this.showToast('Connected to server', 'success');
        };

        this.websocket.onmessage = (event) => {
            console.log('Raw WebSocket message received:', event.data);
            try {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            } catch (error) {
                console.error('Failed to parse WebSocket message:', error, event.data);
            }
        };

        this.websocket.onclose = (event) => {
            console.log('WebSocket disconnected:', event.code, event.reason);
            this.showToast('Disconnected from server', 'warning');
            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.setupWebSocket(), 3000);
        };

        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showToast('Connection error', 'error');
        };
    }

    handleWebSocketMessage(data) {
        console.log('WebSocket message received:', data);
        switch (data.type) {
            case 'download_progress':
                console.log('Handling download progress:', data);
                this.updateDownloadProgress(data);
                break;
            case 'download_complete':
                console.log('Handling download complete:', data);
                this.handleDownloadComplete(data);
                break;
            case 'dependency_progress':
                console.log('Handling dependency progress:', data);
                this.updateDependencyProgress(data);
                break;
            case 'dependency_complete':
                console.log('Handling dependency complete:', data);
                this.handleDependencyComplete(data);
                break;
            case 'pong':
                console.log('WebSocket pong received');
                break;
            default:
                console.log('Unknown WebSocket message:', data);
        }
    }

    updateDownloadProgress(data) {
        const { engine, model_id, progress, status, message } = data;

        // Update progress modal
        document.getElementById('progressModalTitle').textContent = 'Downloading Model';
        document.getElementById('progressModelName').textContent = `${engine}/${model_id}`;
        document.getElementById('progressStatus').textContent = status;
        document.getElementById('downloadProgressFill').style.width = `${progress}%`;
        document.getElementById('downloadProgressText').textContent = message || `${progress}%`;

        // Show progress modal if not already visible
        if (document.getElementById('downloadProgressModal').style.display !== 'flex') {
            this.showDownloadProgressModal();
        }

        // Update model list to show downloading status
        this.updateModelDownloadStatus(engine, model_id, true, false);
    }

    updateDependencyProgress(data) {
        const { dependency, progress, status, message, engine, model_id } = data;

        // Update progress modal with dependency info
        const modelName = engine && model_id ? `${engine}/${model_id}` : dependency;
        document.getElementById('progressModalTitle').textContent = 'Installing Dependencies';
        document.getElementById('progressModelName').textContent = modelName;
        document.getElementById('progressStatus').textContent = `Installing ${dependency} - ${status}`;
        document.getElementById('downloadProgressFill').style.width = `${progress}%`;
        document.getElementById('downloadProgressText').textContent = message || `${progress}%`;

        // Show progress modal if not already visible
        if (document.getElementById('downloadProgressModal').style.display !== 'flex') {
            this.showDownloadProgressModal();
        }

        // Show toast for major dependency progress updates
        if (status === 'installing') {
            this.showToast(`Installing ${dependency} dependencies...`, 'info');
        }
    }

    async handleDependencyComplete(data) {
        const { dependency, success, error, engine, model_id } = data;

        if (success) {
            this.showToast(`${dependency} dependencies installed successfully!`, 'success');

            // Refresh status to update dependency availability
            await this.loadStatus();

            // If this was part of a model download, the download will continue automatically
            if (engine && model_id) {
                this.showToast(`Continuing with ${engine}/${model_id} download...`, 'info');
            } else {
                // Hide progress modal if this was a standalone dependency install
                this.hideDownloadProgressModal();
            }
        } else {
            this.showToast(`Dependency installation failed: ${error}`, 'error');
            this.hideDownloadProgressModal();
        }
    }

    async handleDownloadComplete(data) {
        const { engine, model_id, success, error } = data;

        if (success) {
            this.showToast(`Model ${engine}/${model_id} downloaded successfully!`, 'success');
            this.hideDownloadProgressModal();

            // Refresh models to update cache status
            await this.loadModels();

            // If this was a pending download for transcription, proceed
            if (this.pendingDownload &&
                this.pendingDownload.engine === engine &&
                this.pendingDownload.model_id === model_id) {

                setTimeout(() => {
                    this.startTranscription();
                }, 1000);
                this.pendingDownload = null;
            }
        } else {
            this.showToast(`Download failed: ${error}`, 'error');
            this.hideDownloadProgressModal();
        }

        // Update model status
        this.updateModelDownloadStatus(engine, model_id, false, success);
    }

    updateModelDownloadStatus(engine, model_id, downloading, cached) {
        // Update the model in our local array
        const model = this.availableModels.find(m => m.engine === engine && m.model_id === model_id);
        if (model) {
            model.downloading = downloading;
            if (cached !== undefined) {
                model.cached = cached;
            }
        }

        // Update UI
        this.updateModelSelects();
        this.updateCompareModels();
        this.updateDownloadButton();
    }

    updateDownloadButton() {
        const engineSelect = document.getElementById('engineSelect');
        const modelSelect = document.getElementById('modelSelect');
        const downloadBtn = document.getElementById('downloadModelBtn');

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

        // Show download button if model is not cached and not downloading
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

    handleDownloadModelClick() {
        const engineSelect = document.getElementById('engineSelect');
        const modelSelect = document.getElementById('modelSelect');

        const selectedEngine = engineSelect.value;
        const selectedModelId = modelSelect.value;

        if (!selectedEngine || !selectedModelId) {
            this.showToast('Please select an engine and model first', 'warning');
            return;
        }

        const selectedModel = this.availableModels.find(
            m => m.engine === selectedEngine && m.model_id === selectedModelId
        );

        if (!selectedModel) {
            this.showToast('Selected model not found', 'error');
            return;
        }

        if (selectedModel.cached) {
            this.showToast('Model is already cached', 'info');
            return;
        }

        if (selectedModel.downloading) {
            this.showToast('Model is already downloading', 'info');
            return;
        }

        // Show download confirmation modal
        this.showDownloadModal(selectedModel);
    }

    setupEventListeners() {
        // Theme toggle
        document.getElementById('themeToggle').addEventListener('click', () => {
            this.toggleTheme();
        });

        // Mode selection
        document.querySelectorAll('input[name="mode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.currentMode = e.target.value;
                this.updateUI();
            });
        });

        // File upload
        const fileUploadArea = document.getElementById('fileUploadArea');
        const audioFileInput = document.getElementById('audioFile');

        fileUploadArea.addEventListener('click', () => {
            audioFileInput.click();
        });

        fileUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            fileUploadArea.classList.add('dragover');
        });

        fileUploadArea.addEventListener('dragleave', () => {
            fileUploadArea.classList.remove('dragover');
        });

        fileUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            fileUploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });

        audioFileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });

        // Model selection
        document.getElementById('engineSelect').addEventListener('change', () => {
            this.updateModelSelect();
        });

        document.getElementById('modelSelect').addEventListener('change', () => {
            this.updateDownloadButton();
            this.updateUI();
        });

        // Download model button
        document.getElementById('downloadModelBtn').addEventListener('click', () => {
            this.handleDownloadModelClick();
        });

        // Action buttons
        document.getElementById('transcribeBtn').addEventListener('click', () => {
            this.handleTranscribeClick();
        });

        document.getElementById('compareBtn').addEventListener('click', () => {
            this.handleCompareClick();
        });

        // Dependency installation
        document.getElementById('installVoxtral').addEventListener('click', () => {
            this.installDependency('voxtral');
        });

        document.getElementById('installNemo').addEventListener('click', () => {
            this.installDependency('nvidia');
        });

        // Transcription history
        document.getElementById('viewAllHistoryBtn').addEventListener('click', () => {
            this.showHistoryModal();
        });

        document.getElementById('refreshHistoryBtn').addEventListener('click', () => {
            this.loadTranscriptionHistory();
        });

        // Modal event listeners
        this.setupModalEventListeners();
    }

    setupModalEventListeners() {
        // Download confirmation modal
        document.getElementById('downloadModalClose').addEventListener('click', () => {
            this.hideDownloadModal();
        });

        document.getElementById('cancelDownload').addEventListener('click', () => {
            this.hideDownloadModal();
        });

        document.getElementById('confirmDownload').addEventListener('click', () => {
            this.confirmModelDownload();
        });

        // Download progress modal
        document.getElementById('cancelDownloadProgress').addEventListener('click', () => {
            // TODO: Implement download cancellation
            this.hideDownloadProgressModal();
        });

        // Delete confirmation modal
        document.getElementById('deleteModalClose').addEventListener('click', () => {
            this.hideDeleteConfirmModal();
        });

        document.getElementById('cancelDelete').addEventListener('click', () => {
            this.hideDeleteConfirmModal();
        });

        document.getElementById('confirmDelete').addEventListener('click', () => {
            this.confirmDeleteModel();
        });

        // Close modals when clicking outside
        document.getElementById('downloadModal').addEventListener('click', (e) => {
            if (e.target.id === 'downloadModal') {
                this.hideDownloadModal();
            }
        });

        document.getElementById('downloadProgressModal').addEventListener('click', (e) => {
            if (e.target.id === 'downloadProgressModal') {
                // Don't allow closing progress modal by clicking outside
            }
        });

        document.getElementById('deleteConfirmModal').addEventListener('click', (e) => {
            if (e.target.id === 'deleteConfirmModal') {
                this.hideDeleteConfirmModal();
            }
        });

        // History modal
        document.getElementById('historyModalClose').addEventListener('click', () => {
            this.hideHistoryModal();
        });

        document.getElementById('historyModal').addEventListener('click', (e) => {
            if (e.target.id === 'historyModal') {
                this.hideHistoryModal();
            }
        });

        // Transcription detail modal
        document.getElementById('transcriptionDetailClose').addEventListener('click', () => {
            this.hideTranscriptionDetailModal();
        });

        document.getElementById('transcriptionDetailModal').addEventListener('click', (e) => {
            if (e.target.id === 'transcriptionDetailModal') {
                this.hideTranscriptionDetailModal();
            }
        });

        document.getElementById('downloadTranscriptionBtn').addEventListener('click', () => {
            this.downloadTranscription();
        });

        document.getElementById('deleteTranscriptionBtn').addEventListener('click', () => {
            this.deleteTranscriptionFromHistory();
        });
    }

    async handleTranscribeClick() {
        if (!this.selectedFile) {
            this.showToast('Please select an audio file', 'warning');
            return;
        }

        const engine = document.getElementById('engineSelect').value;
        const modelId = document.getElementById('modelSelect').value;

        if (!engine || !modelId) {
            this.showToast('Please select an engine and model', 'warning');
            return;
        }

        // Check if model is cached
        const model = this.availableModels.find(m => m.engine === engine && m.model_id === modelId);
        if (!model) {
            this.showToast('Selected model not found', 'error');
            return;
        }

        if (!model.cached && !model.downloading) {
            // Show download confirmation modal
            this.showDownloadModal(model);
            return;
        }

        if (model.downloading) {
            this.showToast('Model is currently downloading. Please wait.', 'info');
            return;
        }

        // Model is cached, proceed with transcription
        this.startTranscription();
    }

    async startTranscription() {
        const engine = document.getElementById('engineSelect').value;
        const modelId = document.getElementById('modelSelect').value;

        this.showProgress('Transcribing audio...');

        const formData = new FormData();
        formData.append('file', this.selectedFile);
        formData.append('engine', engine);
        formData.append('model_id', modelId);

        try {
            const response = await fetch('/api/transcribe', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || 'Transcription failed');
            }

            this.hideProgress();
            this.displayResults(result, 'single');
            this.showToast('Transcription completed!', 'success');

            // Refresh transcription history
            await this.loadTranscriptionHistory();

        } catch (error) {
            this.hideProgress();
            this.showToast(`Transcription failed: ${error.message}`, 'error');
            console.error('Transcription error:', error);
        }
    }

    showDownloadModal(model) {
        document.getElementById('downloadModelName').textContent = model.display_name;
        document.getElementById('downloadModelSize').textContent = model.size;
        document.getElementById('downloadModelEngine').textContent = model.engine;

        this.pendingDownload = {
            engine: model.engine,
            model_id: model.model_id
        };

        document.getElementById('downloadModal').style.display = 'flex';
    }

    hideDownloadModal() {
        document.getElementById('downloadModal').style.display = 'none';
        this.pendingDownload = null;
    }

    showDownloadProgressModal() {
        document.getElementById('downloadProgressModal').style.display = 'flex';
    }

    hideDownloadProgressModal() {
        document.getElementById('downloadProgressModal').style.display = 'none';
    }

    async confirmModelDownload() {
        console.log('confirmModelDownload called');
        if (!this.pendingDownload) {
            console.log('No pending download');
            return;
        }

        // Store the download info before hiding modal (which sets pendingDownload to null)
        const { engine, model_id } = this.pendingDownload;
        this.hideDownloadModal();
        console.log(`Starting download for ${engine}/${model_id}`);

        try {
            const formData = new FormData();
            formData.append('engine', engine);
            formData.append('model_id', model_id);

            console.log('Sending download request to /api/download-model');
            const response = await fetch('/api/download-model', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            console.log('Download API response:', result);

            if (!response.ok) {
                throw new Error(result.detail || 'Download request failed');
            }

            if (result.cached) {
                // Model was already cached
                console.log('Model was already cached');
                this.showToast('Model is already cached!', 'info');
                await this.loadModels();
                setTimeout(() => this.startTranscription(), 500);
            } else {
                // Download started - preserve pending download for when it completes
                this.pendingDownload = { engine, model_id };
                console.log('Download started successfully');
                this.showToast(`Download started for ${engine}/${model_id}. You can continue using the app and check progress anytime.`, 'info');
                await this.loadModels(); // Refresh to show downloading status
            }

        } catch (error) {
            console.error('Download failed:', error);
            this.showToast(`Download failed: ${error.message}`, 'error');
        }
    }

    async handleCompareClick() {
        if (!this.selectedFile) {
            this.showToast('Please select an audio file', 'warning');
            return;
        }

        const selectedModels = this.getSelectedCompareModels();
        if (selectedModels.length < 2) {
            this.showToast('Please select at least 2 models for comparison', 'warning');
            return;
        }

        // Check if all models are cached
        const uncachedModels = selectedModels.filter(model => !model.cached);
        if (uncachedModels.length > 0) {
            const modelNames = uncachedModels.map(m => `${m.engine}/${m.model_id}`).join(', ');
            this.showToast(`The following models need to be downloaded first: ${modelNames}`, 'warning');
            return;
        }

        this.showProgress('Comparing models...');

        const formData = new FormData();
        formData.append('file', this.selectedFile);
        formData.append('engines', JSON.stringify(selectedModels.map(m => ({
            engine: m.engine,
            model_id: m.model_id
        }))));

        try {
            const response = await fetch('/api/compare', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || 'Comparison failed');
            }

            this.hideProgress();
            this.displayResults(result, 'compare');
            this.showToast('Comparison completed!', 'success');

            // Refresh transcription history
            await this.loadTranscriptionHistory();

        } catch (error) {
            this.hideProgress();
            this.showToast(`Comparison failed: ${error.message}`, 'error');
            console.error('Comparison error:', error);
        }
    }

    getSelectedCompareModels() {
        const checkboxes = document.querySelectorAll('#compareModels input[type="checkbox"]:checked');
        return Array.from(checkboxes).map(checkbox => {
            const [engine, modelId] = checkbox.value.split(':');
            return this.availableModels.find(m => m.engine === engine && m.model_id === modelId);
        }).filter(Boolean);
    }

    handleFileSelect(file) {
        const supportedFormats = ['wav', 'mp3', 'flac', 'm4a', 'ogg'];
        const fileExtension = file.name.split('.').pop().toLowerCase();

        if (!supportedFormats.includes(fileExtension)) {
            this.showToast(`Unsupported file format. Supported: ${supportedFormats.join(', ')}`, 'error');
            return;
        }

        this.selectedFile = file;
        this.updateFileInfo();
        this.updateUI();
    }

    updateFileInfo() {
        if (!this.selectedFile) {
            document.getElementById('fileInfo').style.display = 'none';
            return;
        }

        document.getElementById('fileName').textContent = this.selectedFile.name;
        document.getElementById('fileSize').textContent = this.formatFileSize(this.selectedFile.size);

        const audioPreview = document.getElementById('audioPreview');
        audioPreview.src = URL.createObjectURL(this.selectedFile);

        document.getElementById('fileInfo').style.display = 'block';
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async loadStatus() {
        try {
            const response = await fetch('/api/status');
            const status = await response.json();

            document.getElementById('deviceStatus').textContent = status.device;

            // Update dependency status
            this.updateDependencyStatus('voxtral', status.dependencies.voxtral_supported);
            this.updateDependencyStatus('nemo', status.dependencies.nemo_supported);

            const transformersVersion = status.dependencies.transformers_version;
            const dependencyText = transformersVersion ?
                `Transformers ${transformersVersion}` : 'Dependencies loading...';
            document.getElementById('dependencyStatus').textContent = dependencyText;

        } catch (error) {
            console.error('Failed to load status:', error);
            this.showToast('Failed to load system status', 'error');
        }
    }

    updateDependencyStatus(dependency, supported) {
        const statusElement = document.getElementById(`${dependency}Status`);
        const installButton = document.getElementById(`install${dependency.charAt(0).toUpperCase() + dependency.slice(1)}`);

        if (supported) {
            statusElement.textContent = 'Available';
            statusElement.className = 'dependency-status status-success';
            installButton.style.display = 'none';
        } else {
            statusElement.textContent = 'Not Available';
            statusElement.className = 'dependency-status status-error';
            installButton.style.display = 'inline-block';
        }
    }

    async loadModels() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            this.availableModels = data.models;
            this.updateModelSelects();
            this.updateCompareModels();
            this.updateCacheInfo();
        } catch (error) {
            console.error('Failed to load models:', error);
            this.showToast('Failed to load models', 'error');
        }
    }

    updateModelSelects() {
        const engineSelect = document.getElementById('engineSelect');
        const modelSelect = document.getElementById('modelSelect');

        // Preserve current selections
        const currentEngine = engineSelect.value;
        const currentModel = modelSelect.value;

        // Get unique engines
        const engines = [...new Set(this.availableModels.map(m => m.engine))];

        engineSelect.innerHTML = '<option value="">Select Engine...</option>';
        engines.forEach(engine => {
            const option = document.createElement('option');
            option.value = engine;
            option.textContent = engine.charAt(0).toUpperCase() + engine.slice(1);
            engineSelect.appendChild(option);
        });

        // Restore engine selection if it still exists
        if (currentEngine && engines.includes(currentEngine)) {
            engineSelect.value = currentEngine;
        }

        this.updateModelSelect(currentModel);
    }

    updateModelSelect(preserveModel = null) {
        const engineSelect = document.getElementById('engineSelect');
        const modelSelect = document.getElementById('modelSelect');
        const downloadBtn = document.getElementById('downloadModelBtn');
        const selectedEngine = engineSelect.value;

        // Use preserved model or current selection
        const currentModel = preserveModel || modelSelect.value;

        modelSelect.innerHTML = '<option value="">Select Model...</option>';

        if (selectedEngine) {
            const engineModels = this.availableModels.filter(m => m.engine === selectedEngine);
            engineModels.forEach(model => {
                const option = document.createElement('option');
                option.value = model.model_id;

                let displayText = model.display_name;
                if (!model.cached) {
                    displayText += model.downloading ? ' (Downloading...)' : ' (Not Downloaded)';
                }

                option.textContent = displayText;
                option.disabled = model.downloading;
                modelSelect.appendChild(option);
            });

            // Restore model selection if it still exists and is available
            if (currentModel && engineModels.some(m => m.model_id === currentModel)) {
                modelSelect.value = currentModel;
            }
        }

        // Update download button visibility
        this.updateDownloadButton();
        this.updateUI();
    }

    updateCompareModels() {
        const compareModelsContainer = document.getElementById('compareModels');
        compareModelsContainer.innerHTML = '';

        // Sort models: cached first, then downloading, then not downloaded
        const sortedModels = [...this.availableModels].sort((a, b) => {
            // Priority: cached (0) > downloading (1) > not cached (2)
            const getPriority = (model) => {
                if (model.cached && !model.downloading) return 0;
                if (model.downloading) return 1;
                return 2;
            };

            const priorityA = getPriority(a);
            const priorityB = getPriority(b);

            if (priorityA !== priorityB) {
                return priorityA - priorityB;
            }

            // Within same priority, sort by display name
            return a.display_name.localeCompare(b.display_name);
        });

        sortedModels.forEach(model => {
            const modelCard = document.createElement('div');
            modelCard.className = 'model-card-compare';

            let statusClass = 'status-cached';
            let statusText = 'Cached';
            let disabled = false;

            if (model.downloading) {
                statusClass = 'status-downloading';
                statusText = 'Downloading...';
                disabled = true;
            } else if (!model.cached) {
                statusClass = 'status-not-cached';
                statusText = 'Not Downloaded';
                disabled = true;
            }

            modelCard.innerHTML = `
                <label class="model-checkbox-label ${disabled ? 'disabled' : ''}">
                    <input type="checkbox" value="${model.engine}:${model.model_id}" ${disabled ? 'disabled' : ''}>
                    <div class="model-info">
                        <div class="model-name">${model.display_name}</div>
                        <div class="model-details">
                            <span class="model-engine">${model.engine}</span>
                            <span class="model-size">${model.size}</span>
                        </div>
                        <div class="model-status ${statusClass}">${statusText}</div>
                    </div>
                </label>
            `;

            compareModelsContainer.appendChild(modelCard);
        });

        this.updateUI();
    }

    updateCacheInfo() {
        const cachedModels = this.availableModels.filter(m => m.cached);
        const totalModels = this.availableModels.length;

        document.getElementById('cacheStatus').textContent =
            `${cachedModels.length} of ${totalModels} models cached`;

        const cachedModelsContainer = document.getElementById('cachedModels');
        cachedModelsContainer.innerHTML = '';

        if (cachedModels.length === 0) {
            cachedModelsContainer.innerHTML = '<p class="info-text">No models cached yet</p>';
            return;
        }

        cachedModels.forEach(model => {
            const modelItem = document.createElement('div');
            modelItem.className = 'cached-model-item';
            modelItem.innerHTML = `
                <div class="cached-model-info">
                    <div class="cached-model-name">${model.display_name}</div>
                    <div class="cached-model-details">${model.engine} • ${model.size}</div>
                </div>
                <button class="delete-model-btn" onclick="speechHub.deleteModel('${model.engine}', '${model.model_id}')" title="Delete cached model">
                    <i class="fas fa-trash"></i>
                </button>
            `;
            cachedModelsContainer.appendChild(modelItem);
        });
    }

    async loadRecentLogs() {
        try {
            const response = await fetch('/api/logs?log_type=transcriptions&limit=5');
            const data = await response.json();

            const recentLogsContainer = document.getElementById('recentLogs');

            if (data.logs.length === 0) {
                recentLogsContainer.innerHTML = '<p class="info-text">No recent activity</p>';
                return;
            }

            recentLogsContainer.innerHTML = '';
            data.logs.reverse().forEach(log => {
                const logItem = document.createElement('div');
                logItem.className = 'log-item';

                const timestamp = new Date(log.timestamp).toLocaleTimeString();
                const event = log.event.replace('_', ' ');

                logItem.innerHTML = `
                    <div class="log-time">${timestamp}</div>
                    <div class="log-event">${event}</div>
                `;

                recentLogsContainer.appendChild(logItem);
            });

        } catch (error) {
            console.error('Failed to load recent logs:', error);
        }
    }

    async installDependency(dependency) {
        const button = document.getElementById(`install${dependency.charAt(0).toUpperCase() + dependency.slice(1)}`);
        const originalText = button.innerHTML;

        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Installing...';
        button.disabled = true;

        try {
            // Show progress modal immediately
            document.getElementById('progressModalTitle').textContent = 'Installing Dependencies';
            document.getElementById('progressModelName').textContent = dependency;
            document.getElementById('progressStatus').textContent = 'Preparing installation...';
            document.getElementById('downloadProgressFill').style.width = '0%';
            document.getElementById('downloadProgressText').textContent = 'Starting...';
            this.showDownloadProgressModal();

            const response = await fetch('/api/install-dependency', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ dependency })
            });

            const result = await response.json();

            if (response.ok) {
                // Success message will be handled by WebSocket
                // Just reload status to update dependency info
                await this.loadStatus();
            } else {
                this.hideDownloadProgressModal();
                throw new Error(result.detail || 'Installation failed');
            }

        } catch (error) {
            this.hideDownloadProgressModal();
            this.showToast(`Installation failed: ${error.message}`, 'error');
        } finally {
            button.innerHTML = originalText;
            button.disabled = false;
        }
    }

    async loadTranscriptionHistory() {
        try {
            const response = await fetch('/api/transcriptions?limit=5');
            const data = await response.json();

            const historyContainer = document.getElementById('transcriptionHistoryList');

            if (data.transcriptions.length === 0) {
                historyContainer.innerHTML = '<p class="info-text">No transcriptions yet</p>';
                return;
            }

            historyContainer.innerHTML = '';
            data.transcriptions.forEach(transcription => {
                const historyItem = document.createElement('div');
                historyItem.className = 'history-item';
                historyItem.onclick = () => this.showTranscriptionDetail(transcription.id);

                const statusClass = transcription.success ? 'success' : 'error';
                const statusText = transcription.success ? 'Success' : 'Failed';
                const date = new Date(transcription.timestamp).toLocaleDateString();
                const duration = transcription.transcription_duration_seconds
                    ? `${transcription.transcription_duration_seconds.toFixed(1)}s`
                    : 'N/A';

                historyItem.innerHTML = `
                    <div class="history-item-header">
                        <div class="history-filename" title="${transcription.audio_filename}">
                            ${transcription.audio_filename}
                        </div>
                        <div class="history-status ${statusClass}">${statusText}</div>
                    </div>
                    <div class="history-item-details">
                        <div class="history-model">${transcription.model_display_name}</div>
                        <div class="history-duration">${duration}</div>
                    </div>
                `;

                historyContainer.appendChild(historyItem);
            });

        } catch (error) {
            console.error('Failed to load transcription history:', error);
            document.getElementById('transcriptionHistoryList').innerHTML =
                '<p class="info-text">Failed to load history</p>';
        }
    }

    showHistoryModal() {
        this.loadFullTranscriptionHistory();
        document.getElementById('historyModal').style.display = 'flex';
    }

    hideHistoryModal() {
        document.getElementById('historyModal').style.display = 'none';
    }

    async loadFullTranscriptionHistory() {
        try {
            const response = await fetch('/api/transcriptions?limit=50');
            const data = await response.json();

            const historyContent = document.getElementById('historyContent');

            if (data.transcriptions.length === 0) {
                historyContent.innerHTML = '<p class="info-text">No transcriptions found</p>';
                return;
            }

            let tableHTML = `
                <table class="history-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Audio File</th>
                            <th>Model</th>
                            <th>Duration</th>
                            <th>Status</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            data.transcriptions.forEach(transcription => {
                const date = new Date(transcription.timestamp).toLocaleString();
                const statusClass = transcription.success ? 'success' : 'error';
                const statusText = transcription.success ? 'Success' : 'Failed';
                const duration = transcription.transcription_duration_seconds
                    ? `${transcription.transcription_duration_seconds.toFixed(1)}s`
                    : 'N/A';

                tableHTML += `
                    <tr>
                        <td>${date}</td>
                        <td title="${transcription.audio_filename}">${transcription.audio_filename}</td>
                        <td>${transcription.model_display_name}</td>
                        <td>${duration}</td>
                        <td><span class="history-status ${statusClass}">${statusText}</span></td>
                        <td>
                            <div class="history-table-actions">
                                <button class="btn-view" onclick="speechHub.showTranscriptionDetail('${transcription.id}')">
                                    <i class="fas fa-eye"></i> View
                                </button>
                                <button class="btn-delete-small" onclick="speechHub.deleteTranscriptionFromHistory('${transcription.id}')">
                                    <i class="fas fa-trash"></i>
                                </button>
                            </div>
                        </td>
                    </tr>
                `;
            });

            tableHTML += '</tbody></table>';
            historyContent.innerHTML = tableHTML;

        } catch (error) {
            console.error('Failed to load full transcription history:', error);
            document.getElementById('historyContent').innerHTML =
                '<p class="info-text">Failed to load history</p>';
        }
    }

    async showTranscriptionDetail(transcriptionId) {
        try {
            this.hideHistoryModal();
            this.showProgress('Loading transcription details...');

            const response = await fetch(`/api/transcriptions/${transcriptionId}`);
            const transcription = await response.json();

            if (!response.ok) {
                throw new Error(transcription.detail || 'Failed to load transcription');
            }

            this.hideProgress();
            this.currentTranscriptionId = transcriptionId;

            // Update modal title
            document.getElementById('transcriptionDetailTitle').textContent =
                `Transcription: ${transcription.audio_filename}`;

            // Build metadata section
            const date = new Date(transcription.timestamp).toLocaleString();
            const duration = transcription.transcription_duration_seconds
                ? `${transcription.transcription_duration_seconds.toFixed(2)} seconds`
                : 'N/A';
            const audioDuration = transcription.audio_duration_seconds
                ? `${transcription.audio_duration_seconds.toFixed(2)} seconds`
                : 'N/A';

            let metaHTML = `
                <div class="transcription-meta">
                    <div class="meta-item">
                        <div class="meta-label">Date</div>
                        <div class="meta-value">${date}</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">Audio File</div>
                        <div class="meta-value">${transcription.audio_filename}</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">Model</div>
                        <div class="meta-value">${transcription.model_display_name}</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">Audio Duration</div>
                        <div class="meta-value">${audioDuration}</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">Transcription Time</div>
                        <div class="meta-value">${duration}</div>
                    </div>
                    <div class="meta-item">
                        <div class="meta-label">Status</div>
                        <div class="meta-value">
                            <span class="history-status ${transcription.success ? 'success' : 'error'}">
                                ${transcription.success ? 'Success' : 'Failed'}
                            </span>
                        </div>
                    </div>
                </div>
            `;

            // Add results section
            if (transcription.success && transcription.csv_data) {
                metaHTML += `
                    <div class="transcription-results">
                        <h4>Transcription Results</h4>
                        ${this.generateCSVTable(transcription.csv_data)}
                    </div>
                `;
            } else if (!transcription.success && transcription.error) {
                metaHTML += `
                    <div class="transcription-results">
                        <h4>Error Details</h4>
                        <div class="result-error">
                            <i class="fas fa-exclamation-triangle"></i>
                            <p>${transcription.error}</p>
                        </div>
                    </div>
                `;
            }

            document.getElementById('transcriptionDetailContent').innerHTML = metaHTML;
            document.getElementById('transcriptionDetailModal').style.display = 'flex';

        } catch (error) {
            this.hideProgress();
            this.showToast(`Failed to load transcription details: ${error.message}`, 'error');
        }
    }

    hideTranscriptionDetailModal() {
        document.getElementById('transcriptionDetailModal').style.display = 'none';
        this.currentTranscriptionId = null;
    }

    async downloadTranscription() {
        if (!this.currentTranscriptionId) {
            this.showToast('No transcription selected', 'error');
            return;
        }

        try {
            const response = await fetch(`/api/transcriptions/${this.currentTranscriptionId}/download`);

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Download failed');
            }

            // Create download link
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = response.headers.get('Content-Disposition')?.split('filename=')[1] || 'transcription.csv';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            this.showToast('Transcription downloaded successfully', 'success');

        } catch (error) {
            this.showToast(`Download failed: ${error.message}`, 'error');
        }
    }

    async deleteTranscriptionFromHistory(transcriptionId = null) {
        const idToDelete = transcriptionId || this.currentTranscriptionId;

        if (!idToDelete) {
            this.showToast('No transcription selected', 'error');
            return;
        }

        if (!confirm('Are you sure you want to delete this transcription from history?')) {
            return;
        }

        try {
            const response = await fetch(`/api/transcriptions/${idToDelete}`, {
                method: 'DELETE'
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || 'Failed to delete transcription');
            }

            this.showToast('Transcription deleted from history', 'success');

            // Refresh history
            await this.loadTranscriptionHistory();

            // If we're in the detail modal, close it
            if (idToDelete === this.currentTranscriptionId) {
                this.hideTranscriptionDetailModal();
            }

            // If we're in the full history modal, refresh it
            if (document.getElementById('historyModal').style.display === 'flex') {
                await this.loadFullTranscriptionHistory();
            }

        } catch (error) {
            this.showToast(`Failed to delete transcription: ${error.message}`, 'error');
        }
    }

    showDeleteConfirmModal(engine, modelId) {
        // Find the model for display name
        const model = this.availableModels.find(m => m.engine === engine && m.model_id === modelId);
        const displayName = model ? model.display_name : `${engine}/${modelId}`;

        document.getElementById('deleteModelName').textContent = displayName;

        // Store the model info for deletion
        this.pendingDelete = { engine, modelId };

        document.getElementById('deleteConfirmModal').style.display = 'flex';
    }

    hideDeleteConfirmModal() {
        document.getElementById('deleteConfirmModal').style.display = 'none';
        this.pendingDelete = null;
    }

    async deleteModel(engine, modelId) {
        this.showDeleteConfirmModal(engine, modelId);
    }

    async confirmDeleteModel() {
        if (!this.pendingDelete) {
            return;
        }

        const { engine, modelId } = this.pendingDelete;
        this.hideDeleteConfirmModal();

        try {
            this.showProgress('Deleting model cache...');

            const response = await fetch(`/api/models/${encodeURIComponent(engine)}/${encodeURIComponent(modelId)}`, {
                method: 'DELETE'
            });

            const result = await response.json();

            if (!response.ok) {
                throw new Error(result.detail || 'Failed to delete model');
            }

            this.hideProgress();
            this.showToast(result.message, 'success');

            // Refresh models to update the UI
            await this.loadModels();

        } catch (error) {
            this.hideProgress();
            this.showToast(`Failed to delete model: ${error.message}`, 'error');
            console.error('Delete model error:', error);
        }
    }

    updateUI() {
        // Show/hide mode-specific elements
        const singleSelect = document.getElementById('singleModelSelect');
        const compareSelect = document.getElementById('compareModelSelect');
        const transcribeBtn = document.getElementById('transcribeBtn');
        const compareBtn = document.getElementById('compareBtn');

        if (this.currentMode === 'single') {
            singleSelect.style.display = 'block';
            compareSelect.style.display = 'none';
            transcribeBtn.style.display = 'inline-flex';
            compareBtn.style.display = 'none';

            // Enable transcribe button if file and model selected and model is cached
            const engine = document.getElementById('engineSelect').value;
            const modelId = document.getElementById('modelSelect').value;

            let canTranscribe = false;
            if (this.selectedFile && engine && modelId) {
                const selectedModel = this.availableModels.find(m => m.engine === engine && m.model_id === modelId);
                canTranscribe = selectedModel && selectedModel.cached && !selectedModel.downloading;
            }

            transcribeBtn.disabled = !canTranscribe;

            // Update transcribe button text based on model status
            const transcribeBtnText = document.getElementById('transcribeBtnText');
            if (this.selectedFile && engine && modelId) {
                const selectedModel = this.availableModels.find(m => m.engine === engine && m.model_id === modelId);
                if (selectedModel && !selectedModel.cached && !selectedModel.downloading) {
                    transcribeBtnText.textContent = 'Download Model First';
                } else if (selectedModel && selectedModel.downloading) {
                    transcribeBtnText.textContent = 'Downloading...';
                } else {
                    transcribeBtnText.textContent = 'Transcribe';
                }
            } else {
                transcribeBtnText.textContent = 'Transcribe';
            }

        } else {
            singleSelect.style.display = 'none';
            compareSelect.style.display = 'block';
            transcribeBtn.style.display = 'none';
            compareBtn.style.display = 'inline-flex';

            // Enable compare button if file and at least 2 cached models selected
            const selectedModels = this.getSelectedCompareModels();
            const cachedSelectedModels = selectedModels.filter(m => m.cached && !m.downloading);
            compareBtn.disabled = !(this.selectedFile && cachedSelectedModels.length >= 2);
        }
    }

    displayResults(result, mode) {
        const resultsSection = document.getElementById('resultsSection');
        const resultsContent = document.getElementById('resultsContent');

        resultsContent.innerHTML = '';

        if (mode === 'single') {
            this.displaySingleResult(result, resultsContent);
        } else {
            this.displayComparisonResults(result, resultsContent);
        }

        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    displaySingleResult(result, container) {
        if (!result.success) {
            container.innerHTML = `
                <div class="result-error">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Transcription failed: ${result.error}</p>
                </div>
            `;
            return;
        }

        const resultCard = document.createElement('div');
        resultCard.className = 'result-card';

        const processingTime = result.processing_time.toFixed(2);
        const duration = result.duration.toFixed(2);

        resultCard.innerHTML = `
            <div class="result-header">
                <h4><i class="fas fa-file-audio"></i> Transcription Result</h4>
                <div class="result-stats">
                    <span class="stat">Processing: ${processingTime}s</span>
                    <span class="stat">Duration: ${duration}s</span>
                </div>
            </div>
            <div class="result-content">
                ${this.generateCSVTable(result.csv_data)}
            </div>
        `;

        container.appendChild(resultCard);
    }

    displayComparisonResults(result, container) {
        const results = result.results;

        Object.entries(results).forEach(([modelKey, modelResult]) => {
            const resultCard = document.createElement('div');
            resultCard.className = 'result-card';

            const [engine, modelId] = modelKey.split('_');
            const model = this.availableModels.find(m => m.engine === engine && m.model_id === modelId);
            const displayName = model ? model.display_name : modelKey;

            if (!modelResult.success) {
                resultCard.innerHTML = `
                    <div class="result-header">
                        <h4><i class="fas fa-exclamation-triangle"></i> ${displayName}</h4>
                        <div class="result-status error">Failed</div>
                    </div>
                    <div class="result-content">
                        <p class="error-message">${modelResult.error}</p>
                    </div>
                `;
            } else {
                const processingTime = modelResult.processing_time.toFixed(2);
                const duration = modelResult.duration.toFixed(2);

                resultCard.innerHTML = `
                    <div class="result-header">
                        <h4><i class="fas fa-check-circle"></i> ${displayName}</h4>
                        <div class="result-stats">
                            <span class="stat">Processing: ${processingTime}s</span>
                            <span class="stat">Duration: ${duration}s</span>
                        </div>
                    </div>
                    <div class="result-content">
                        ${this.generateCSVTable(modelResult.csv_data)}
                    </div>
                `;
            }

            container.appendChild(resultCard);
        });
    }

    generateCSVTable(csvData) {
        if (!csvData || csvData.length === 0) {
            return '<p class="info-text">No transcription data available</p>';
        }

        const headers = csvData[0];
        const rows = csvData.slice(1);

        let tableHTML = '<div class="csv-table-container"><table class="csv-table">';

        // Headers
        tableHTML += '<thead><tr>';
        headers.forEach(header => {
            tableHTML += `<th>${header}</th>`;
        });
        tableHTML += '</tr></thead>';

        // Rows
        tableHTML += '<tbody>';
        rows.forEach(row => {
            tableHTML += '<tr>';
            row.forEach(cell => {
                tableHTML += `<td>${cell}</td>`;
            });
            tableHTML += '</tr>';
        });
        tableHTML += '</tbody></table></div>';

        return tableHTML;
    }

    showProgress(message) {
        document.getElementById('progressText').textContent = message;
        document.getElementById('progressSection').style.display = 'block';
        document.getElementById('progressFill').style.width = '0%';

        // Animate progress bar
        setTimeout(() => {
            document.getElementById('progressFill').style.width = '100%';
        }, 100);
    }

    hideProgress() {
        document.getElementById('progressSection').style.display = 'none';
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;

        const icon = {
            success: 'fas fa-check-circle',
            error: 'fas fa-exclamation-circle',
            warning: 'fas fa-exclamation-triangle',
            info: 'fas fa-info-circle'
        }[type] || 'fas fa-info-circle';

        toast.innerHTML = `
            <i class="${icon}"></i>
            <span>${message}</span>
        `;

        const container = document.getElementById('toastContainer');
        container.appendChild(toast);

        // Auto remove after 5 seconds
        setTimeout(() => {
            toast.remove();
        }, 5000);

        // Remove on click
        toast.addEventListener('click', () => {
            toast.remove();
        });
    }

    setupTheme() {
        const savedTheme = localStorage.getItem('speechhub-theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        this.updateThemeIcon(savedTheme);
    }

    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('speechhub-theme', newTheme);
        this.updateThemeIcon(newTheme);
    }

    updateThemeIcon(theme) {
        const icon = document.querySelector('#themeToggle i');
        icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }
}

// Initialize the application
let speechHub;
document.addEventListener('DOMContentLoaded', () => {
    speechHub = new SpeechHub();
});
