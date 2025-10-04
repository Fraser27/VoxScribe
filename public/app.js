// SpeechHub Frontend JavaScript

class SpeechHub {
    constructor() {
        this.currentMode = 'single';
        this.selectedFile = null;
        this.availableModels = [];
        this.selectedModels = [];

        this.init();
    }

    async init() {
        this.setupEventListeners();
        this.setupTheme();
        await this.loadStatus();
        await this.loadModels();
        await this.loadRecentLogs();
        this.updateUI();
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

        // Engine/Model selection
        document.getElementById('engineSelect').addEventListener('change', (e) => {
            this.updateModelOptions(e.target.value);
            this.updateUI();
        });

        // Model selection
        document.getElementById('modelSelect').addEventListener('change', () => {
            this.updateUI();
        });

        // Action buttons
        document.getElementById('transcribeBtn').addEventListener('click', () => {
            this.transcribe();
        });

        document.getElementById('compareBtn').addEventListener('click', () => {
            this.compareModels();
        });

        // Dependency installation
        document.getElementById('installVoxtral').addEventListener('click', () => {
            this.installDependency('voxtral');
        });

        document.getElementById('installNemo').addEventListener('click', () => {
            this.installDependency('nemo');
        });

        // Logs viewer
        document.getElementById('viewLogsBtn').addEventListener('click', () => {
            this.showLogsModal();
        });
    }

    setupTheme() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        this.updateThemeIcon(savedTheme);
    }

    toggleTheme() {
        const currentTheme = document.documentElement.getAttribute('data-theme');
        const newTheme = currentTheme === 'dark' ? 'light' : 'dark';

        document.documentElement.setAttribute('data-theme', newTheme);
        localStorage.setItem('theme', newTheme);
        this.updateThemeIcon(newTheme);
    }

    updateThemeIcon(theme) {
        const icon = document.querySelector('#themeToggle i');
        icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }

    async loadStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();

            document.getElementById('deviceStatus').textContent = data.device.toUpperCase();

            // Update dependency status
            const deps = data.dependencies;
            this.updateDependencyStatus('voxtral', deps.voxtral_supported);
            this.updateDependencyStatus('nemo', deps.nemo_supported);

            const dependencyText = [];
            if (deps.voxtral_supported) dependencyText.push('Voxtral ✅');
            else dependencyText.push('Voxtral ❌');

            if (deps.nemo_supported) dependencyText.push('NeMo ✅');
            else dependencyText.push('NeMo ❌');

            document.getElementById('dependencyStatus').textContent = dependencyText.join(', ');

        } catch (error) {
            console.error('Failed to load status:', error);
            this.showToast('Failed to load system status', 'error');
        }
    }

    updateDependencyStatus(type, supported) {
        const statusElement = document.getElementById(`${type}Status`);
        const installButton = document.getElementById(`install${type.charAt(0).toUpperCase() + type.slice(1)}`);

        if (supported) {
            statusElement.textContent = '✅ Ready';
            statusElement.className = 'dependency-status status-ready';
            installButton.style.display = 'none';
        } else {
            statusElement.textContent = '❌ Missing';
            statusElement.className = 'dependency-status status-missing';
            installButton.style.display = 'inline-block';
        }
    }

    async loadModels() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            this.availableModels = data.models;

            this.populateEngineSelect();
            this.populateCompareModels();
            this.updateCacheInfo();

        } catch (error) {
            console.error('Failed to load models:', error);
            this.showToast('Failed to load model information', 'error');
        }
    }

    populateEngineSelect() {
        const engineSelect = document.getElementById('engineSelect');
        const engines = [...new Set(this.availableModels.map(m => m.engine))];

        engineSelect.innerHTML = '<option value="">Select Engine...</option>';
        engines.forEach(engine => {
            const option = document.createElement('option');
            option.value = engine;
            
            // Special handling for engine display names
            let displayName;
            if (engine === 'nvidia') {
                displayName = 'NVIDIA';
            } else {
                displayName = engine.charAt(0).toUpperCase() + engine.slice(1);
            }
            
            option.textContent = displayName;
            engineSelect.appendChild(option);
        });
    }

    updateModelOptions(engine) {
        const modelSelect = document.getElementById('modelSelect');
        modelSelect.innerHTML = '<option value="">Select Model...</option>';

        if (!engine) return;

        const engineModels = this.availableModels.filter(m => m.engine === engine);
        engineModels.forEach(model => {
            const option = document.createElement('option');
            option.value = model.model_id;
            option.textContent = `${model.display_name} (${model.size}) ${model.cached ? '✅' : '⬇️'}`;
            modelSelect.appendChild(option);
        });
    }

    populateCompareModels() {
        const container = document.getElementById('compareModels');
        container.innerHTML = '';

        const groupedModels = {};
        this.availableModels.forEach(model => {
            if (!groupedModels[model.engine]) {
                groupedModels[model.engine] = [];
            }
            groupedModels[model.engine].push(model);
        });

        Object.entries(groupedModels).forEach(([engine, models]) => {
            const engineSection = document.createElement('div');
            engineSection.className = 'engine-section';

            const engineTitle = document.createElement('h5');
            
            // Special handling for engine display names
            let displayName;
            if (engine === 'nvidia') {
                displayName = 'NVIDIA';
            } else {
                displayName = engine.charAt(0).toUpperCase() + engine.slice(1);
            }
            
            engineTitle.textContent = displayName;
            engineTitle.style.marginBottom = '0.5rem';
            engineTitle.style.color = 'var(--primary-color)';
            engineSection.appendChild(engineTitle);

            models.forEach(model => {
                const checkbox = document.createElement('label');
                checkbox.className = 'model-checkbox';

                const input = document.createElement('input');
                input.type = 'checkbox';
                input.value = `${model.engine}:${model.model_id}`;
                input.addEventListener('change', () => this.updateSelectedModels());

                const text = document.createElement('span');
                text.textContent = `${model.display_name} (${model.size}) ${model.cached ? '✅' : '⬇️'}`;

                checkbox.appendChild(input);
                checkbox.appendChild(text);
                engineSection.appendChild(checkbox);
            });

            container.appendChild(engineSection);
        });
    }

    updateSelectedModels() {
        const checkboxes = document.querySelectorAll('#compareModels input[type="checkbox"]:checked');
        this.selectedModels = Array.from(checkboxes).map(cb => {
            const [engine, model_id] = cb.value.split(':');
            return { engine, model_id };
        });

        this.updateUI();
    }

    updateCacheInfo() {
        const cachedModels = this.availableModels.filter(m => m.cached);
        const cacheStatus = document.getElementById('cacheStatus');
        const cachedModelsContainer = document.getElementById('cachedModels');

        if (cachedModels.length === 0) {
            cacheStatus.textContent = 'No models cached yet';
            cachedModelsContainer.innerHTML = '';
        } else {
            cacheStatus.textContent = `${cachedModels.length} model(s) cached`;

            const groupedCached = {};
            cachedModels.forEach(model => {
                if (!groupedCached[model.engine]) {
                    groupedCached[model.engine] = [];
                }
                groupedCached[model.engine].push(model);
            });

            cachedModelsContainer.innerHTML = '';
            Object.entries(groupedCached).forEach(([engine, models]) => {
                const engineDiv = document.createElement('div');
                
                // Special handling for engine display names
                let displayName;
                if (engine === 'nvidia') {
                    displayName = 'NVIDIA';
                } else {
                    displayName = engine.charAt(0).toUpperCase() + engine.slice(1);
                }
                
                engineDiv.innerHTML = `<strong>${displayName}:</strong>`;
                models.forEach(model => {
                    const modelDiv = document.createElement('div');
                    modelDiv.style.fontSize = '0.875rem';
                    modelDiv.style.color = 'var(--text-secondary)';
                    modelDiv.style.marginLeft = '1rem';
                    modelDiv.textContent = `• ${model.display_name} (${model.size})`;
                    engineDiv.appendChild(modelDiv);
                });
                cachedModelsContainer.appendChild(engineDiv);
            });
        }
    }

    handleFileSelect(file) {
        const supportedFormats = ['wav', 'mp3', 'flac', 'm4a', 'ogg'];
        const fileExtension = file.name.split('.').pop().toLowerCase();

        if (!supportedFormats.includes(fileExtension)) {
            this.showToast(`Unsupported file format. Supported: ${supportedFormats.join(', ').toUpperCase()}`, 'error');
            return;
        }

        this.selectedFile = file;

        // Update file info display
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('fileSize').textContent = this.formatFileSize(file.size);

        // Create audio preview
        const audioPreview = document.getElementById('audioPreview');
        audioPreview.src = URL.createObjectURL(file);

        // Show file info
        document.getElementById('fileInfo').style.display = 'flex';

        this.updateUI();
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
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

            // Update transcribe button
            const engine = document.getElementById('engineSelect').value;
            const model = document.getElementById('modelSelect').value;
            const canTranscribe = this.selectedFile && engine && model;

            transcribeBtn.disabled = !canTranscribe;
            
            let engineDisplayName = 'Engine';
            if (engine) {
                if (engine === 'nvidia') {
                    engineDisplayName = 'NVIDIA';
                } else {
                    engineDisplayName = engine.charAt(0).toUpperCase() + engine.slice(1);
                }
            }
            
            document.getElementById('transcribeBtnText').textContent =
                canTranscribe ? `Transcribe with ${engineDisplayName}` : 'Transcribe';

        } else {
            singleSelect.style.display = 'none';
            compareSelect.style.display = 'block';
            transcribeBtn.style.display = 'none';
            compareBtn.style.display = 'inline-flex';

            // Update compare button
            const canCompare = this.selectedFile && this.selectedModels.length >= 2;
            compareBtn.disabled = !canCompare;
            document.getElementById('compareBtnText').textContent =
                canCompare ? `Compare ${this.selectedModels.length} Models` : 'Compare Models';
        }
    }

    async transcribe() {
        if (!this.selectedFile) {
            this.showToast('Please select an audio file', 'error');
            return;
        }

        const engine = document.getElementById('engineSelect').value;
        const model_id = document.getElementById('modelSelect').value;

        if (!engine || !model_id) {
            this.showToast('Please select engine and model', 'error');
            return;
        }

        // Clear previous results immediately
        this.clearResults();
        
        this.showLoading('Transcribing audio...');

        try {
            const formData = new FormData();
            formData.append('file', this.selectedFile);
            formData.append('engine', engine);
            formData.append('model_id', model_id);

            const response = await fetch('/api/transcribe', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                // Update engine display name
                let engineDisplayName;
                if (engine === 'nvidia') {
                    engineDisplayName = 'NVIDIA';
                } else {
                    engineDisplayName = engine.charAt(0).toUpperCase() + engine.slice(1);
                }
                
                this.displayResults([{
                    name: `${engineDisplayName} - ${model_id}`,
                    data: result
                }]);
                this.showToast(`Transcription completed in ${result.processing_time.toFixed(1)}s`, 'success');
                
                // Refresh models, cache info, and logs
                await this.loadModels();
                await this.loadRecentLogs();
            } else {
                throw new Error(result.error);
            }

        } catch (error) {
            console.error('Transcription failed:', error);
            this.showToast(`Transcription failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    async compareModels() {
        if (!this.selectedFile) {
            this.showToast('Please select an audio file', 'error');
            return;
        }

        if (this.selectedModels.length < 2) {
            this.showToast('Please select at least 2 models for comparison', 'error');
            return;
        }

        // Clear previous results immediately
        this.clearResults();
        
        this.showLoading('Comparing models...');
        this.showProgress(0, `Starting comparison of ${this.selectedModels.length} models...`);

        try {
            const formData = new FormData();
            formData.append('file', this.selectedFile);
            formData.append('engines', JSON.stringify(this.selectedModels));

            const response = await fetch('/api/compare', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.results) {
                const results = Object.entries(data.results).map(([key, result]) => {
                    let displayName = key.replace('_', ' - ');
                    // Update engine names in the display
                    displayName = displayName.replace(/^nvidia/, 'NVIDIA');
                    displayName = displayName.replace(/^whisper/, 'Whisper');
                    displayName = displayName.replace(/^voxtral/, 'Voxtral');
                    
                    return {
                        name: displayName,
                        data: result
                    };
                });

                this.displayResults(results);
                this.showToast('Model comparison completed', 'success');
                
                // Refresh models, cache info, and logs
                await this.loadModels();
                await this.loadRecentLogs();
            } else {
                throw new Error('No results received');
            }

        } catch (error) {
            console.error('Comparison failed:', error);
            this.showToast(`Comparison failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
            this.hideProgress();
        }
    }

    clearResults() {
        const resultsSection = document.getElementById('resultsSection');
        const resultsContent = document.getElementById('resultsContent');
        
        // Hide results section and clear content
        resultsSection.style.display = 'none';
        resultsContent.innerHTML = '';
    }

    displayResults(results) {
        const resultsSection = document.getElementById('resultsSection');
        const resultsContent = document.getElementById('resultsContent');

        resultsContent.innerHTML = '';

        results.forEach(result => {
            if (!result.data.success) {
                const errorDiv = document.createElement('div');
                errorDiv.className = 'result-item';
                errorDiv.innerHTML = `
                    <div class="result-header">
                        <span class="result-title">${result.name}</span>
                        <span class="result-meta error-text">Failed</span>
                    </div>
                    <p class="error-text">${result.data.error}</p>
                `;
                resultsContent.appendChild(errorDiv);
                return;
            }

            const resultDiv = document.createElement('div');
            resultDiv.className = 'result-item';

            const csvData = result.data.csv_data;
            const processingTime = result.data.processing_time;
            const duration = result.data.duration;

            // Create table
            const table = document.createElement('table');
            table.className = 'result-table';

            // Header
            const thead = document.createElement('thead');
            const headerRow = document.createElement('tr');
            csvData[0].forEach(header => {
                const th = document.createElement('th');
                th.textContent = header;
                headerRow.appendChild(th);
            });
            thead.appendChild(headerRow);
            table.appendChild(thead);

            // Body
            const tbody = document.createElement('tbody');
            csvData.slice(1).forEach(row => {
                const tr = document.createElement('tr');
                row.forEach(cell => {
                    const td = document.createElement('td');
                    td.textContent = cell;
                    tr.appendChild(td);
                });
                tbody.appendChild(tr);
            });
            table.appendChild(tbody);

            // Create download buttons
            const csvContent = this.arrayToCSV(csvData);
            const textContent = csvData.slice(1).map(row => row[row.length - 1]).join('\n\n');

            resultDiv.innerHTML = `
                <div class="result-header">
                    <span class="result-title">${result.name}</span>
                    <span class="result-meta">
                        ${processingTime.toFixed(1)}s processing time | ${duration.toFixed(1)}s audio
                    </span>
                </div>
            `;

            resultDiv.appendChild(table);

            const actionsDiv = document.createElement('div');
            actionsDiv.className = 'result-actions';

            const csvBtn = document.createElement('button');
            csvBtn.className = 'btn btn-small btn-primary';
            csvBtn.innerHTML = '<i class="fas fa-download"></i> Download CSV';
            csvBtn.onclick = () => this.downloadContent(csvContent, `${this.selectedFile.name}_${result.name.replace(/[^a-zA-Z0-9]/g, '_')}.csv`, 'text/csv');

            const txtBtn = document.createElement('button');
            txtBtn.className = 'btn btn-small btn-secondary';
            txtBtn.innerHTML = '<i class="fas fa-download"></i> Download Text';
            txtBtn.onclick = () => this.downloadContent(textContent, `${this.selectedFile.name}_${result.name.replace(/[^a-zA-Z0-9]/g, '_')}.txt`, 'text/plain');

            actionsDiv.appendChild(csvBtn);
            actionsDiv.appendChild(txtBtn);

            resultDiv.appendChild(actionsDiv);
            resultsContent.appendChild(resultDiv);
        });

        resultsSection.style.display = 'block';
    }

    arrayToCSV(data) {
        return data.map(row =>
            row.map(cell =>
                typeof cell === 'string' && cell.includes(',') ? `"${cell}"` : cell
            ).join(',')
        ).join('\n');
    }

    async installDependency(type) {
        this.showLoading(`Installing ${type} dependency...`);

        try {
            const response = await fetch('/api/install-dependency', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ dependency: type })
            });

            const result = await response.json();

            if (result.success) {
                this.showToast(result.message, 'success');
                // Reload status after installation
                setTimeout(() => this.loadStatus(), 2000);
            } else {
                throw new Error(result.detail || 'Installation failed');
            }

        } catch (error) {
            console.error('Installation failed:', error);
            this.showToast(`Installation failed: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    showLoading(text = 'Loading...') {
        document.getElementById('loadingText').textContent = text;
        document.getElementById('loadingOverlay').style.display = 'flex';
    }

    hideLoading() {
        document.getElementById('loadingOverlay').style.display = 'none';
    }

    showProgress(percent, text) {
        document.getElementById('progressFill').style.width = `${percent}%`;
        document.getElementById('progressText').textContent = text;
        document.getElementById('progressSection').style.display = 'block';
    }

    hideProgress() {
        document.getElementById('progressSection').style.display = 'none';
    }

    showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;

        document.getElementById('toastContainer').appendChild(toast);

        setTimeout(() => {
            toast.remove();
        }, 5000);
    }

    downloadContent(content, filename, mimeType) {
        const blob = new Blob([content], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    async loadRecentLogs() {
        try {
            const response = await fetch('/api/logs?log_type=transcriptions&limit=5');
            const data = await response.json();
            
            this.displayRecentLogs(data.logs);
            
        } catch (error) {
            console.error('Failed to load recent logs:', error);
            document.getElementById('recentLogs').innerHTML = '<p class="error-text">Failed to load logs</p>';
        }
    }

    displayRecentLogs(logs) {
        const container = document.getElementById('recentLogs');
        
        if (!logs || logs.length === 0) {
            container.innerHTML = '<p class="info-text">No recent activity</p>';
            return;
        }

        container.innerHTML = '';
        
        logs.reverse().forEach(log => {
            const logDiv = document.createElement('div');
            logDiv.className = 'log-entry';
            
            // Determine log type for styling
            if (log.event === 'transcription_complete' && log.success) {
                logDiv.classList.add('success');
            } else if (log.event === 'transcription_complete' && !log.success) {
                logDiv.classList.add('error');
            } else {
                logDiv.classList.add('info');
            }
            
            const timestamp = new Date(log.timestamp).toLocaleTimeString();
            let message = '';
            
            switch (log.event) {
                case 'transcription_start':
                    message = `Started: ${log.engine}/${log.model_id}`;
                    break;
                case 'transcription_complete':
                    if (log.success) {
                        message = `✅ ${log.engine}/${log.model_id} (${log.processing_time_seconds.toFixed(1)}s)`;
                    } else {
                        message = `❌ ${log.engine}/${log.model_id} failed`;
                    }
                    break;
                case 'model_load_start':
                    message = `Loading ${log.engine}/${log.model_id}`;
                    break;
                case 'dependency_error':
                    message = `❌ ${log.dependency} error`;
                    break;
                default:
                    message = log.event;
            }
            
            logDiv.innerHTML = `
                <div class="log-timestamp">${timestamp}</div>
                <div class="log-message">${message}</div>
            `;
            
            container.appendChild(logDiv);
        });
    }

    showLogsModal() {
        // For now, just show a simple alert. In a full implementation, 
        // you'd create a proper modal with detailed logs
        this.showToast('Full logs viewer coming soon! Check the logs/ directory for detailed logs.', 'info');
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
    new SpeechHub();
});