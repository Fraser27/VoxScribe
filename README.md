# VoxScribe 🎙️

## VoxScribe: A platform to test Opensource Speech-to-Text models

VoxScribe is a lightweight, unified platform for testing and comparing multiple open-source speech-to-text (STT) models through a single interface. Born from real-world enterprise challenges where proprietary STT solutions become prohibitively expensive at scale, VoxScribe democratizes access to cutting-edge open-source alternatives.

## The Problem We Solve

Startups transcribing speech at scale face a common dilemma: **cost vs. control**. A contact center processing 100,000 hours of calls monthly can easily spend \$150,000+ on transcription alone. While open-source STT models like Whisper, Voxtral, Parakeet, and Canary-Qwen now rival proprietary solutions in accuracy, evaluating them has been a nightmare:

- **Dependency Hell** 🔥: Conflicting library versions between models (transformers version conflicts between Voxtral and NeMo models)
- **Inconsistent APIs** 🔄: Each model requires different integration approaches
- **Complex Setup** ⚙️: Hours or days managing CUDA drivers, Python environments, and debugging
- **Limited Comparison** 📊: No unified way to test multiple models against your specific use cases

## What VoxScribe Offers

✅ **Unified Interface**: Test 5+ open-source STT models through a single FastAPI backend and clean web UI  
✅ **Dependency Management**: Handles version conflicts and library incompatibilities automatically  
✅ **Side-by-Side Comparison**: Upload audio and compare transcriptions across multiple models  
✅ **Model Caching**: Intelligent caching for faster subsequent runs  
✅ **Clean API**: RESTful endpoints for easy integration into existing workflows  
✅ **Cost Control**: Self-hosted solution puts you in control of transcription costs  

## Supported Models

- **OpenAI Whisper** - Industry standard baseline
- **Mistral Voxtral** - Latest transformer-based approach
- **NVIDIA Parakeet** - Enterprise-grade accuracy
- **Canary-Qwen-2.5B** - Multilingual capabilities
- **And growing...** - Easy to add new models


## Architecture

```
├── backend.py          # FastAPI backend with STT logic
├── public/             # Frontend static files
│   ├── index.html      # Main HTML interface
│   ├── styles.css      # CSS styling with dark/light theme
│   └── app.js          # JavaScript frontend logic
├── run.py              # Startup script
└── requirements.txt    # Python dependencies
```

[![Watch the video](https://github.com/user-attachments/assets/cf5be7f8-ce30-4f72-8d2e-05904ca48b9e)](https://www.youtube.com/watch?v=mX9L-x2zj6k)

## Features

### Backend (FastAPI)
- **RESTful API** for all STT operations
- **Unified model management** for Whisper, Voxtral, Parakeet, Canary
- **Automatic dependency handling** with version conflict resolution
- **File upload and processing** with background tasks
- **Model comparison** endpoint for side-by-side evaluation
- **Dependency installation** endpoints with subprocess management

### Frontend (HTML/CSS/JS)
- **Modern responsive design** with dark/light theme toggle
- **Drag & drop file upload** with audio preview
- **Real-time status updates** for dependencies and models
- **Single model transcription** with engine/model selection
- **Multi-model comparison** with checkbox selection
- **Progress tracking** and result visualization
- **Download options** for CSV and text formats

## Quick Start

### Prerequisites
- AWS EC2 g6.xlarge instance with Amazon Linux 2023
- NVIDIA GPU drivers installed

### Installation Steps

1. **Install NVIDIA GRID drivers**
   ```bash
   # Follow AWS documentation for GRID driver installation
   # https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html#nvidia-GRID-driver
   ```

2. **Verify CUDA installation**
   ```bash
   nvidia-smi
   ```

3. **Install system dependencies**
   ```bash
   sudo dnf update -y
   sudo dnf install git -y
   ```

4. **Install Miniconda**
   ```bash
   wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```
   - Accept the license agreement (type `yes`)
   - Confirm installation location (default is fine)
   - Initialize Conda (type `yes` when prompted)

5. **Restart your shell or source bashrc**
   ```bash
   source ~/.bashrc
   ```

6. **Create and activate conda environment**
   ```bash
   conda create -n voxscribe python=3.12 -y
   conda activate voxscribe
   ```

7. **Install ffmpeg in Conda env**
   ```bash
   conda install ffmpeg -y
   ```

8. **Clone the repository**
   ```bash
   git clone https://github.com/Fraser27/VoxScribe.git
   cd VoxScribe
   ```

9. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

10. **Start the application**
   ```bash
   python run.py
   ```

11. **Open your browser**
    ```
    http://localhost:8000
    ```

## API Endpoints

### System Status
- `GET /api/status` - Get system and dependency status
- `GET /api/models` - Get available models and cache status

### Transcription
- `POST /api/transcribe` - Single model transcription
- `POST /api/compare` - Multi-model comparison

### Dependencies
- `POST /api/install-dependency` - Install missing dependencies

## Model Support

| Engine | Models | Dependencies | Features |
|--------|--------|--------------|----------|
| **Whisper** | tiny, base, small, medium, large, large-v2, large-v3 | ✅ Built-in | Detailed timestamps, multiple sizes |
| **Voxtral** | Mini-3B, Small-24B | transformers 4.56.0+ | Advanced audio understanding, multilingual |
| **Parakeet** | TDT-0.6B-V2 | NeMo toolkit | NVIDIA optimized, fast inference |
| **Canary** | Qwen-2.5B | NeMo toolkit | State-of-the-art English ASR |

## Dependency Management

The system automatically handles version conflicts between:
- **Voxtral**: Requires transformers 4.56.0+
- **NeMo models**: Require transformers 4.51.3

Installation buttons are provided in the UI for missing dependencies.

## File Support

Supported audio formats: **WAV, MP3, FLAC, M4A, OGG**

## Development

### Backend Development
```bash
# Run with auto-reload
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
Static files are served from the `public/` directory. Changes to HTML, CSS, or JS files are reflected immediately.

### Adding New Models
1. Update `MODEL_REGISTRY` in `backend.py`
2. Add loading logic in `load_model()` function
3. Add transcription logic in `transcribe_audio()` function

## Benefits over Streamlit

1. **No ScriptRunContext warnings** - Clean separation eliminates context issues
2. **Better performance** - FastAPI is faster and more efficient
3. **Modern UI** - Custom HTML/CSS/JS with better UX
4. **API-first design** - Can be integrated with other applications
5. **Easier deployment** - Standard web application deployment
6. **Better error handling** - Proper HTTP status codes and error responses
7. **Scalability** - Can handle multiple concurrent requests

## Deployment

### Local Development
```bash
python run.py
```

### Production
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker (Optional)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Use the install buttons in the UI
2. **Model download failures**: Check internet connection and disk space
3. **Audio processing errors**: Ensure ffmpeg is installed
4. **CUDA issues**: Check PyTorch CUDA installation

### Logs
Server logs are displayed in the terminal where you run `python run.py`.

## Contributing

1. Backend changes: Modify `backend.py`
2. Frontend changes: Modify files in `public/`
3. New features: Add API endpoints and corresponding UI elements
4. Testing: Use the built-in FastAPI docs at `/docs`
