# Unified Speech Transcription Suite

A comprehensive Streamlit application for speech-to-text transcription using OpenAI's Whisper, Mistral's Voxtral, NVIDIA's Parakeet, and NVIDIA's Canary-Qwen models with local model caching.

## Features

### Whisper Features
- **High-quality speech recognition** using OpenAI Whisper
- **Local model caching** - models are downloaded once and cached locally
- **Multiple model sizes** - choose between tiny, base, small, medium, and large
- **Detailed timestamps** - get word-level timestamps
- **Fast processing** - optimized for efficiency

### Voxtral Features  
- **Advanced audio understanding** using Mistral's Voxtral
- **Multilingual support** - automatic language detection
- **Long-form context** - process up to 30 minutes of audio
- **Audio Q&A** - ask questions about audio content
- **Function calling** - trigger workflows from voice input

### Parakeet Features
- **NVIDIA optimized** - built for high performance
- **Fast inference** - optimized for real-time processing
- **Timestamps support** - detailed timing information
- **Punctuation & capitalization** - properly formatted output
- **600M parameters** - balanced size and accuracy

### Canary-Qwen Features
- **State-of-the-art English ASR** - best-in-class accuracy
- **Ultra-fast processing** - 418 RTFx performance
- **Dual mode operation** - ASR mode and LLM mode
- **Punctuation & capitalization** - properly formatted output
- **2.5B parameters** - optimized for commercial use

### Common Features
- **Multiple audio formats** - supports WAV, MP3, FLAC, M4A, OGG
- **Export options** - download as CSV, TXT, or SRT subtitle format
- **macOS compatible** - Whisper models work, others need a GPU
- **Unified interface** - compare models side-by-side

## Installation

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
```

2. Install dependencies:

**Core dependencies (Whisper + Voxtral):**
```bash
pip install -r requirements.txt
```

**For Parakeet and Canary-Qwen support (optional):**
```bash
# Standard NeMo (for Parakeet)
pip install nemo_toolkit[asr]

# Latest NeMo trunk (for Canary-Qwen)
python -m pip install "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git"
```

3. Run the application:
```bash
streamlit run app_unified.py
```

## Model Caching

Models are automatically cached in the `models/` directory within the project:

- **Whisper models**: Cached in `models/` directory
- **Voxtral models**: Cached in `models/voxtral/` directory  
- **Parakeet models**: Cached in `models/parakeet/` directory
- **Canary-Qwen models**: Cached in `models/canary/` directory
- **First run**: Downloads and caches the selected model
- **Subsequent runs**: Loads from local cache (much faster)
- **Cache management**: Use the sidebar to view cached models and clear cache if needed

### Whisper Model Sizes and Performance

| Model | Parameters | Speed | Accuracy | Size |
|-------|------------|-------|----------|------|
| tiny  | 39M        | ~32x  | Good     | ~39MB |
| base  | 74M        | ~16x  | Better   | ~139MB |
| small | 244M       | ~6x   | Great    | ~244MB |
| medium| 769M       | ~2x   | Excellent| ~769MB |
| large | 1550M      | 1x    | Best     | ~1550MB |

### Voxtral Model Sizes and Performance

| Model | Parameters | Speed | Accuracy | Size | GPU Memory |
|-------|------------|-------|----------|------|------------|
| Voxtral Mini 3B | 3B | Fast | Excellent | ~6GB | 8GB+ |
| Voxtral Small 24B | 24B | Slower | Superior | ~48GB | 24GB+ |

### Parakeet Model Sizes and Performance

| Model | Parameters | Speed | Accuracy | Size | GPU Memory |
|-------|------------|-------|----------|------|------------|
| Parakeet TDT 0.6B V2 | 600M | Very Fast | Excellent | ~2.4GB | 2GB+ |

### Canary-Qwen Model Sizes and Performance

| Model | Parameters | Speed | Accuracy | Size | GPU Memory |
|-------|------------|-------|----------|------|------------|
| Canary-Qwen 2.5B | 2.5B | Ultra Fast (418 RTFx) | State-of-the-art | ~5GB | 4GB+ |

## Usage

1. **Select Model**: Choose your preferred model size in the sidebar
2. **Upload Audio**: Drag and drop or select an audio file
3. **Transcribe**: Click the transcribe button
4. **Export**: Download results in your preferred format

## Testing

**Test individual models:**
```bash
python test_whisper.py    # Test Whisper
python test_voxtral.py    # Test Voxtral  
python test_canary.py     # Test Canary-Qwen
python test_cache.py      # Test cache management
```

## Pre-downloading Models

**Download models using the unified downloader:**
```bash
# Download Whisper models
python download_models.py --engine whisper --whisper-models tiny base

# Download Voxtral models  
python download_models.py --engine voxtral --voxtral-models mistralai/Voxtral-Mini-3B-2507

# Download Parakeet models
python download_models.py --engine parakeet --parakeet-models nvidia/parakeet-tdt-0.6b-v2

# Download Canary-Qwen models
python download_canary.py

# Download all models
python download_models.py --engine all
```

## Project Structure

```
├── app_unified.py              # Unified Streamlit app (main)
├── test_whisper.py             # Whisper test script
├── test_voxtral.py             # Voxtral test script
├── test_canary.py              # Canary-Qwen test script
├── test_cache.py               # Cache management test
├── download_models.py          # Unified model downloader
├── download_canary.py          # Canary-Qwen model downloader
├── requirements.txt            # All dependencies
├── models/                     # Cached Whisper models (auto-created)
├── models/voxtral/             # Cached Voxtral models (auto-created)
├── models/parakeet/            # Cached Parakeet models (auto-created)
├── models/canary/              # Cached Canary-Qwen models (auto-created)
├── Dataset/                    # Sample audio files
├── .gitignore                  # Excludes models from version control
└── README.md                   # This file
```

## Model Comparison

### When to Use Whisper
- ✅ Need detailed word-level timestamps
- ✅ Want fast processing with lower resource requirements
- ✅ Working with shorter audio files
- ✅ Need proven accuracy across various accents
- ✅ Limited GPU memory available

### When to Use Voxtral  
- ✅ Need advanced audio understanding beyond transcription
- ✅ Want to ask questions about audio content
- ✅ Working with long-form audio (up to 30 minutes)
- ✅ Need multilingual support with automatic detection
- ✅ Have sufficient GPU resources (8GB+ VRAM)

### When to Use Parakeet
- ✅ Need NVIDIA-optimized performance
- ✅ Want fast inference with good accuracy
- ✅ Working with 16kHz audio files
- ✅ Need punctuation and capitalization
- ✅ Have moderate GPU resources (2GB+ VRAM)

### When to Use Canary-Qwen
- ✅ Need state-of-the-art English ASR accuracy
- ✅ Want ultra-fast processing (418 RTFx)
- ✅ Need both ASR and LLM capabilities
- ✅ Working with English audio content
- ✅ Have moderate GPU resources (4GB+ VRAM)

## System Requirements

### Whisper
- **CPU**: Any modern processor
- **RAM**: 4GB+ recommended
- **Storage**: 2GB+ for all models
- **GPU**: Optional (CUDA support available)

### Voxtral
- **CPU**: Modern multi-core processor
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for models
- **GPU**: 8GB+ VRAM for Mini, 24GB+ for Small model
- **CUDA**: Required for optimal performance

### Parakeet
- **CPU**: Modern multi-core processor
- **RAM**: 8GB+ recommended
- **Storage**: 3GB+ for models
- **GPU**: 2GB+ VRAM recommended
- **CUDA**: Required for optimal performance

### Canary-Qwen
- **CPU**: Modern multi-core processor
- **RAM**: 8GB+ recommended
- **Storage**: 6GB+ for models
- **GPU**: 4GB+ VRAM recommended
- **CUDA**: Required for optimal performance

## Troubleshooting

### SSL Certificate Issues
If you encounter SSL certificate errors on macOS:
```bash
/Applications/Python\ 3.13/Install\ Certificates.command
```

### Whisper Issues
- Check internet connection for first download
- Clear cache and try again
- Try a smaller model first (tiny or base)
- CPU transcription works well for most use cases

### Voxtral Issues
- Ensure sufficient GPU memory is available
- Check CUDA installation for GPU acceleration
- Try the Mini model first before Small model
- Monitor system resources during processing

### Parakeet Issues
- Install NeMo toolkit: `pip install nemo_toolkit[asr]`
- Ensure CUDA is properly installed
- Check that audio is 16kHz for optimal performance
- Monitor GPU memory usage

### Canary-Qwen Issues
- Install latest NeMo: `python -m pip install "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git"`
- Ensure PyTorch 2.6+ is installed for FSDP2 support
- Check that audio is 16kHz mono for optimal performance
- Monitor GPU memory usage during processing

### Performance Tips
- **Whisper**: Use smaller models for speed, larger for accuracy
- **Voxtral**: Ensure adequate GPU memory to avoid OOM errors
- **Parakeet**: Works best with 16kHz mono audio
- **Canary-Qwen**: Optimized for 16kHz mono audio, ultra-fast processing
- **All models**: Process shorter audio clips for faster results
- **All models**: Close other applications to free up resources

## License : MIT

This project uses:
- OpenAI's Whisper model - refer to their license terms
- Mistral's Voxtral model - refer to their license terms  
- NVIDIA's Parakeet model - refer to their license terms (CC-BY-4.0)
- NVIDIA's Canary-Qwen model - refer to their license terms (Commercial use ready)
