#!/usr/bin/env python3
"""
Configuration constants and settings for VoxScribe
"""

import os
import torch
from pathlib import Path

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Audio format support
SUPPORTED_FORMATS = ["wav", "mp3", "flac", "m4a", "ogg"]

# Processing limits
MAX_RECOMMENDED_DURATION = 30 * 60

# Transformers version requirement
UNIFIED_TRANSFORMERS_VERSION = "4.57.0"

# Directory paths
BASE_MODELS_DIR = Path(os.getcwd()) / "models"
TRANSCRIPTIONS_DIR = Path(os.getcwd()) / "transcriptions"

# Ensure directories exist
BASE_MODELS_DIR.mkdir(exist_ok=True)
TRANSCRIPTIONS_DIR.mkdir(exist_ok=True)

# Set global cache directories for Hugging Face and NeMo
os.environ["HF_HOME"] = str(BASE_MODELS_DIR / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(BASE_MODELS_DIR / "huggingface" / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(BASE_MODELS_DIR / "huggingface" / "transformers")

# Model configurations - Speech-to-Text (STT)
MODEL_REGISTRY = {
    "whisper": {
        "tiny": {"size": "39MB", "cache_dir": BASE_MODELS_DIR / "whisper"},
        "base": {"size": "142MB", "cache_dir": BASE_MODELS_DIR / "whisper"},
        "small": {"size": "461MB", "cache_dir": BASE_MODELS_DIR / "whisper"},
        "medium": {"size": "1.5GB", "cache_dir": BASE_MODELS_DIR / "whisper"},
        "large": {"size": "2.9GB", "cache_dir": BASE_MODELS_DIR / "whisper"},
        "large-v2": {"size": "2.9GB", "cache_dir": BASE_MODELS_DIR / "whisper"},
        "large-v3": {"size": "2.9GB", "cache_dir": BASE_MODELS_DIR / "whisper"},
    },
    "voxtral": {
        "mistralai/Voxtral-Mini-3B-2507": {
            "size": "6GB",
            "cache_dir": BASE_MODELS_DIR / "voxtral" / "mini-3b",
            "display_name": "Voxtral Mini 3B",
        },
        "mistralai/Voxtral-Small-24B-2507": {
            "size": "48GB",
            "cache_dir": BASE_MODELS_DIR / "voxtral" / "small-24b",
            "display_name": "Voxtral Small 24B",
        },
    },
    "nvidia": {
        "nvidia/parakeet-tdt-0.6b-v2": {
            "size": "2.4GB",
            "cache_dir": BASE_MODELS_DIR / "nvidia" / "parakeet-tdt-0.6b-v2",
            "display_name": "Parakeet TDT 0.6B V2",
        },
        "nvidia/canary-qwen-2.5b": {
            "size": "5GB",
            "cache_dir": BASE_MODELS_DIR / "nvidia" / "canary-qwen-2.5b",
            "display_name": "Canary-Qwen 2.5B",
        },
    },
    "granite": {
        "ibm-granite/granite-speech-3.3-2b": {
            "size": "10GB",
            "cache_dir": BASE_MODELS_DIR / "granite" / "granite-speech-3.3-2b",
            "display_name": "IBM-Granite-Speech 3.3 2B",
        },
        "ibm-granite/granite-speech-3.3-8b": {
            "size": "17GB",
            "cache_dir": BASE_MODELS_DIR / "granite" / "granite-speech-3.3-8b",
            "display_name": "IBM-Granite-Speech 3.3 8B",
        }
    }
}

# TTS Model configurations - Text-to-Speech
TTS_MODEL_REGISTRY = {
    "parler": {
        "parler-tts/parler-tts-mini-multilingual-v1.1": {
            "size": "1.2GB",
            "cache_dir": BASE_MODELS_DIR / "parler" / "mini-multilingual-v1.1",
            "display_name": "Parler-TTS Mini Multilingual",
            "languages": ["English", "French", "Spanish", "Portuguese", "Polish", "German", "Italian", "Dutch"],
            "speakers": {
                "English": ["Steven", "Olivia", "Megan"],
                "French": ["Daniel", "Michelle", "Christine", "Megan"],
                "German": ["Nicole", "Christopher", "Megan", "Michelle"],
                "Italian": ["Julia", "Richard", "Megan"],
                "Polish": ["Alex", "Natalie"],
                "Portuguese": ["Sophia", "Nicholas"],
                "Spanish": ["Steven", "Olivia", "Megan"],
                "Dutch": ["Mark", "Jessica", "Michelle"]
            }
        }
    }
}

# Ensure all model directories exist
for engine_models in MODEL_REGISTRY.values():
    for model_config in engine_models.values():
        if isinstance(model_config, dict) and "cache_dir" in model_config:
            model_config["cache_dir"].mkdir(parents=True, exist_ok=True)

# Ensure all TTS model directories exist
for engine_models in TTS_MODEL_REGISTRY.values():
    for model_config in engine_models.values():
        if isinstance(model_config, dict) and "cache_dir" in model_config:
            model_config["cache_dir"].mkdir(parents=True, exist_ok=True)