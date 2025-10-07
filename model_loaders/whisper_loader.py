#!/usr/bin/env python3
"""
Whisper model loader for VoxScribe
"""

import whisper
from typing import Any, Tuple
from pathlib import Path
from .base_loader import BaseModelLoader


class WhisperLoader(BaseModelLoader):
    """Loader for OpenAI Whisper models."""
    
    def __init__(self):
        super().__init__("whisper")
    
    def check_dependencies(self) -> Tuple[bool, str]:
        """Check if Whisper dependencies are available."""
        try:
            import whisper
            import torch
            return True, ""
        except ImportError as e:
            return False, f"Whisper dependencies not available: {e}"
    
    def load_model(self, model_id: str, cache_dir: Path, device: str) -> Any:
        """Load a Whisper model."""
        model = whisper.load_model(
            model_id, 
            download_root=str(cache_dir), 
            device=device
        )
        return model
    
    def get_model_info(self, model_id: str) -> dict:
        """Get information about a Whisper model."""
        # Model size information
        size_map = {
            "tiny": "39MB",
            "base": "142MB", 
            "small": "461MB",
            "medium": "1.5GB",
            "large": "2.9GB",
            "large-v2": "2.9GB",
            "large-v3": "2.9GB"
        }
        
        return {
            "display_name": f"Whisper {model_id.title()}",
            "size": size_map.get(model_id, "Unknown"),
            "description": f"OpenAI Whisper {model_id} model",
            "supports_timestamps": True,
            "supports_word_timestamps": True
        }