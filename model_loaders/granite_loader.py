#!/usr/bin/env python3
"""
IBM Granite Speech model loader for VoxScribe
"""

import torch
import torchaudio
from typing import Any, Tuple
from pathlib import Path
from .base_loader import BaseModelLoader


class GraniteLoader(BaseModelLoader):
    """Loader for IBM Granite Speech models."""
    
    def __init__(self):
        super().__init__("granite")
    
    def check_dependencies(self) -> Tuple[bool, str]:
        """Check if Granite dependencies are available."""
        try:
            from dependencies import _check_granite_support, UNIFIED_TRANSFORMERS_VERSION
            
            if not _check_granite_support():
                return False, f"Granite Speech requires transformers {UNIFIED_TRANSFORMERS_VERSION}+"
            
            return True, ""
        except ImportError as e:
            return False, f"Granite dependencies not available: {e}"
    
    def setup_cache_environment(self, cache_dir: Path) -> None:
        """Setup Granite-specific cache environment variables."""
        # HuggingFace cache is already set globally in config.py
        # Granite models will use the global HF cache, no need to override
        pass
    
    def load_model(self, model_id: str, cache_dir: Path, device: str) -> Any:
        """Load a Granite Speech model."""
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        
        # Use the global HuggingFace cache (set in config.py)
        # The models will be downloaded to the HF cache automatically
        processor = AutoProcessor.from_pretrained(model_id)
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.bfloat16
        )
        
        return {
            "model": model,
            "processor": processor,
            "tokenizer": processor.tokenizer
        }

    
    def get_model_info(self, model_id: str) -> dict:
        """Get information about a Granite Speech model."""
        model_info_map = {
            "ibm-granite/granite-speech-3.3-2b": {
                "display_name": "Granite Speech 3.3-2B",
                "size": "10GB",
                "description": "IBM Granite Speech 3.3-2B ASR/AST model with two-pass design"
            },
            "ibm-granite/granite-speech-3.3-8b": {
                "display_name": "Granite Speech 3.3-8B",
                "size": "17GB",
                "description": "IBM Granite Speech 3.3-8B ASR/AST model with two-pass design"
            }
        }
        
        info = model_info_map.get(model_id, {
            "display_name": model_id,
            "size": "Unknown",
            "description": f"IBM Granite Speech model {model_id}"
        })
        
        info.update({
            "supports_timestamps": True,
            "supports_word_timestamps": False,
            "is_two_pass": True,
            "supports_translation": True
        })
        
        return info