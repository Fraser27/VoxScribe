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
        import os
        os.environ["HF_HOME"] = str(cache_dir / "huggingface")
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "huggingface" / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "huggingface" / "transformers")
        
        # Ensure the huggingface cache directory exists
        (cache_dir / "huggingface" / "hub").mkdir(parents=True, exist_ok=True)
        (cache_dir / "huggingface" / "transformers").mkdir(parents=True, exist_ok=True)
    
    def load_model(self, model_id: str, cache_dir: Path, device: str) -> Any:
        """Load a Granite Speech model."""
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        
        processor = AutoProcessor.from_pretrained(
            model_id, 
            cache_dir=str(cache_dir)
        )
        
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
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
                "size": "4.6GB",
                "description": "IBM Granite Speech 3.3-2B ASR/AST model with two-pass design"
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