#!/usr/bin/env python3
"""
Voxtral model loader for VoxScribe
"""

import torch
from typing import Any, Tuple
from pathlib import Path
from .base_loader import BaseModelLoader


class VoxtralLoader(BaseModelLoader):
    """Loader for Mistral Voxtral models."""
    
    def __init__(self):
        super().__init__("voxtral")
    
    def check_dependencies(self) -> Tuple[bool, str]:
        """Check if Voxtral dependencies are available."""
        try:
            from dependencies import _check_voxtral_support, UNIFIED_TRANSFORMERS_VERSION
            
            if not _check_voxtral_support():
                return False, f"Voxtral requires transformers {UNIFIED_TRANSFORMERS_VERSION}+"
            
            return True, ""
        except ImportError as e:
            return False, f"Voxtral dependencies not available: {e}"
    
    def load_model(self, model_id: str, cache_dir: Path, device: str) -> Any:
        """Load a Voxtral model."""
        from transformers import VoxtralForConditionalGeneration, AutoProcessor
        
        processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=str(cache_dir)
        )
        model = VoxtralForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=str(cache_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        return (model, processor)
    
    def get_model_info(self, model_id: str) -> dict:
        """Get information about a Voxtral model."""
        model_info_map = {
            "mistralai/Voxtral-Mini-3B-2507": {
                "display_name": "Voxtral Mini 3B",
                "size": "6GB",
                "description": "Mistral Voxtral Mini 3B speech model"
            },
            "mistralai/Voxtral-Small-24B-2507": {
                "display_name": "Voxtral Small 24B", 
                "size": "48GB",
                "description": "Mistral Voxtral Small 24B speech model"
            }
        }
        
        info = model_info_map.get(model_id, {
            "display_name": model_id,
            "size": "Unknown",
            "description": f"Voxtral model {model_id}"
        })
        
        info.update({
            "supports_timestamps": False,
            "supports_word_timestamps": False
        })
        
        return info