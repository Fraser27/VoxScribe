#!/usr/bin/env python3
"""
NVIDIA NeMo model loader for VoxScribe
"""

import os
from typing import Any, Tuple
from pathlib import Path
from .base_loader import BaseModelLoader
import torch


class NvidiaLoader(BaseModelLoader):
    """Loader for NVIDIA NeMo models."""
    
    def __init__(self):
        super().__init__("nvidia")
    
    def check_dependencies(self) -> Tuple[bool, str]:
        """Check if NeMo dependencies are available."""
        try:
            from dependencies import _check_nemo_support, UNIFIED_TRANSFORMERS_VERSION
            
            if not _check_nemo_support():
                return False, f"NeMo toolkit requires transformers {UNIFIED_TRANSFORMERS_VERSION}+"
            
            return True, ""
        except ImportError as e:
            return False, f"NeMo dependencies not available: {e}"
    
    def setup_cache_environment(self, cache_dir: Path) -> None:
        """Setup NeMo-specific cache environment variables."""
        # Set NeMo cache to the model-specific directory
        os.environ["NEMO_CACHE_DIR"] = str(cache_dir)
        
        # HuggingFace cache is already set globally in config.py
        # No need to override it here as NeMo models will use the global HF cache
    
    def load_model(self, model_id: str, cache_dir: Path, device: str) -> Any:
        """Load a NeMo model."""
        import nemo.collections.asr as nemo_asr
        from nemo.collections.speechlm2.models import SALM
        
        if "parakeet" in model_id:
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
            if torch.cuda.is_available():
               model = model.cuda() 
        elif "canary" in model_id:
            model = SALM.from_pretrained(model_id)
            if torch.cuda.is_available():
               model = model.cuda()
        else:
            raise ValueError(f"Unknown NVIDIA model type: {model_id}")
        
        return model
    
    def get_model_info(self, model_id: str) -> dict:
        """Get information about a NeMo model."""
        model_info_map = {
            "nvidia/parakeet-tdt-0.6b-v2": {
                "display_name": "Parakeet TDT 0.6B V2",
                "size": "2.4GB",
                "description": "NVIDIA Parakeet TDT 0.6B V2 ASR model",
                "supports_timestamps": True
            },
            "nvidia/canary-qwen-2.5b": {
                "display_name": "Canary-Qwen 2.5B",
                "size": "5GB", 
                "description": "NVIDIA Canary-Qwen 2.5B speech model",
                "supports_timestamps": False
            }
        }
        
        info = model_info_map.get(model_id, {
            "display_name": model_id,
            "size": "Unknown",
            "description": f"NVIDIA model {model_id}",
            "supports_timestamps": False
        })
        
        info.update({
            "supports_word_timestamps": False
        })
        
        return info