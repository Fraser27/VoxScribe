#!/usr/bin/env python3
"""
TTS Model Manager - handles TTS model caching and loading
"""

import os
import json
import datetime
import shutil
from pathlib import Path
from typing import Dict, Optional, Any
from logger_setup import setup_logger

logger = setup_logger()


class TTSModelManager:
    """Manages TTS model caching and loading."""
    
    def __init__(self, model_registry: Dict, base_models_dir: Path):
        self.model_registry = model_registry
        self.base_models_dir = base_models_dir
        self.loaded_models = {}
        self.cache_info_file = base_models_dir / "tts_cache_info.json"
        self.load_cache_info()
    
    def load_cache_info(self):
        """Load cached model information."""
        if self.cache_info_file.exists():
            with open(self.cache_info_file, "r") as f:
                self.cache_info = json.load(f)
        else:
            self.cache_info = {}
    
    def save_cache_info(self):
        """Save cached model information."""
        self.base_models_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_info_file, "w") as f:
            json.dump(self.cache_info, f, indent=2)
        
    def scan_existing_models(self):
        """Scan for existing cached TTS models."""
        cached_count = 0
        for engine, models in self.model_registry.items():
            for model_id, config in models.items():
                if self.is_model_cached(engine, model_id):
                    cached_count += 1
        
        logger.info(f"Found {cached_count} cached TTS models")
        return cached_count
    
    def is_model_cached(self, engine: str, model_id: str) -> bool:
        """Check if a TTS model is cached."""
        if engine not in self.model_registry:
            return False
        
        if model_id not in self.model_registry[engine]:
            return False
        
        cache_dir = self.model_registry[engine][model_id].get("cache_dir")
        if not cache_dir:
            return False
        
        # Check if cache directory exists and has content
        if not cache_dir.exists():
            return False
        
        # Check for model files
        has_files = any(cache_dir.iterdir())
        return has_files
    
    def get_cache_dir(self, engine: str, model_id: str) -> Optional[Path]:
        """Get the cache directory for a TTS model."""
        if engine not in self.model_registry:
            return None
        
        if model_id not in self.model_registry[engine]:
            return None
        
        return self.model_registry[engine][model_id].get("cache_dir")
    
    def get_display_name(self, engine: str, model_id: str) -> str:
        """Get the display name for a TTS model."""
        if engine not in self.model_registry:
            return model_id
        
        if model_id not in self.model_registry[engine]:
            return model_id
        
        config = self.model_registry[engine][model_id]
        return config.get("display_name", model_id)
    
    def delete_model_cache(self, engine: str, model_id: str) -> bool:
        """Delete a cached TTS model."""
        cache_dir = self.get_cache_dir(engine, model_id)
        
        if not cache_dir or not cache_dir.exists():
            logger.warning(f"Cache directory not found for {engine}/{model_id}")
            return False
        
        try:
            # Remove the cache directory
            shutil.rmtree(cache_dir)
            logger.info(f"Deleted TTS model cache: {engine}/{model_id}")
            
            # Remove from loaded models if present
            model_key = f"{engine}:{model_id}"
            if model_key in self.loaded_models:
                del self.loaded_models[model_key]
            
            return True
        except Exception as e:
            logger.error(f"Failed to delete TTS model cache {engine}/{model_id}: {e}")
            return False
    
    def get_model_config(self, engine: str, model_id: str) -> Optional[Dict]:
        """Get the configuration for a TTS model."""
        if engine not in self.model_registry:
            return None
        
        if model_id not in self.model_registry[engine]:
            return None
        
        return self.model_registry[engine][model_id]
    
    def mark_model_cached(self, engine: str, model_id: str, cache_path: Path) -> None:
        """Mark a model as cached."""
        cache_key = f"{engine}:{model_id}"
        self.cache_info[cache_key] = {
            "engine": engine,
            "model_id": model_id,
            "cache_path": str(cache_path),
            "cached_at": datetime.datetime.now().isoformat(),
            "size": self.get_model_size(engine, model_id),
        }
        self.save_cache_info()
    
    def get_model_size(self, engine: str, model_id: str) -> str:
        """Get the size of a model from the registry."""
        if engine not in self.model_registry:
            return "Unknown"
        
        if model_id not in self.model_registry[engine]:
            return "Unknown"
        
        config = self.model_registry[engine][model_id]
        return config.get("size", "Unknown")
