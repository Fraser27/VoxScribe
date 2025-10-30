#!/usr/bin/env python3
"""
Model loader factory for VoxScribe
"""

from typing import Dict, Type
from .base_loader import BaseModelLoader
from .whisper_loader import WhisperLoader
from .voxtral_loader import VoxtralLoader
from .nvidia_loader import NvidiaLoader
from .granite_loader import GraniteLoader


class ModelLoaderFactory:
    """Factory for creating model loaders."""
    
    _loaders: Dict[str, Type[BaseModelLoader]] = {
        "whisper": WhisperLoader,
        "voxtral": VoxtralLoader,
        "nvidia": NvidiaLoader,
        "granite": GraniteLoader,
    }
    
    @classmethod
    def get_loader(cls, engine: str) -> BaseModelLoader:
        """
        Get a model loader for the specified engine.
        
        Args:
            engine: The engine name (whisper, voxtral, nvidia, granite)
            
        Returns:
            BaseModelLoader: The appropriate loader instance
            
        Raises:
            ValueError: If the engine is not supported
        """
        if engine not in cls._loaders:
            raise ValueError(f"Unsupported engine: {engine}. Supported engines: {list(cls._loaders.keys())}")
        
        loader_class = cls._loaders[engine]
        return loader_class()
    
    @classmethod
    def register_loader(cls, engine: str, loader_class: Type[BaseModelLoader]) -> None:
        """
        Register a new model loader.
        
        Args:
            engine: The engine name
            loader_class: The loader class to register
        """
        cls._loaders[engine] = loader_class
    
    @classmethod
    def get_supported_engines(cls) -> list:
        """Get list of supported engines."""
        return list(cls._loaders.keys())
    
    @classmethod
    def is_engine_supported(cls, engine: str) -> bool:
        """Check if an engine is supported."""
        return engine in cls._loaders