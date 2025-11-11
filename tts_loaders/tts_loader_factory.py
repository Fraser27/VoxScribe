#!/usr/bin/env python3
"""
TTS Model loader factory
"""

from typing import Dict, Type
from .base_tts_loader import BaseTTSLoader
from .parler_loader import ParlerTTSLoader
from .higgs_loader import HiggsAudioLoader


class TTSLoaderFactory:
    """Factory for creating TTS model loaders."""
    
    _loaders: Dict[str, Type[BaseTTSLoader]] = {
        # "parler": ParlerTTSLoader,
        "higgs": HiggsAudioLoader,
    }
    
    @classmethod
    def get_loader(cls, engine: str) -> BaseTTSLoader:
        """
        Get a TTS model loader for the specified engine.
        
        Args:
            engine: The engine name (parler, etc.)
            
        Returns:
            BaseTTSLoader: The appropriate loader instance
            
        Raises:
            ValueError: If the engine is not supported
        """
        if engine not in cls._loaders:
            raise ValueError(
                f"Unsupported TTS engine: {engine}. "
                f"Supported engines: {list(cls._loaders.keys())}"
            )
        
        loader_class = cls._loaders[engine]
        return loader_class()
    
    @classmethod
    def register_loader(cls, engine: str, loader_class: Type[BaseTTSLoader]) -> None:
        """
        Register a new TTS model loader.
        
        Args:
            engine: The engine name
            loader_class: The loader class to register
        """
        cls._loaders[engine] = loader_class
    
    @classmethod
    def get_supported_engines(cls) -> list:
        """Get list of supported TTS engines."""
        return list(cls._loaders.keys())
    
    @classmethod
    def is_engine_supported(cls, engine: str) -> bool:
        """Check if a TTS engine is supported."""
        return engine in cls._loaders
