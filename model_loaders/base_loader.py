#!/usr/bin/env python3
"""
Base model loader interface for VoxScribe
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple
from pathlib import Path


class BaseModelLoader(ABC):
    """Abstract base class for all model loaders."""
    
    def __init__(self, engine_name: str):
        self.engine_name = engine_name
    
    @abstractmethod
    def check_dependencies(self) -> Tuple[bool, str]:
        """
        Check if all required dependencies are available.
        
        Returns:
            Tuple[bool, str]: (is_supported, error_message)
        """
        pass
    
    @abstractmethod
    def load_model(self, model_id: str, cache_dir: Path, device: str) -> Any:
        """
        Load the model with the given configuration.
        
        Args:
            model_id: The model identifier
            cache_dir: Directory to cache the model
            device: Device to load the model on (cpu/cuda)
            
        Returns:
            The loaded model object(s)
        """
        pass
    
    @abstractmethod
    def get_model_info(self, model_id: str) -> dict:
        """
        Get information about the model.
        
        Args:
            model_id: The model identifier
            
        Returns:
            Dictionary with model information
        """
        pass
    
    def setup_cache_environment(self, cache_dir: Path) -> None:
        """
        Setup cache environment variables if needed.
        Default implementation does nothing.
        
        Args:
            cache_dir: Directory to use for caching
        """
        pass
    
    def post_load_setup(self, model: Any, model_id: str, cache_dir: Path) -> None:
        """
        Perform any post-load setup if needed.
        Default implementation does nothing.
        
        Args:
            model: The loaded model
            model_id: The model identifier
            cache_dir: Directory used for caching
        """
        pass