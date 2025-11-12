#!/usr/bin/env python3
"""
Global manager registry for VoxScribe - Avoids circular imports
"""

# Global registry for manager instances
_managers = {}

def set_managers(**kwargs):
    """Set the global manager instances. Called by backend.py during initialization."""
    global _managers
    _managers.update(kwargs)

def get_managers():
    """Get all manager instances as a dictionary."""
    return _managers

def get_model_manager(mgr_type:str):
    """Get the global model manager instance."""
    if mgr_type=="stt":
        if "stt_model_manager" not in _managers:
            raise RuntimeError("STT model manager not initialized. Call set_managers() first.")
        return _managers["stt_model_manager"]
    elif mgr_type=="tts":
        if "tts_model_manager" not in _managers:
            raise RuntimeError("TTS model manager not initialized. Call set_managers() first.")
        return _managers["tts_model_manager"]
    else:
        raise ValueError(f"Invalid manager type: {mgr_type}. Must be 'stt' or 'tts'")


def get_transcription_logger():
    """Get the global transcription logger instance."""
    if "transcription_logger" not in _managers:
        raise RuntimeError("Transcription logger not initialized. Call set_managers() first.")
    return _managers["transcription_logger"]

def get_transcription_manager():
    """Get the global transcription manager instance."""
    if "transcription_manager" not in _managers:
        raise RuntimeError("Transcription manager not initialized. Call set_managers() first.")
    return _managers["transcription_manager"]

def get_websocket_manager():
    """Get the global websocket manager instance."""
    if "websocket_manager" not in _managers:
        raise RuntimeError("WebSocket manager not initialized. Call set_managers() first.")
    return _managers["websocket_manager"]

def get_tts_model_manager():
    """Get the global TTS model manager instance."""
    if "tts_model_manager" not in _managers:
        raise RuntimeError("TTS model manager not initialized. Call set_managers() first.")
    return _managers["tts_model_manager"]

def get_loader_factory(mgr_type: str):
    """
    Get the appropriate loader factory based on manager type.
    
    Args:
        mgr_type: Either 'stt' or 'tts'
    
    Returns:
        The appropriate factory class (STTModelLoaderFactory or TTSLoaderFactory)
    
    Raises:
        ValueError: If mgr_type is not 'stt' or 'tts'
    """
    if mgr_type == "stt":
        from stt_loaders.loader_factory import STTModelLoaderFactory
        return STTModelLoaderFactory
    elif mgr_type == "tts":
        from tts_loaders.tts_loader_factory import TTSLoaderFactory
        return TTSLoaderFactory
    else:
        raise ValueError(f"Invalid manager type: {mgr_type}. Must be 'stt' or 'tts'")