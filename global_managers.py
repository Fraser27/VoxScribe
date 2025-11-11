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

def get_model_manager():
    """Get the global model manager instance."""
    if "model_manager" not in _managers:
        raise RuntimeError("Model manager not initialized. Call set_managers() first.")
    return _managers["model_manager"]

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