#!/usr/bin/env python3
"""
Global manager registry for VoxScribe - Avoids circular imports
"""

# Global registry for manager instances
_model_manager = None
_transcription_logger = None
_transcription_manager = None
_websocket_manager = None

def set_managers(model_manager=None, transcription_logger=None, transcription_manager=None, websocket_manager=None):
    """Set the global manager instances. Called by backend.py during initialization."""
    global _model_manager, _transcription_logger, _transcription_manager, _websocket_manager
    
    if model_manager is not None:
        _model_manager = model_manager
    if transcription_logger is not None:
        _transcription_logger = transcription_logger
    if transcription_manager is not None:
        _transcription_manager = transcription_manager
    if websocket_manager is not None:
        _websocket_manager = websocket_manager

def get_model_manager():
    """Get the global model manager instance."""
    if _model_manager is None:
        raise RuntimeError("Model manager not initialized. Call set_managers() first.")
    return _model_manager

def get_transcription_logger():
    """Get the global transcription logger instance."""
    if _transcription_logger is None:
        raise RuntimeError("Transcription logger not initialized. Call set_managers() first.")
    return _transcription_logger

def get_transcription_manager():
    """Get the global transcription manager instance."""
    if _transcription_manager is None:
        raise RuntimeError("Transcription manager not initialized. Call set_managers() first.")
    return _transcription_manager

def get_websocket_manager():
    """Get the global websocket manager instance."""
    if _websocket_manager is None:
        raise RuntimeError("WebSocket manager not initialized. Call set_managers() first.")
    return _websocket_manager