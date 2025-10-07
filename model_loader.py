#!/usr/bin/env python3
"""
Model loading functionality for VoxScribe - Unified model loading for all STT engines
"""

import time
import logging
from model_loaders.loader_factory import ModelLoaderFactory

logger = logging.getLogger("voxscribe")


async def load_model_async(engine, model_id):
    """Async wrapper for load_model with WebSocket notifications."""
    import asyncio

    return await asyncio.get_event_loop().run_in_executor(
        None, load_model, engine, model_id
    )


# Global registry for manager instances (set by backend.py)
_model_manager = None
_transcription_logger = None

def set_managers(model_manager, transcription_logger):
    """Set the global manager instances. Called by backend.py during initialization."""
    global _model_manager, _transcription_logger
    _model_manager = model_manager
    _transcription_logger = transcription_logger

def load_model(engine, model_id):
    """Unified model loading for all STT engines using the factory pattern."""
    from config import DEVICE

    load_start_time = time.time()

    try:
        # Use global manager instances
        if _model_manager is None or _transcription_logger is None:
            raise Exception("Model managers not initialized. Call set_managers() first.")

        # Get the appropriate loader for this engine
        loader = ModelLoaderFactory.get_loader(engine)

        # Check dependencies first
        is_supported, error_msg = loader.check_dependencies()
        if not is_supported:
            _transcription_logger.log_dependency_error(
                engine, model_id, loader.engine_name, error_msg
            )
            raise Exception(error_msg)

        cache_dir = _model_manager.get_cache_dir(engine, model_id)
        is_cached = _model_manager.is_model_cached(engine, model_id)

        # Log model load start
        _transcription_logger.log_model_load_start(engine, model_id, is_cached)

        # Setup cache environment if needed
        loader.setup_cache_environment(cache_dir)

        # Load the model using the specific loader
        result = loader.load_model(model_id, cache_dir, DEVICE)

        # Perform any post-load setup
        loader.post_load_setup(result, model_id, cache_dir)

        # Mark as cached if it wasn't before
        if not is_cached:
            _model_manager.mark_model_cached(engine, model_id, cache_dir)

        # Log successful model load
        load_time = time.time() - load_start_time
        _transcription_logger.log_model_load_complete(engine, model_id, load_time, True)

        return result

    except Exception as e:
        # Log failed model load
        load_time = time.time() - load_start_time
        error_msg = str(e)
        if _transcription_logger:
            _transcription_logger.log_model_load_complete(
                engine, model_id, load_time, False, error_msg
            )
        raise Exception(f"Error loading {engine} model: {error_msg}")


def get_model_info(engine, model_id):
    """Get information about a model using the appropriate loader."""
    try:
        loader = ModelLoaderFactory.get_loader(engine)
        return loader.get_model_info(model_id)
    except Exception as e:
        logger.error(f"Error getting model info for {engine}/{model_id}: {e}")
        return {
            "display_name": model_id,
            "size": "Unknown",
            "description": f"{engine} model {model_id}",
            "supports_timestamps": False,
            "supports_word_timestamps": False,
        }
