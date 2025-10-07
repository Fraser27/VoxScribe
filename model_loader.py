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


from global_managers import get_model_manager, get_transcription_logger

def load_model(engine, model_id):
    """Unified model loading for all STT engines using the factory pattern."""
    from config import DEVICE

    load_start_time = time.time()

    try:
        # Get global manager instances
        model_manager = get_model_manager()
        transcription_logger = get_transcription_logger()

        # Get the appropriate loader for this engine
        loader = ModelLoaderFactory.get_loader(engine)

        # Check dependencies first
        is_supported, error_msg = loader.check_dependencies()
        if not is_supported:
            transcription_logger.log_dependency_error(
                engine, model_id, loader.engine_name, error_msg
            )
            raise Exception(error_msg)

        cache_dir = model_manager.get_cache_dir(engine, model_id)
        is_cached = model_manager.is_model_cached(engine, model_id)

        # Log model load start
        transcription_logger.log_model_load_start(engine, model_id, is_cached)

        # Setup cache environment if needed
        loader.setup_cache_environment(cache_dir)

        # Load the model using the specific loader
        result = loader.load_model(model_id, cache_dir, DEVICE)

        # Perform any post-load setup
        loader.post_load_setup(result, model_id, cache_dir)

        # Mark as cached if it wasn't before
        if not is_cached:
            model_manager.mark_model_cached(engine, model_id, cache_dir)

        # Log successful model load
        load_time = time.time() - load_start_time
        transcription_logger.log_model_load_complete(engine, model_id, load_time, True)

        return result

    except Exception as e:
        # Log failed model load
        load_time = time.time() - load_start_time
        error_msg = str(e)
        try:
            transcription_logger = get_transcription_logger()
            transcription_logger.log_model_load_complete(
                engine, model_id, load_time, False, error_msg
            )
        except:
            pass  # If logging fails, don't crash the whole operation
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
