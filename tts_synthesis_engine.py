#!/usr/bin/env python3
"""
TTS Synthesis Engine - handles text-to-speech synthesis
"""

import time
import traceback
from pathlib import Path
from typing import Dict, Any
import soundfile as sf

from config import DEVICE, TTS_MODEL_REGISTRY
from logger_setup import setup_logger
from tts_loaders.tts_loader_factory import TTSLoaderFactory

logger = setup_logger()


async def synthesize_speech(
    engine: str,
    text: str,
    model_id: str,
    description: str = None,
    output_path: str = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Synthesize speech from text using specified TTS engine and model.
    
    Args:
        engine: TTS engine name (e.g., 'parler')
        text: Text to synthesize
        model_id: Model identifier
        description: Voice description/prompt
        output_path: Path to save the audio file
        **kwargs: Additional synthesis parameters
        
    Returns:
        Dictionary with synthesis results
    """
    from global_managers import get_managers
    managers = get_managers()
    tts_model_manager = managers.get("tts_model_manager")
    websocket_manager = managers.get("websocket_manager")
    
    start_time = time.time()
    
    try:
        # Validate engine and model
        if engine not in TTS_MODEL_REGISTRY:
            raise ValueError(f"Unknown TTS engine: {engine}")
        
        if model_id not in TTS_MODEL_REGISTRY[engine]:
            raise ValueError(f"Unknown model: {model_id} for engine {engine}")
        
        # Get cache directory
        cache_dir = tts_model_manager.get_cache_dir(engine, model_id)
        if not cache_dir:
            raise ValueError(f"No cache directory configured for {engine}/{model_id}")
        
        # Send progress update
        if websocket_manager:
            await websocket_manager.send_synthesis_progress(
                engine, model_id, text, "loading_model", "Loading TTS model...", 10
            )
        
        # Get model loader
        loader = TTSLoaderFactory.get_loader(engine)
        
        # Check dependencies
        is_supported, error_msg = loader.check_dependencies()
        if not is_supported:
            raise RuntimeError(f"TTS engine {engine} not supported: {error_msg}")
        
        # Load model
        logger.info(f"Loading TTS model: {engine}/{model_id}")
        model_load_start = time.time()
        
        model = loader.load_model(model_id, cache_dir, DEVICE)
        
        model_load_time = time.time() - model_load_start
        logger.info(f"Model loaded in {model_load_time:.2f}s")
        
        # Send progress update
        if websocket_manager:
            await websocket_manager.send_synthesis_progress(
                engine, model_id, text, "synthesizing", "Synthesizing speech...", 50
            )
        
        # Synthesize speech
        logger.info(f"Synthesizing speech with {engine}/{model_id}")
        synthesis_start = time.time()
        
        audio_array, sample_rate = loader.synthesize(
            model, text, description, **kwargs
        )
        
        synthesis_time = time.time() - synthesis_start
        logger.info(f"Synthesis completed in {synthesis_time:.2f}s")
        
        # Save audio file if output path provided
        if output_path:
            sf.write(output_path, audio_array, sample_rate)
            logger.info(f"Audio saved to: {output_path}")
        
        # Calculate total processing time
        total_time = time.time() - start_time
        
        # Calculate audio duration
        audio_duration = len(audio_array) / sample_rate
        
        # Send completion update
        if websocket_manager:
            await websocket_manager.send_synthesis_complete(
                engine, model_id, text, True, audio_duration, total_time
            )
        
        result = {
            "success": True,
            "engine": engine,
            "model_id": model_id,
            "text": text,
            "description": description,
            "audio_duration": audio_duration,
            "sample_rate": sample_rate,
            "processing_time": total_time,
            "model_load_time": model_load_time,
            "synthesis_time": synthesis_time,
            "output_path": output_path,
        }
        
        logger.info(f"TTS synthesis successful: {engine}/{model_id}")
        return result
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"TTS synthesis failed: {error_msg}")
        logger.error(traceback.format_exc())
        
        # Send error update
        if websocket_manager:
            await websocket_manager.send_synthesis_complete(
                engine, model_id, text, False, 0, 0, error_msg
            )
        
        return {
            "success": False,
            "engine": engine,
            "model_id": model_id,
            "text": text,
            "error": error_msg,
            "processing_time": time.time() - start_time,
        }
