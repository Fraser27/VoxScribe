#!/usr/bin/env python3
"""
FastAPI backend for VoxScribe - TTS Service
"""

import os
import tempfile
import datetime
from pathlib import Path

from fastapi import (
    FastAPI,
    HTTPException,
    BackgroundTasks,
    Form,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import json

# Import our modular components
from config import TTS_MODEL_REGISTRY, BASE_MODELS_DIR
from transcription_logger import TranscriptionLogger
from logger_setup import setup_logger
from websocket_manager import WebSocketManager
from tts_manager import TTSModelManager

# Initialize FastAPI app
app = FastAPI(
    title="VoxScribe TTS API",
    description="Text-to-Speech Service API",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize logger
logger = setup_logger()

# Initialize WebSocket manager (for download progress)
websocket_manager = WebSocketManager()
transcription_logger = TranscriptionLogger()
# Initialize TTS model manager
tts_model_manager = TTSModelManager(TTS_MODEL_REGISTRY, BASE_MODELS_DIR)
tts_model_manager.scan_existing_models()

# Set the global manager instances
from global_managers import set_managers
set_managers(
    tts_model_manager=tts_model_manager,
    transcription_logger=transcription_logger,
    websocket_manager=websocket_manager,
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "VoxScribe TTS",
        "version": "1.0.0",
        "status": "running"
    }


@app.websocket("/ws/tts")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time TTS updates."""
    await websocket_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                if message.get("type") == "ping":
                    await websocket.send_json({"type": "pong", "message": "Connection active"})
                else:
                    await websocket.send_json({"type": "pong", "message": "Connection active"})
            except json.JSONDecodeError:
                await websocket.send_json({"type": "pong", "message": "Connection active"})
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)


@app.websocket("/ws/tts/stream")
async def tts_stream_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming TTS synthesis"""
    await websocket.accept()
    
    engine = None
    model_id = None
    description = None
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            msg_type = message.get("type")
            
            if msg_type == "configure":
                engine = message.get("engine")
                model_id = message.get("model_id")
                description = message.get("description", "A clear, friendly voice")
                
                if engine not in TTS_MODEL_REGISTRY or model_id not in TTS_MODEL_REGISTRY[engine]:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Invalid engine or model_id"
                    })
                    continue
                
                if not tts_model_manager.is_model_cached(engine, model_id):
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Model {engine}/{model_id} not cached"
                    })
                    continue
                
                await websocket.send_json({
                    "type": "configured",
                    "engine": engine,
                    "model_id": model_id
                })
            
            elif msg_type == "text":
                if not engine or not model_id:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Must configure before sending text"
                    })
                    continue
                
                text = message.get("text", "")
                if not text:
                    continue
                
                # Store text for synthesis
                text_to_synthesize = text
            
            elif msg_type == "finalize":
                if not text_to_synthesize:
                    await websocket.send_json({
                        "type": "error",
                        "message": "No text to synthesize"
                    })
                    continue
                
                # Generate audio
                temp_dir = tempfile.gettempdir()
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(temp_dir, f"tts_stream_{timestamp}.wav")
                
                try:
                    from tts_synthesis_engine import synthesize_speech
                    
                    result = await synthesize_speech(
                        engine=engine,
                        text=text_to_synthesize,
                        model_id=model_id,
                        description=description,
                        output_path=output_path
                    )
                    
                    if not result["success"]:
                        await websocket.send_json({
                            "type": "error",
                            "message": result.get("error", "Synthesis failed")
                        })
                        continue
                    
                    # Read audio file and send in chunks
                    with open(output_path, "rb") as f:
                        audio_data = f.read()
                    
                    import base64
                    
                    # Send audio in chunks (e.g., 4KB chunks)
                    chunk_size = 4096
                    chunk_index = 0
                    
                    for i in range(0, len(audio_data), chunk_size):
                        chunk = audio_data[i:i + chunk_size]
                        chunk_base64 = base64.b64encode(chunk).decode('utf-8')
                        
                        await websocket.send_json({
                            "type": "audio_chunk",
                            "data": chunk_base64,
                            "chunk_index": chunk_index
                        })
                        chunk_index += 1
                    
                    # Send completion
                    await websocket.send_json({
                        "type": "audio_complete",
                        "total_chunks": chunk_index,
                        "duration": result.get("audio_duration", 0),
                        "sample_rate": result.get("sample_rate", 24000)
                    })
                    
                    # Cleanup
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    
                    text_to_synthesize = ""
                    
                except Exception as e:
                    logger.error(f"Error in streaming TTS: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                    if os.path.exists(output_path):
                        os.remove(output_path)
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })
    
    except WebSocketDisconnect:
        logger.info("TTS stream client disconnected")
    except Exception as e:
        logger.error(f"TTS stream error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })


@app.get("/api/tts/models")
async def get_tts_models():
    """Get available TTS models."""
    models = []
    
    for engine, engine_models in TTS_MODEL_REGISTRY.items():
        for model_id, config in engine_models.items():
            is_cached = tts_model_manager.is_model_cached(engine, model_id)
            is_downloading = websocket_manager.is_downloading(engine, model_id)
            
            models.append({
                "engine": engine,
                "model_id": model_id,
                "display_name": tts_model_manager.get_display_name(engine, model_id),
                "size": config["size"],
                "cached": is_cached,
                "downloading": is_downloading,
                "languages": config.get("languages", []),
                "speakers": config.get("speakers", {})
            })
    
    return {"models": models}


@app.post("/api/tts/download-model")
async def download_tts_model(
    background_tasks: BackgroundTasks,
    engine: str = Form(...),
    model_id: str = Form(...),
):
    """Start async TTS model download with progress updates."""
    
    logger.info(f"TTS download request received for {engine}/{model_id}")
    
    if engine not in TTS_MODEL_REGISTRY or model_id not in TTS_MODEL_REGISTRY[engine]:
        logger.error(f"Invalid TTS engine or model_id: {engine}/{model_id}")
        raise HTTPException(status_code=400, detail="Invalid engine or model_id")
    
    if tts_model_manager.is_model_cached(engine, model_id):
        logger.info(f"TTS Model {engine}/{model_id} already cached")
        return {"success": True, "message": "Model already cached", "cached": True}
    
    if websocket_manager.is_downloading(engine, model_id):
        logger.info(f"TTS Model {engine}/{model_id} already downloading")
        return {
            "success": False,
            "message": "Download already in progress",
            "downloading": True,
        }
    
    logger.info(f"Starting background download task for TTS {engine}/{model_id}")
    background_tasks.add_task(websocket_manager.start_download_task, engine, model_id)
    
    return {
        "success": True,
        "message": "Download started",
        "downloading": True,
        "engine": engine,
        "model_id": model_id,
    }


@app.post("/api/tts/synthesize")
async def synthesize_endpoint(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    engine: str = Form(...),
    model_id: str = Form(...),
    description: str = Form(None),
    speaker: str = Form(None),
    language: str = Form(None),
):
    """Synthesize speech from text with specified TTS engine and model."""
    from tts_synthesis_engine import synthesize_speech
    
    if engine not in TTS_MODEL_REGISTRY or model_id not in TTS_MODEL_REGISTRY[engine]:
        raise HTTPException(status_code=400, detail="Invalid engine or model_id")
    
    if not tts_model_manager.is_model_cached(engine, model_id):
        raise HTTPException(
            status_code=400,
            detail=f"Model {engine}/{model_id} is not cached. Please download the model first.",
        )
    
    if websocket_manager.is_downloading(engine, model_id):
        raise HTTPException(
            status_code=400,
            detail=f"Model {engine}/{model_id} is currently downloading. Please wait for download to complete.",
        )
    
    # Create output file path
    temp_dir = tempfile.gettempdir()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"tts_{engine}_{timestamp}.wav"
    output_path = os.path.join(temp_dir, output_filename)
    
    try:
        # Build voice description if speaker/language provided
        if speaker and not description:
            description = f"{speaker}'s voice"
            if language:
                description += f" speaking {language}"
        
        # Synthesize speech
        result = await synthesize_speech(
            engine=engine,
            text=text,
            model_id=model_id,
            description=description,
            output_path=output_path
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Synthesis failed"))
        
        # Return the audio file
        def cleanup():
            if os.path.exists(output_path):
                os.remove(output_path)
        
        background_tasks.add_task(cleanup)
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=output_filename,
            headers={
                "X-Audio-Duration": str(result["audio_duration"]),
                "X-Processing-Time": str(result["processing_time"]),
                "X-Sample-Rate": str(result["sample_rate"])
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/tts/models/{engine}/{model_id}")
async def delete_tts_model_cache(engine: str, model_id: str):
    """Delete a cached TTS model."""
    
    if engine not in TTS_MODEL_REGISTRY or model_id not in TTS_MODEL_REGISTRY[engine]:
        raise HTTPException(status_code=400, detail="Invalid engine or model_id")
    
    if websocket_manager.is_downloading(engine, model_id):
        raise HTTPException(
            status_code=400, detail="Cannot delete model while it's being downloaded"
        )
    
    if not tts_model_manager.is_model_cached(engine, model_id):
        raise HTTPException(status_code=404, detail="Model is not cached")
    
    try:
        logger.info(f"Attempting to delete cached TTS model: {engine}/{model_id}")
        
        success = tts_model_manager.delete_model_cache(engine, model_id)
        
        if success:
            return {
                "success": True,
                "message": f"TTS Model {engine}/{model_id} deleted successfully",
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete model cache")
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error deleting TTS model cache {engine}/{model_id}: {e}")
        raise HTTPException(status_code=500, detail=error_msg)


if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 50)
    logger.info("VoxScribe TTS Service Starting Up")
    logger.info("=" * 50)
    logger.info(f"Base models directory: {BASE_MODELS_DIR}")
    logger.info("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8002)
