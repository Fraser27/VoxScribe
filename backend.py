#!/usr/bin/env python3
"""
FastAPI backend for VoxScribe - Universal STT Platform
"""

import os
import json
import time
import tempfile
import datetime
from pathlib import Path
from typing import List, Dict, Optional
import io
import csv

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    BackgroundTasks,
    Form,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our modular components
from config import (
    DEVICE,
    SUPPORTED_FORMATS,
    MODEL_REGISTRY,
    BASE_MODELS_DIR,
    TRANSCRIPTIONS_DIR,
)
from dependencies import (
    ensure_transformers_version,
    _check_voxtral_support,
    _check_nemo_support,
    get_transformers_version,
    UNIFIED_TRANSFORMERS_VERSION,
)
from logger_setup import setup_logger
from websocket_manager import WebSocketManager
from transcription_logger import TranscriptionLogger
from transcription_manager import TranscriptionManager
from model_manager import ModelManager
from transcription_engine import transcribe_audio

# Ensure correct transformers version before proceeding
print("Checking transformers version...")
current_version = ensure_transformers_version()

if current_version is None:
    print(
        "✗ Failed to ensure correct transformers version. Application may not work correctly."
    )
    print("Please manually install transformers >= 4.57.0")
else:
    print(f"✓ Using transformers version: {current_version}")

# Initialize FastAPI app
app = FastAPI(
    title="VoxScribe API",
    description="Universal Speech-to-Text Platform API",
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

# Mount static files
app.mount("/static", StaticFiles(directory="public"), name="static")

# Initialize logger
logger = setup_logger()

# Initialize WebSocket manager
websocket_manager = WebSocketManager()

# Initialize transcription logger
transcription_logger = TranscriptionLogger()

# Initialize transcription manager
transcription_manager = TranscriptionManager(TRANSCRIPTIONS_DIR)

# Initialize model manager
model_manager = ModelManager(MODEL_REGISTRY, BASE_MODELS_DIR)
model_manager.scan_existing_models()

# Set the global manager instances for model_loader
from model_loader import set_managers
set_managers(model_manager, transcription_logger)

# Pydantic models
class TranscriptionRequest(BaseModel):
    engine: str
    model_id: str


class ComparisonRequest(BaseModel):
    engines: List[Dict[str, str]]  # [{"engine": "whisper", "model_id": "base"}, ...]


class DependencyStatus(BaseModel):
    voxtral_supported: bool
    nemo_supported: bool
    transformers_version: Optional[str] = None


class ModelInfo(BaseModel):
    engine: str
    model_id: str
    display_name: str
    size: str
    cached: bool


# API Routes
@app.get("/")
async def serve_index():
    """Serve the main HTML page."""
    return FileResponse("public/index.html")


@app.get("/api/status")
async def get_status():
    """Get system and dependency status."""
    return {
        "device": DEVICE,
        "supported_formats": SUPPORTED_FORMATS,
        "dependencies": {
            "voxtral_supported": _check_voxtral_support(),
            "nemo_supported": _check_nemo_support(),
            "transformers_version": get_transformers_version(),
            "required_transformers_version": UNIFIED_TRANSFORMERS_VERSION,
        },
    }


@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify API is working."""
    return {
        "message": "API is working!",
        "timestamp": datetime.datetime.now().isoformat(),
    }


@app.get("/api/logs")
async def get_logs(log_type: str = "transcriptions", limit: int = 100):
    """Get recent log entries."""
    try:
        if log_type == "transcriptions":
            log_file = transcription_logger.transcription_log
        elif log_type == "models":
            log_file = transcription_logger.model_usage_log
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid log_type. Use 'transcriptions' or 'models'",
            )

        if not log_file.exists():
            return {"logs": [], "total": 0}

        logs = []
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Get the last 'limit' lines
        recent_lines = lines[-limit:] if len(lines) > limit else lines

        for line in recent_lines:
            try:
                log_entry = json.loads(line.strip())
                logs.append(log_entry)
            except json.JSONDecodeError:
                continue

        return {"logs": logs, "total": len(logs), "log_type": log_type}

    except Exception as e:
        logger.error(f"Error retrieving logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            # Echo back for connection testing
            await websocket.send_json({"type": "pong", "message": "Connection active"})
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        websocket_manager.disconnect(websocket)


@app.get("/api/models")
async def get_models():
    """Get available models for all engines."""
    models = []

    for engine, engine_models in MODEL_REGISTRY.items():
        for model_id, config in engine_models.items():
            is_cached = model_manager.is_model_cached(engine, model_id)
            is_downloading = websocket_manager.is_downloading(engine, model_id)

            models.append(
                {
                    "engine": engine,
                    "model_id": model_id,
                    "display_name": model_manager.get_display_name(engine, model_id),
                    "size": config["size"],
                    "cached": is_cached,
                    "downloading": is_downloading,
                }
            )

    return {"models": models}


@app.get("/api/test-download/{engine}/{model_id}")
async def test_download(engine: str, model_id: str):
    """Test endpoint to manually trigger download and check WebSocket flow."""
    logger.info(f"Test download endpoint called for {engine}/{model_id}")

    # Test WebSocket connection count
    connection_count = len(websocket_manager.active_connections)
    logger.info(f"Active WebSocket connections: {connection_count}")

    # Test sending a message
    await websocket_manager.send_download_progress(
        engine, model_id, 0, "test", "Testing WebSocket connection"
    )

    return {
        "message": "Test download triggered",
        "websocket_connections": connection_count,
        "engine": engine,
        "model_id": model_id,
    }


@app.get("/api/download-status")
async def get_download_status():
    """Get current download status for all models."""
    downloads = []
    for task_key, task in websocket_manager.download_tasks.items():
        if not task.done():
            engine, model_id = task_key.split(":", 1)
            model = None
            for eng, models in MODEL_REGISTRY.items():
                if eng == engine and model_id in models:
                    model = {
                        "engine": engine,
                        "model_id": model_id,
                        "display_name": model_manager.get_display_name(
                            engine, model_id
                        ),
                        "size": models[model_id]["size"],
                    }
                    break

            if model:
                downloads.append(model)

    return {"downloads": downloads}


@app.post("/api/download-model")
async def download_model(
    background_tasks: BackgroundTasks,
    engine: str = Form(...),
    model_id: str = Form(...),
):
    """Start async model download with WebSocket progress updates."""

    logger.info(f"Download request received for {engine}/{model_id}")

    # Validate engine and model
    if engine not in MODEL_REGISTRY or model_id not in MODEL_REGISTRY[engine]:
        logger.error(f"Invalid engine or model_id: {engine}/{model_id}")
        raise HTTPException(status_code=400, detail="Invalid engine or model_id")

    # Check if already cached
    if model_manager.is_model_cached(engine, model_id):
        logger.info(f"Model {engine}/{model_id} already cached")
        return {"success": True, "message": "Model already cached", "cached": True}

    # Check if already downloading
    if websocket_manager.is_downloading(engine, model_id):
        logger.info(f"Model {engine}/{model_id} already downloading")
        return {
            "success": False,
            "message": "Download already in progress",
            "downloading": True,
        }

    # Start download task in background
    logger.info(f"Starting background download task for {engine}/{model_id}")
    background_tasks.add_task(websocket_manager.start_download_task, engine, model_id)

    return {
        "success": True,
        "message": "Download started",
        "downloading": True,
        "engine": engine,
        "model_id": model_id,
    }


@app.post("/api/transcribe")
async def transcribe_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    engine: str = Form(...),
    model_id: str = Form(...),
):
    """Transcribe audio file with specified engine and model."""

    # Validate engine and model
    if engine not in MODEL_REGISTRY or model_id not in MODEL_REGISTRY[engine]:
        raise HTTPException(status_code=400, detail="Invalid engine or model_id")

    # Check if model is cached (required for transcription)
    if not model_manager.is_model_cached(engine, model_id):
        raise HTTPException(
            status_code=400,
            detail=f"Model {engine}/{model_id} is not cached. Please download the model first.",
        )

    # Check if model is currently downloading
    if websocket_manager.is_downloading(engine, model_id):
        raise HTTPException(
            status_code=400,
            detail=f"Model {engine}/{model_id} is currently downloading. Please wait for download to complete.",
        )

    # Validate file format
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: {SUPPORTED_FORMATS}",
        )

    # Save uploaded file
    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, file.filename)

    try:
        with open(audio_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Transcribe (model is guaranteed to be cached)
        result = transcribe_audio(
            engine, audio_path, model_id, file.filename, len(content)
        )

        # Schedule cleanup
        background_tasks.add_task(
            lambda: os.remove(audio_path) if os.path.exists(audio_path) else None
        )

        return result

    except Exception as e:
        # Cleanup on error
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare")
async def compare_models(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    engines: str = Form(...),  # JSON string of engine configs
):
    """Compare multiple models on the same audio file."""

    # Parse engines configuration
    try:
        engine_configs = json.loads(engines)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid engines configuration")

    if len(engine_configs) < 2:
        raise HTTPException(
            status_code=400, detail="At least 2 models required for comparison"
        )

    # Validate file format
    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: {SUPPORTED_FORMATS}",
        )

    # Save uploaded file
    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, file.filename)

    try:
        with open(audio_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Log comparison start
        comparison_start_time = time.time()
        transcription_logger.log_comparison_start(
            engine_configs, file.filename, len(content)
        )

        # Run comparisons
        results = {}
        for config in engine_configs:
            engine = config.get("engine")
            model_id = config.get("model_id")

            if not engine or not model_id:
                continue

            result = transcribe_audio(
                engine,
                audio_path,
                model_id,
                file.filename,
                len(content),
                save_to_history=True,
            )
            results[f"{engine}_{model_id}"] = result

        # Log comparison completion
        comparison_time = time.time() - comparison_start_time
        transcription_logger.log_comparison_complete(
            engine_configs, file.filename, comparison_time, results
        )

        # Schedule cleanup
        background_tasks.add_task(
            lambda: os.remove(audio_path) if os.path.exists(audio_path) else None
        )

        return {"results": results}

    except Exception as e:
        # Cleanup on error
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/models/{engine}/{model_id}")
async def delete_model_cache(engine: str, model_id: str):
    """Delete a cached model."""

    # Validate engine and model
    if engine not in MODEL_REGISTRY or model_id not in MODEL_REGISTRY[engine]:
        raise HTTPException(status_code=400, detail="Invalid engine or model_id")

    # Check if model is currently being used (downloading)
    if websocket_manager.is_downloading(engine, model_id):
        raise HTTPException(
            status_code=400, detail="Cannot delete model while it's being downloaded"
        )

    # Check if model is actually cached
    if not model_manager.is_model_cached(engine, model_id):
        raise HTTPException(status_code=404, detail="Model is not cached")

    try:
        # Log the deletion attempt
        logger.info(f"Attempting to delete cached model: {engine}/{model_id}")

        # Delete the model cache
        success = model_manager.delete_model_cache(engine, model_id)

        if success:
            # Log successful deletion
            transcription_logger.log_model_delete(engine, model_id, True)

            return {
                "success": True,
                "message": f"Model {engine}/{model_id} deleted successfully",
            }
        else:
            # Log failed deletion
            transcription_logger.log_model_delete(
                engine, model_id, False, "Failed to delete model cache"
            )
            raise HTTPException(status_code=500, detail="Failed to delete model cache")

    except Exception as e:
        error_msg = str(e)
        # Log failed deletion
        transcription_logger.log_model_delete(engine, model_id, False, error_msg)
        logger.error(f"Error deleting model cache {engine}/{model_id}: {e}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/api/transcriptions")
async def get_transcriptions(limit: int = 50, offset: int = 0):
    """Get list of transcription history."""
    try:
        result = transcription_manager.get_transcriptions_list(
            limit=limit, offset=offset
        )
        return result
    except Exception as e:
        logger.error(f"Error getting transcriptions list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/transcriptions/{transcription_id}")
async def get_transcription(transcription_id: str):
    """Get a specific transcription by ID."""
    try:
        transcription = transcription_manager.get_transcription(transcription_id)
        if not transcription:
            raise HTTPException(status_code=404, detail="Transcription not found")
        return transcription
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting transcription {transcription_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/transcriptions/{transcription_id}")
async def delete_transcription(transcription_id: str):
    """Delete a transcription from history."""
    try:
        success = transcription_manager.delete_transcription(transcription_id)
        if not success:
            raise HTTPException(status_code=404, detail="Transcription not found")
        return {"success": True, "message": "Transcription deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting transcription {transcription_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/transcriptions/{transcription_id}/download")
async def download_transcription_csv(transcription_id: str):
    """Download transcription as CSV file."""
    try:
        transcription = transcription_manager.get_transcription(transcription_id)
        if not transcription:
            raise HTTPException(status_code=404, detail="Transcription not found")

        if not transcription.get("csv_data"):
            raise HTTPException(
                status_code=404, detail="No transcription data available"
            )

        # Convert to CSV format
        output = io.StringIO()
        writer = csv.writer(output)

        for row in transcription["csv_data"]:
            writer.writerow(row)

        csv_content = output.getvalue()
        output.close()

        # Create filename
        timestamp = transcription["timestamp"][:10]  # YYYY-MM-DD
        filename = f"transcription_{timestamp}_{transcription['engine']}_{transcription['model_id']}.csv"

        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading transcription {transcription_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Log startup information
    logger.info("=" * 50)
    logger.info("VoxScribe Starting Up")
    logger.info("=" * 50)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Supported formats: {SUPPORTED_FORMATS}")
    logger.info(f"Base models directory: {BASE_MODELS_DIR}")
    logger.info(f"Voxtral support: {_check_voxtral_support()}")
    logger.info(f"NeMo support: {_check_nemo_support()}")
    logger.info("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000)