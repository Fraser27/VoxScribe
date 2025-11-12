#!/usr/bin/env python3
"""
FastAPI backend for VoxScribe - STT + Frontend Service
"""

import os
import json
import time
import tempfile
import datetime
from pathlib import Path
from typing import List, Dict
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
from fastapi.responses import JSONResponse, Response
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
    _check_granite_support,
    get_transformers_version,
    UNIFIED_TRANSFORMERS_VERSION,
)
from logger_setup import setup_logger
from websocket_manager import WebSocketManager
from transcription_logger import TranscriptionLogger
from transcription_manager import TranscriptionManager
from stt_manager import STTModelManager
from transcription_engine import transcribe_audio

# Ensure correct transformers version before proceeding
print("Checking transformers version...")
current_version = ensure_transformers_version()

if current_version is None:
    print("✗ Failed to ensure correct transformers version. Application may not work correctly.")
    print("Please manually install transformers >= 4.57.0")
else:
    print(f"✓ Using transformers version: {current_version}")

# Initialize FastAPI app
app = FastAPI(
    title="VoxScribe STT API",
    description="Speech-to-Text Platform API",
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

# No static files - frontend is separate service

# Initialize logger
logger = setup_logger()

# Initialize WebSocket manager
websocket_manager = WebSocketManager()

# Initialize transcription logger
transcription_logger = TranscriptionLogger()
transcription_manager = TranscriptionManager(TRANSCRIPTIONS_DIR)

# Initialize model manager
stt_model_manager = STTModelManager(MODEL_REGISTRY, BASE_MODELS_DIR)
stt_model_manager.scan_existing_models()

# Set the global manager instances
from global_managers import set_managers
set_managers(
    stt_model_manager=stt_model_manager,
    transcription_logger=transcription_logger,
    transcription_manager=transcription_manager,
    websocket_manager=websocket_manager,
)

# Pydantic models
class TranscriptionRequest(BaseModel):
    engine: str
    model_id: str


class ComparisonRequest(BaseModel):
    engines: List[Dict[str, str]]


# API Routes
@app.get("/")
async def root():
    """Root endpoint."""
    return {"service": "VoxScribe STT", "version": "1.0.0", "status": "running"}


@app.get("/api/stt/status")
async def get_status():
    """Get system and dependency status."""
    return {
        "device": DEVICE,
        "supported_formats": SUPPORTED_FORMATS,
        "dependencies": {
            "voxtral_supported": _check_voxtral_support(),
            "nemo_supported": _check_nemo_support(),
            "granite_supported": _check_granite_support(),
            "transformers_version": get_transformers_version(),
            "required_transformers_version": UNIFIED_TRANSFORMERS_VERSION,
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "stt",
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


@app.get("/api/stt/models")
async def get_models():
    """Get available STT models."""
    models = []

    for engine, engine_models in MODEL_REGISTRY.items():
        for model_id, config in engine_models.items():
            is_cached = stt_model_manager.is_model_cached(engine, model_id)
            is_downloading = websocket_manager.is_downloading(engine, model_id)

            models.append(
                {
                    "engine": engine,
                    "model_id": model_id,
                    "display_name": stt_model_manager.get_display_name(engine, model_id),
                    "size": config["size"],
                    "cached": is_cached,
                    "downloading": is_downloading,
                }
            )

    return {"models": models}


@app.post("/api/stt/download-model")
async def download_model(
    background_tasks: BackgroundTasks,
    engine: str = Form(...),
    model_id: str = Form(...),
):
    """Start async model download with WebSocket progress updates."""

    logger.info(f"Download request received for {engine}/{model_id}")

    if engine not in MODEL_REGISTRY or model_id not in MODEL_REGISTRY[engine]:
        logger.error(f"Invalid engine or model_id: {engine}/{model_id}")
        raise HTTPException(status_code=400, detail="Invalid engine or model_id")

    if stt_model_manager.is_model_cached(engine, model_id):
        logger.info(f"Model {engine}/{model_id} already cached")
        return {"success": True, "message": "Model already cached", "cached": True}

    if websocket_manager.is_downloading(engine, model_id):
        logger.info(f"Model {engine}/{model_id} already downloading")
        return {
            "success": False,
            "message": "Download already in progress",
            "downloading": True,
        }

    logger.info(f"Starting background download task for {engine}/{model_id}")
    background_tasks.add_task(websocket_manager.start_download_task, engine, model_id)

    return {
        "success": True,
        "message": "Download started",
        "downloading": True,
        "engine": engine,
        "model_id": model_id,
    }


@app.post("/api/stt/transcribe")
async def transcribe_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    engine: str = Form(...),
    model_id: str = Form(...),
):
    """Transcribe audio file with specified engine and model."""

    if engine not in MODEL_REGISTRY or model_id not in MODEL_REGISTRY[engine]:
        raise HTTPException(status_code=400, detail="Invalid engine or model_id")

    if not stt_model_manager.is_model_cached(engine, model_id):
        raise HTTPException(
            status_code=400,
            detail=f"Model {engine}/{model_id} is not cached. Please download the model first.",
        )

    if websocket_manager.is_downloading(engine, model_id):
        raise HTTPException(
            status_code=400,
            detail=f"Model {engine}/{model_id} is currently downloading. Please wait for download to complete.",
        )

    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: {SUPPORTED_FORMATS}",
        )

    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, file.filename)

    try:
        with open(audio_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        result = await transcribe_audio(
            engine, audio_path, model_id, file.filename, len(content)
        )

        background_tasks.add_task(
            lambda: os.remove(audio_path) if os.path.exists(audio_path) else None
        )

        return result

    except Exception as e:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stt/compare")
async def compare_models(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    engines: str = Form(...),
):
    """Compare multiple models on the same audio file."""

    try:
        engine_configs = json.loads(engines)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid engines configuration")

    if len(engine_configs) < 2:
        raise HTTPException(
            status_code=400, detail="At least 2 models required for comparison"
        )

    file_extension = file.filename.split(".")[-1].lower()
    if file_extension not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: {SUPPORTED_FORMATS}",
        )

    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, file.filename)

    try:
        with open(audio_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        comparison_start_time = time.time()
        transcription_logger.log_comparison_start(
            engine_configs, file.filename, len(content)
        )

        results = {}
        for config in engine_configs:
            engine = config.get("engine")
            model_id = config.get("model_id")

            if not engine or not model_id:
                continue

            result = await transcribe_audio(
                engine,
                audio_path,
                model_id,
                file.filename,
                len(content),
                save_to_history=True,
            )
            results[f"{engine}_{model_id}"] = result

        comparison_time = time.time() - comparison_start_time
        transcription_logger.log_comparison_complete(
            engine_configs, file.filename, comparison_time, results
        )

        background_tasks.add_task(
            lambda: os.remove(audio_path) if os.path.exists(audio_path) else None
        )

        return {"results": results}

    except Exception as e:
        if os.path.exists(audio_path):
            os.remove(audio_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/stt/models/{engine}/{model_id}")
async def delete_model_cache(engine: str, model_id: str):
    """Delete a cached model."""

    if engine not in MODEL_REGISTRY or model_id not in MODEL_REGISTRY[engine]:
        raise HTTPException(status_code=400, detail="Invalid engine or model_id")

    if websocket_manager.is_downloading(engine, model_id):
        raise HTTPException(
            status_code=400, detail="Cannot delete model while it's being downloaded"
        )

    if not stt_model_manager.is_model_cached(engine, model_id):
        raise HTTPException(status_code=404, detail="Model is not cached")

    try:
        logger.info(f"Attempting to delete cached model: {engine}/{model_id}")

        success = stt_model_manager.delete_model_cache(engine, model_id)

        if success:
            transcription_logger.log_model_delete(engine, model_id, True)

            return {
                "success": True,
                "message": f"Model {engine}/{model_id} deleted successfully",
            }
        else:
            transcription_logger.log_model_delete(
                engine, model_id, False, "Failed to delete model cache"
            )
            raise HTTPException(status_code=500, detail="Failed to delete model cache")

    except Exception as e:
        error_msg = str(e)
        transcription_logger.log_model_delete(engine, model_id, False, error_msg)
        logger.error(f"Error deleting model cache {engine}/{model_id}: {e}")
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/api/stt/transcriptions")
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


@app.get("/api/stt/transcriptions/{transcription_id}")
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


@app.delete("/api/stt/transcriptions/{transcription_id}")
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


@app.get("/api/stt/transcriptions/{transcription_id}/download")
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

        output = io.StringIO()
        writer = csv.writer(output)

        for row in transcription["csv_data"]:
            writer.writerow(row)

        csv_content = output.getvalue()
        output.close()

        timestamp = transcription["timestamp"][:10]
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


# TTS endpoints are handled by the TTS service directly
# Frontend service routes requests appropriately


if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 50)
    logger.info("VoxScribe STT Service Starting Up")
    logger.info("=" * 50)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Supported formats: {SUPPORTED_FORMATS}")
    logger.info(f"Base models directory: {BASE_MODELS_DIR}")
    logger.info("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8001)
