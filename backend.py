#!/usr/bin/env python3
"""
FastAPI backend for SpeechHub - Universal STT Platform
"""

import os
import sys
import json
import time
import tempfile
import subprocess
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
import datetime
import gc

import torch
import whisper
from pydub import AudioSegment
import numpy as np
import pandas as pd

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
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
from typing import Set, Dict

# Initialize FastAPI app
app = FastAPI(
    title="SpeechHub API",
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

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED_FORMATS = ["wav", "mp3", "flac", "m4a", "ogg"]
MAX_RECOMMENDED_DURATION = 30 * 60

# Base model directory
BASE_MODELS_DIR = Path(os.getcwd()) / "models"

# Transcription storage directory
TRANSCRIPTIONS_DIR = Path(os.getcwd()) / "transcriptions"
TRANSCRIPTIONS_DIR.mkdir(exist_ok=True)

# Set global cache directories for Hugging Face and NeMo
os.environ["HF_HOME"] = str(BASE_MODELS_DIR / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(BASE_MODELS_DIR / "huggingface" / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(BASE_MODELS_DIR / "huggingface" / "transformers")


# Setup logging
def setup_logger():
    """Setup comprehensive logging for SpeechHub."""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger("speechhub")
    logger.setLevel(logging.INFO)

    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # File handler for all logs
    file_handler = logging.FileHandler(logs_dir / "speechhub.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(detailed_formatter)

    # File handler for errors only
    error_handler = logging.FileHandler(logs_dir / "errors.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(error_handler)
    logger.addHandler(console_handler)

    return logger


# Initialize logger
logger = setup_logger()


class WebSocketManager:
    """Centralized WebSocket connection manager for real-time updates."""

    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.download_tasks: Dict[str, asyncio.Task] = {}

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(
            f"WebSocket connected. Total connections: {len(self.active_connections)}"
        )

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        logger.info(
            f"WebSocket disconnected. Total connections: {len(self.active_connections)}"
        )

    async def send_to_all(self, message: dict):
        """Send message to all connected clients."""
        logger.info(
            f"Attempting to send message to {len(self.active_connections)} WebSocket connections"
        )
        if not self.active_connections:
            logger.warning("No active WebSocket connections to send message to")
            return

        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
                logger.info(f"Successfully sent message to WebSocket connection")
            except Exception as e:
                logger.warning(f"Failed to send message to WebSocket: {e}")
                disconnected.add(connection)

        # Remove disconnected clients
        for connection in disconnected:
            self.disconnect(connection)

    async def send_download_progress(
        self,
        engine: str,
        model_id: str,
        progress: float,
        status: str,
        message: str = "",
    ):
        """Send model download progress update."""
        logger.info(
            f"Sending download progress: {engine}/{model_id} - {progress}% - {status} - {message}"
        )
        await self.send_to_all(
            {
                "type": "download_progress",
                "engine": engine,
                "model_id": model_id,
                "progress": progress,
                "status": status,
                "message": message,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

    async def send_download_complete(
        self, engine: str, model_id: str, success: bool, error: str = None
    ):
        """Send model download completion notification."""
        await self.send_to_all(
            {
                "type": "download_complete",
                "engine": engine,
                "model_id": model_id,
                "success": success,
                "error": error,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

    async def send_dependency_progress(
        self,
        dependency: str,
        progress: float,
        status: str,
        message: str = "",
        engine: str = None,
        model_id: str = None,
    ):
        """Send dependency installation progress update."""
        logger.info(
            f"Sending dependency progress: {dependency} - {progress}% - {status} - {message}"
        )
        await self.send_to_all(
            {
                "type": "dependency_progress",
                "dependency": dependency,
                "progress": progress,
                "status": status,
                "message": message,
                "engine": engine,
                "model_id": model_id,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

    async def send_dependency_complete(
        self, dependency: str, success: bool, error: str = None, engine: str = None, model_id: str = None
    ):
        """Send dependency installation completion notification."""
        await self.send_to_all(
            {
                "type": "dependency_complete",
                "dependency": dependency,
                "success": success,
                "error": error,
                "engine": engine,
                "model_id": model_id,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

    def is_downloading(self, engine: str, model_id: str) -> bool:
        """Check if a model is currently being downloaded."""
        task_key = f"{engine}:{model_id}"
        return (
            task_key in self.download_tasks and not self.download_tasks[task_key].done()
        )

    async def start_download_task(self, engine: str, model_id: str):
        """Start an async model download task."""
        task_key = f"{engine}:{model_id}"
        logger.info(f"WebSocketManager.start_download_task called for {task_key}")

        if self.is_downloading(engine, model_id):
            logger.warning(f"Download already in progress for {task_key}")
            await self.send_download_progress(
                engine, model_id, 0, "error", "Download already in progress"
            )
            return

        # Create and start the download task
        logger.info(f"Creating download task for {task_key}")
        task = asyncio.create_task(self._download_model_async(engine, model_id))
        self.download_tasks[task_key] = task
        logger.info(f"Download task created and stored for {task_key}")

        try:
            await task
            logger.info(f"Download task completed for {task_key}")
        except Exception as e:
            logger.error(f"Download task failed for {task_key}: {e}")
        finally:
            # Clean up completed task
            if task_key in self.download_tasks:
                del self.download_tasks[task_key]
                logger.info(f"Cleaned up download task for {task_key}")

    async def _download_model_async(self, engine: str, model_id: str):
        """Async model download with progress updates."""
        try:
            logger.info(f"Starting download for {engine}/{model_id}")
            await self.send_download_progress(
                engine, model_id, 0, "starting", "Initializing download..."
            )

            # Check if model is already cached
            if model_manager.is_model_cached(engine, model_id):
                logger.info(f"Model {engine}/{model_id} already cached")
                await self.send_download_complete(engine, model_id, True)
                return

            await self.send_download_progress(
                engine, model_id, 20, "downloading", "Downloading model files..."
            )

            # Check if dependencies need to be installed first
            dependency_needed = None
            if engine == "voxtral" and not _check_voxtral_support():
                dependency_needed = "voxtral"
            elif engine == "nvidia" and not _check_nemo_support():
                dependency_needed = "nvidia"

            if dependency_needed:
                await self.send_download_progress(
                    engine, model_id, 10, "preparing", f"Installing {dependency_needed} dependencies..."
                )
                
                try:
                    await install_deps_with_websocket(dependency_needed, engine, model_id)
                except Exception as e:
                    logger.error(f"Dependency installation failed: {e}")
                    raise

            # Create a wrapper function that provides progress updates
            def download_with_progress():
                try:
                    logger.info(f"Loading model {engine}/{model_id}")
                    return load_model(engine, model_id)
                except Exception as e:
                    logger.error(f"Model loading failed: {e}")
                    raise

            # Load model (this will download if not cached)
            await self.send_download_progress(
                engine, model_id, 50, "downloading", "Loading model components..."
            )

            model_result = await asyncio.get_event_loop().run_in_executor(
                None, download_with_progress
            )

            await self.send_download_progress(
                engine, model_id, 90, "finalizing", "Finalizing download..."
            )

            # Verify model is now cached
            if model_manager.is_model_cached(engine, model_id):
                logger.info(
                    f"Model {engine}/{model_id} download completed successfully"
                )
                await self.send_download_progress(
                    engine, model_id, 100, "complete", "Download completed successfully"
                )
                await self.send_download_complete(engine, model_id, True)
            else:
                raise Exception("Model download completed but not found in cache")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Model download failed for {engine}/{model_id}: {error_msg}")
            await self.send_download_progress(
                engine, model_id, 0, "error", f"Download failed: {error_msg}"
            )
            await self.send_download_complete(engine, model_id, False, error_msg)


# Initialize WebSocket manager
websocket_manager = WebSocketManager()


class TranscriptionLogger:
    """Specialized logger for tracking transcription activities."""

    def __init__(self):
        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)
        self.transcription_log = self.logs_dir / "transcriptions.jsonl"
        self.model_usage_log = self.logs_dir / "model_usage.jsonl"

    def log_transcription_start(
        self, engine: str, model_id: str, filename: str, file_size: int
    ):
        """Log the start of a transcription."""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "transcription_start",
            "engine": engine,
            "model_id": model_id,
            "filename": filename,
            "file_size_bytes": file_size,
            "device": DEVICE,
        }

        self._write_log(self.transcription_log, log_entry)
        logger.info(
            f"Starting transcription: {engine}/{model_id} for {filename} ({file_size} bytes)"
        )

    def log_transcription_complete(
        self,
        engine: str,
        model_id: str,
        filename: str,
        processing_time: float,
        duration: float,
        success: bool,
        error: str = None,
    ):
        """Log the completion of a transcription."""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "transcription_complete",
            "engine": engine,
            "model_id": model_id,
            "filename": filename,
            "processing_time_seconds": processing_time,
            "audio_duration_seconds": duration,
            "success": success,
            "error": error,
            "device": DEVICE,
        }

        self._write_log(self.transcription_log, log_entry)

        if success:
            logger.info(
                f"Transcription completed: {engine}/{model_id} - {processing_time:.2f}s processing, {duration:.2f}s audio"
            )
        else:
            logger.error(f"Transcription failed: {engine}/{model_id} - {error}")

    def log_model_load_start(self, engine: str, model_id: str, cached: bool):
        """Log the start of model loading."""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "model_load_start",
            "engine": engine,
            "model_id": model_id,
            "was_cached": cached,
            "device": DEVICE,
        }

        self._write_log(self.model_usage_log, log_entry)
        cache_status = "cached" if cached else "downloading"
        logger.info(f"Loading model: {engine}/{model_id} ({cache_status})")

    def log_model_load_complete(
        self,
        engine: str,
        model_id: str,
        load_time: float,
        success: bool,
        error: str = None,
    ):
        """Log the completion of model loading."""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "model_load_complete",
            "engine": engine,
            "model_id": model_id,
            "load_time_seconds": load_time,
            "success": success,
            "error": error,
            "device": DEVICE,
        }

        self._write_log(self.model_usage_log, log_entry)

        if success:
            logger.info(
                f"Model loaded successfully: {engine}/{model_id} in {load_time:.2f}s"
            )
        else:
            logger.error(f"Model load failed: {engine}/{model_id} - {error}")

    def log_dependency_error(
        self, engine: str, model_id: str, dependency: str, error: str
    ):
        """Log dependency-related errors."""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "dependency_error",
            "engine": engine,
            "model_id": model_id,
            "dependency": dependency,
            "error": error,
            "device": DEVICE,
        }

        self._write_log(self.transcription_log, log_entry)
        logger.error(
            f"Dependency error for {engine}/{model_id}: {dependency} - {error}"
        )

    def log_dependency_install_start(self, dependency: str):
        """Log the start of dependency installation."""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "dependency_install_start",
            "dependency": dependency,
            "device": DEVICE,
        }

        self._write_log(self.transcription_log, log_entry)
        logger.info(f"Starting installation of dependency: {dependency}")

    def log_dependency_install_complete(
        self,
        dependency: str,
        success: bool,
        error: str = None,
        install_time: float = None,
    ):
        """Log the completion of dependency installation."""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "dependency_install_complete",
            "dependency": dependency,
            "success": success,
            "error": error,
            "install_time_seconds": install_time,
            "device": DEVICE,
        }

        self._write_log(self.transcription_log, log_entry)

        if success:
            logger.info(
                f"Dependency installation completed: {dependency} in {install_time:.2f}s"
            )
        else:
            logger.error(f"Dependency installation failed: {dependency} - {error}")

    def log_comparison_start(self, models: List[Dict], filename: str, file_size: int):
        """Log the start of a model comparison."""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "comparison_start",
            "models": models,
            "filename": filename,
            "file_size_bytes": file_size,
            "model_count": len(models),
            "device": DEVICE,
        }

        self._write_log(self.transcription_log, log_entry)
        model_list = ", ".join([f"{m['engine']}/{m['model_id']}" for m in models])
        logger.info(
            f"Starting comparison of {len(models)} models: {model_list} for {filename}"
        )

    def log_comparison_complete(
        self, models: List[Dict], filename: str, total_time: float, results: Dict
    ):
        """Log the completion of a model comparison."""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "comparison_complete",
            "models": models,
            "filename": filename,
            "total_time_seconds": total_time,
            "results_summary": {
                key: {
                    "success": result.get("success", False),
                    "processing_time": result.get("processing_time", 0),
                }
                for key, result in results.items()
            },
            "device": DEVICE,
        }

        self._write_log(self.transcription_log, log_entry)
        successful_models = sum(1 for r in results.values() if r.get("success", False))
        logger.info(
            f"Comparison completed: {successful_models}/{len(models)} models successful in {total_time:.2f}s"
        )

    def log_model_delete(self, engine: str, model_id: str, success: bool, error: str = None):
        """Log model cache deletion."""
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "event": "model_delete",
            "engine": engine,
            "model_id": model_id,
            "success": success,
            "error": error,
            "device": DEVICE,
        }

        self._write_log(self.transcription_log, log_entry)

        if success:
            logger.info(f"Model cache deleted successfully: {engine}/{model_id}")
        else:
            logger.error(f"Model cache deletion failed: {engine}/{model_id} - {error}")

    def _write_log(self, log_file: Path, log_entry: Dict):
        """Write a log entry to a JSONL file."""
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to log file {log_file}: {e}")


# Initialize transcription logger
transcription_logger = TranscriptionLogger()


class TranscriptionManager:
    """Manages transcription storage and retrieval."""

    def __init__(self):
        self.transcriptions_dir = TRANSCRIPTIONS_DIR
        self.metadata_file = self.transcriptions_dir / "transcriptions_metadata.json"
        self.load_metadata()

    def load_metadata(self):
        """Load transcription metadata from file."""
        if self.metadata_file.exists():
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"transcriptions": []}

    def save_metadata(self):
        """Save transcription metadata to file."""
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

    def generate_transcription_id(self):
        """Generate a unique transcription ID."""
        import uuid
        return str(uuid.uuid4())

    def save_transcription(
        self,
        transcription_id: str,
        engine: str,
        model_id: str,
        audio_filename: str,
        audio_duration: float,
        transcription_duration: float,
        csv_data: List[List[str]],
        success: bool = True,
        error: str = None,
    ):
        """Save transcription results and metadata."""
        try:
            # Create transcription entry
            transcription_entry = {
                "id": transcription_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "engine": engine,
                "model_id": model_id,
                "model_display_name": model_manager.get_display_name(engine, model_id),
                "audio_filename": audio_filename,
                "audio_duration_seconds": audio_duration,
                "transcription_duration_seconds": transcription_duration,
                "success": success,
                "error": error,
                "device": DEVICE,
            }

            if success and csv_data:
                # Save CSV data to file
                csv_filename = f"{transcription_id}.json"
                csv_filepath = self.transcriptions_dir / csv_filename
                
                with open(csv_filepath, "w", encoding="utf-8") as f:
                    json.dump({"csv_data": csv_data}, f, indent=2, ensure_ascii=False)
                
                transcription_entry["csv_file"] = csv_filename
                transcription_entry["segments_count"] = len(csv_data) - 1  # Subtract header row

            # Add to metadata
            self.metadata["transcriptions"].insert(0, transcription_entry)  # Most recent first
            
            # Keep only last 100 transcriptions to prevent file from growing too large
            if len(self.metadata["transcriptions"]) > 100:
                # Remove old transcription files
                for old_entry in self.metadata["transcriptions"][100:]:
                    if old_entry.get("csv_file"):
                        old_csv_path = self.transcriptions_dir / old_entry["csv_file"]
                        if old_csv_path.exists():
                            old_csv_path.unlink()
                
                self.metadata["transcriptions"] = self.metadata["transcriptions"][:100]

            self.save_metadata()
            logger.info(f"Saved transcription {transcription_id} to storage")
            return transcription_entry

        except Exception as e:
            logger.error(f"Failed to save transcription {transcription_id}: {e}")
            return None

    def get_transcription(self, transcription_id: str):
        """Get a specific transcription by ID."""
        try:
            # Find transcription in metadata
            transcription = None
            for entry in self.metadata["transcriptions"]:
                if entry["id"] == transcription_id:
                    transcription = entry.copy()
                    break

            if not transcription:
                return None

            # Load CSV data if available
            if transcription.get("csv_file"):
                csv_filepath = self.transcriptions_dir / transcription["csv_file"]
                if csv_filepath.exists():
                    with open(csv_filepath, "r", encoding="utf-8") as f:
                        csv_content = json.load(f)
                        transcription["csv_data"] = csv_content.get("csv_data", [])

            return transcription

        except Exception as e:
            logger.error(f"Failed to get transcription {transcription_id}: {e}")
            return None

    def get_transcriptions_list(self, limit: int = 50, offset: int = 0):
        """Get list of transcriptions with pagination."""
        try:
            total = len(self.metadata["transcriptions"])
            transcriptions = self.metadata["transcriptions"][offset:offset + limit]
            
            # Return metadata without CSV data for list view
            result = []
            for entry in transcriptions:
                list_entry = entry.copy()
                # Remove CSV file reference from list view
                list_entry.pop("csv_file", None)
                result.append(list_entry)

            return {
                "transcriptions": result,
                "total": total,
                "limit": limit,
                "offset": offset,
                "has_more": offset + limit < total
            }

        except Exception as e:
            logger.error(f"Failed to get transcriptions list: {e}")
            return {"transcriptions": [], "total": 0, "limit": limit, "offset": offset, "has_more": False}

    def delete_transcription(self, transcription_id: str):
        """Delete a transcription and its files."""
        try:
            # Find and remove from metadata
            transcription = None
            for i, entry in enumerate(self.metadata["transcriptions"]):
                if entry["id"] == transcription_id:
                    transcription = self.metadata["transcriptions"].pop(i)
                    break

            if not transcription:
                return False

            # Delete CSV file if exists
            if transcription.get("csv_file"):
                csv_filepath = self.transcriptions_dir / transcription["csv_file"]
                if csv_filepath.exists():
                    csv_filepath.unlink()

            self.save_metadata()
            logger.info(f"Deleted transcription {transcription_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete transcription {transcription_id}: {e}")
            return False


# Initialize transcription manager
transcription_manager = TranscriptionManager()

# Model configurations
MODEL_REGISTRY = {
    "whisper": {
        "tiny": {"size": "39MB", "cache_dir": BASE_MODELS_DIR / "whisper"},
        "base": {"size": "142MB", "cache_dir": BASE_MODELS_DIR / "whisper"},
        "small": {"size": "461MB", "cache_dir": BASE_MODELS_DIR / "whisper"},
        "medium": {"size": "1.5GB", "cache_dir": BASE_MODELS_DIR / "whisper"},
        "large": {"size": "2.9GB", "cache_dir": BASE_MODELS_DIR / "whisper"},
        "large-v2": {"size": "2.9GB", "cache_dir": BASE_MODELS_DIR / "whisper"},
        "large-v3": {"size": "2.9GB", "cache_dir": BASE_MODELS_DIR / "whisper"},
    },
    "voxtral": {
        "mistralai/Voxtral-Mini-3B-2507": {
            "size": "6GB",
            "cache_dir": BASE_MODELS_DIR / "voxtral" / "mini-3b",
            "display_name": "Voxtral Mini 3B",
        },
        "mistralai/Voxtral-Small-24B-2507": {
            "size": "48GB",
            "cache_dir": BASE_MODELS_DIR / "voxtral" / "small-24b",
            "display_name": "Voxtral Small 24B",
        },
    },
    "nvidia": {
        "nvidia/parakeet-tdt-0.6b-v2": {
            "size": "2.4GB",
            "cache_dir": BASE_MODELS_DIR / "nvidia" / "parakeet-tdt-0.6b-v2",
            "display_name": "Parakeet TDT 0.6B V2",
        },
        "nvidia/canary-qwen-2.5b": {
            "size": "5GB",
            "cache_dir": BASE_MODELS_DIR / "nvidia" / "canary-qwen-2.5b",
            "display_name": "Canary-Qwen 2.5B",
        },
    },
}

# Ensure all model directories exist
for engine_models in MODEL_REGISTRY.values():
    for model_config in engine_models.values():
        if isinstance(model_config, dict) and "cache_dir" in model_config:
            model_config["cache_dir"].mkdir(parents=True, exist_ok=True)


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


def _clear_module_cache_and_refresh(modules_to_clear=None):
    """Clear module cache and refresh imports after installation."""
    import importlib

    if modules_to_clear is None:
        modules_to_clear = ["transformers"]

    # Remove specified modules from cache if they exist
    modules_to_remove = [
        name
        for name in sys.modules.keys()
        if any(name.startswith(prefix) for prefix in modules_to_clear)
    ]
    for module_name in modules_to_remove:
        del sys.modules[module_name]

    # Invalidate import caches
    importlib.invalidate_caches()
    logger.info(f"Cleared module cache for: {', '.join(modules_to_clear)}")


def _check_voxtral_support():
    """Internal function to check Voxtral support."""
    try:
        import transformers

        version = transformers.__version__
        supported = version >= "4.56.0"

        if not supported:
            logger.warning(
                f"Voxtral not supported: transformers version {version} < 4.56.0"
            )
        else:
            logger.info(f"Voxtral supported: transformers version {version}")

        return supported
    except ImportError as e:
        logger.warning(f"Voxtral not supported: transformers not installed - {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking Voxtral support: {e}")
        return False


def _check_nemo_support():
    """Internal function to check NeMo support."""
    try:
        # If we've already successfully imported NeMo, don't try again
        import transformers

        version = transformers.__version__
        supported = version == "4.51.0"

        if supported:
            logger.info(f"NeMo toolkit supported: Transformers=={version} loaded")
        else:
            logger.error(f"NeMo toolkit not supported: Transformers=={version} loaded")
            return False

        # Try to import NeMo components
        import nemo.collections.asr as nemo_asr
        from nemo.collections.speechlm2.models import SALM

        logger.info("NeMo toolkit detected and supported")
        return True
    except ImportError as e:
        logger.warning(f"NeMo not supported: NeMo toolkit not installed - {e}")
        return False
    except Exception as e:
        # Handle resolver registration errors specifically
        if "resolver" in str(e) and "already registered" in str(e):
            logger.info(
                "NeMo toolkit supported (resolver already registered from previous import)"
            )
            return True
        else:
            logger.error(f"Error checking NeMo support: {e}")
            return False


def get_transformers_version():
    """Get transformers version if available."""
    try:
        import transformers

        return transformers.__version__
    except ImportError:
        return None


# Model management
class ModelManager:
    """Unified model management system for all STT engines."""

    def __init__(self):
        self.registry = MODEL_REGISTRY
        self.cache_info_file = BASE_MODELS_DIR / "cache_info.json"
        self.load_cache_info()

    def load_cache_info(self):
        """Load cached model information."""
        if self.cache_info_file.exists():
            with open(self.cache_info_file, "r") as f:
                self.cache_info = json.load(f)
        else:
            self.cache_info = {}

    def save_cache_info(self):
        """Save cached model information."""
        BASE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.cache_info_file, "w") as f:
            json.dump(self.cache_info, f, indent=2)

    def is_model_cached(self, engine, model_id):
        """Check if a model is cached."""
        cache_key = f"{engine}:{model_id}"

        # Check physical presence first
        if engine == "whisper":
            # Whisper models are stored as .pt files in the cache directory
            cache_dir = self.registry[engine][model_id]["cache_dir"]
            model_file = cache_dir / f"{model_id}.pt"
            is_cached = model_file.exists()

            # If cached but not in our cache_info, add it
            if is_cached and cache_key not in self.cache_info:
                self.mark_model_cached(engine, model_id, cache_dir)

            return is_cached
        elif engine == "nvidia":
            # For NeMo models, check both our cache and huggingface cache
            cache_dir = self.registry[engine][model_id]["cache_dir"]

            # Check our designated cache directory
            is_cached_local = cache_dir.exists() and any(cache_dir.iterdir())

            # Check huggingface cache structure
            hf_cache_dir = BASE_MODELS_DIR / "huggingface" / "hub"
            model_name_safe = model_id.replace("/", "--")
            hf_model_dir = hf_cache_dir / f"models--{model_name_safe}"
            is_cached_hf = hf_model_dir.exists() and any(hf_model_dir.rglob("*.nemo"))

            is_cached = is_cached_local or is_cached_hf

            # If cached but not in our cache_info, add it
            if is_cached and cache_key not in self.cache_info:
                self.mark_model_cached(engine, model_id, cache_dir)

            return is_cached
        else:
            # For other engines, check cache_info first, then physical presence
            if cache_key in self.cache_info:
                cache_dir = Path(self.cache_info[cache_key]["cache_path"])
                return cache_dir.exists() and any(cache_dir.iterdir())

            # Fallback to physical check
            cache_dir = self.registry[engine][model_id]["cache_dir"]
            is_cached = cache_dir.exists() and any(cache_dir.iterdir())

            # If cached but not in our cache_info, add it
            if is_cached:
                self.mark_model_cached(engine, model_id, cache_dir)

            return is_cached

    def get_model_size(self, engine, model_id):
        """Get model size information."""
        return self.registry[engine][model_id]["size"]

    def get_cache_dir(self, engine, model_id):
        """Get cache directory for a model."""
        return self.registry[engine][model_id]["cache_dir"]

    def mark_model_cached(self, engine, model_id, cache_path):
        """Mark a model as cached."""
        cache_key = f"{engine}:{model_id}"
        self.cache_info[cache_key] = {
            "engine": engine,
            "model_id": model_id,
            "cache_path": str(cache_path),
            "cached_at": datetime.datetime.now().isoformat(),
            "size": self.get_model_size(engine, model_id),
        }
        self.save_cache_info()

    def get_display_name(self, engine, model_id):
        """Get display name for a model."""
        model_config = self.registry[engine][model_id]
        if "display_name" in model_config:
            return model_config["display_name"]
        return model_id

    def delete_model_cache(self, engine, model_id):
        """Delete a cached model and remove from cache info."""
        cache_key = f"{engine}:{model_id}"
        
        try:
            # Remove from cache info first
            if cache_key in self.cache_info:
                del self.cache_info[cache_key]
                self.save_cache_info()
            
            # Delete physical files
            if engine == "whisper":
                cache_dir = self.registry[engine][model_id]["cache_dir"]
                model_file = cache_dir / f"{model_id}.pt"
                if model_file.exists():
                    model_file.unlink()
                    logger.info(f"Deleted Whisper model file: {model_file}")
                
                # Also check for any other files in the cache directory
                if cache_dir.exists():
                    for file in cache_dir.glob(f"{model_id}*"):
                        file.unlink()
                        logger.info(f"Deleted additional file: {file}")
                        
            elif engine == "nvidia":
                # For NeMo models, delete both local cache and huggingface cache
                cache_dir = self.registry[engine][model_id]["cache_dir"]
                
                # Delete local cache directory
                if cache_dir.exists():
                    import shutil
                    shutil.rmtree(cache_dir)
                    logger.info(f"Deleted NeMo local cache directory: {cache_dir}")
                
                # Delete from huggingface cache
                hf_cache_dir = BASE_MODELS_DIR / "huggingface" / "hub"
                model_name_safe = model_id.replace("/", "--")
                hf_model_dir = hf_cache_dir / f"models--{model_name_safe}"
                if hf_model_dir.exists():
                    import shutil
                    shutil.rmtree(hf_model_dir)
                    logger.info(f"Deleted HuggingFace cache directory: {hf_model_dir}")
                    
            else:
                # For other engines (like voxtral), delete the cache directory
                cache_dir = self.registry[engine][model_id]["cache_dir"]
                if cache_dir.exists():
                    import shutil
                    shutil.rmtree(cache_dir)
                    logger.info(f"Deleted cache directory: {cache_dir}")
            
            logger.info(f"Successfully deleted model cache for {engine}/{model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting model cache for {engine}/{model_id}: {e}")
            return False

    def get_cache_size(self, engine, model_id):
        """Get the actual disk size of a cached model."""
        try:
            cache_dir = self.get_cache_dir(engine, model_id)
            
            if engine == "whisper":
                model_file = cache_dir / f"{model_id}.pt"
                if model_file.exists():
                    return model_file.stat().st_size
            elif engine == "nvidia":
                # Check both local and HF cache
                total_size = 0
                if cache_dir.exists():
                    total_size += sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
                
                # Check HF cache
                hf_cache_dir = BASE_MODELS_DIR / "huggingface" / "hub"
                model_name_safe = model_id.replace("/", "--")
                hf_model_dir = hf_cache_dir / f"models--{model_name_safe}"
                if hf_model_dir.exists():
                    total_size += sum(f.stat().st_size for f in hf_model_dir.rglob('*') if f.is_file())
                
                return total_size
            else:
                if cache_dir.exists():
                    return sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file())
            
            return 0
        except Exception as e:
            logger.error(f"Error getting cache size for {engine}/{model_id}: {e}")
            return 0

    def scan_existing_models(self):
        """Scan for existing cached models and update cache info."""
        for engine, engine_models in self.registry.items():
            for model_id, config in engine_models.items():
                cache_key = f"{engine}:{model_id}"

                # Skip if already tracked
                if cache_key in self.cache_info:
                    continue

                # Check if model exists physically
                if engine == "whisper":
                    cache_dir = config["cache_dir"]
                    model_file = cache_dir / f"{model_id}.pt"
                    if model_file.exists():
                        self.mark_model_cached(engine, model_id, cache_dir)
                else:
                    cache_dir = config["cache_dir"]
                    if cache_dir.exists() and any(cache_dir.iterdir()):
                        self.mark_model_cached(engine, model_id, cache_dir)


# Initialize model manager
model_manager = ModelManager()
model_manager.scan_existing_models()


# Model loading and transcription functions
async def load_model_async(engine, model_id):
    """Async wrapper for load_model with WebSocket notifications."""
    return await asyncio.get_event_loop().run_in_executor(
        None, load_model, engine, model_id
    )


def load_model(engine, model_id):
    """Unified model loading for all STT engines."""
    load_start_time = time.time()

    try:
        cache_dir = model_manager.get_cache_dir(engine, model_id)
        is_cached = model_manager.is_model_cached(engine, model_id)

        # Log model load start
        transcription_logger.log_model_load_start(engine, model_id, is_cached)

        if engine == "whisper":
            model = whisper.load_model(
                model_id, download_root=str(cache_dir), device=DEVICE
            )
            result = model

            # Ensure Whisper model is marked as cached after loading
            if not is_cached:
                model_manager.mark_model_cached(engine, model_id, cache_dir)

        elif engine == "voxtral":
            if not _check_voxtral_support():
                try:
                    # Run dependency installation with WebSocket notifications
                    asyncio.create_task(
                        install_deps_with_websocket(engine, engine, model_id)
                    )
                    # Synchronous fallback for now
                    install_deps(engine)
                except Exception as e:
                    error_msg = f"Voxtral requires transformers 4.56.0+. {e}"
                    transcription_logger.log_dependency_error(
                        engine, model_id, "transformers", error_msg
                    )
                    raise Exception(error_msg)

            if _check_voxtral_support():
                from transformers import VoxtralForConditionalGeneration, AutoProcessor

                processor = AutoProcessor.from_pretrained(
                    model_id, cache_dir=str(cache_dir)
                )
                model = VoxtralForConditionalGeneration.from_pretrained(
                    model_id,
                    cache_dir=str(cache_dir),
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                result = (model, processor)
            else:
                error_msg = f"Voxtral transformers library not loaded successfully."
                transcription_logger.log_dependency_error(
                    engine, model_id, "transformers", error_msg
                )
                raise Exception(error_msg)

        elif engine == "nvidia":
            if not _check_nemo_support():
                try:
                    logger.info("Nemo toolkit needs transformers<4.52.0 and >=4.51.0")
                    # Run dependency installation with WebSocket notifications
                    asyncio.create_task(
                        install_deps_with_websocket(engine, engine, model_id)
                    )
                    # Synchronous fallback for now
                    install_deps(engine)
                except Exception as e:
                    error_msg = "NeMo toolkit required"
                    transcription_logger.log_dependency_error(
                        engine, model_id, "nemo", error_msg
                    )
                    raise Exception(error_msg)

            if _check_nemo_support():
                import nemo.collections.asr as nemo_asr
                from nemo.collections.speechlm2.models import SALM

                # Set multiple cache environment variables to ensure models are stored in our cache
                os.environ["NEMO_CACHE_DIR"] = str(cache_dir)
                os.environ["HF_HOME"] = str(cache_dir / "huggingface")
                os.environ["HUGGINGFACE_HUB_CACHE"] = str(
                    cache_dir / "huggingface" / "hub"
                )
                os.environ["TRANSFORMERS_CACHE"] = str(
                    cache_dir / "huggingface" / "transformers"
                )

                # Ensure the huggingface cache directory exists
                (cache_dir / "huggingface" / "hub").mkdir(parents=True, exist_ok=True)
                (cache_dir / "huggingface" / "transformers").mkdir(
                    parents=True, exist_ok=True
                )

                if "parakeet" in model_id:
                    model = nemo_asr.models.ASRModel.from_pretrained(
                        model_name=model_id
                    )
                elif "canary" in model_id:
                    model = SALM.from_pretrained(model_id)

                result = model
            else:
                error_msg = f"Nemo transformers library not loaded successfully."
                transcription_logger.log_dependency_error(
                    engine, model_id, "transformers", error_msg
                )
                raise Exception(error_msg)

        else:
            raise Exception(f"Unknown engine: {engine}")

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
        transcription_logger.log_model_load_complete(
            engine, model_id, load_time, False, error_msg
        )
        raise Exception(f"Error loading {engine} model: {error_msg}")


def process_audio(audio_path):
    """Process audio file."""
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_sec = audio.duration_seconds

        # Convert to mono and resample
        if audio.channels > 1:
            audio = audio.set_channels(1)

        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)

        # Save processed audio
        temp_dir = tempfile.gettempdir()
        processed_path = os.path.join(temp_dir, "processed_audio.wav")
        audio.export(processed_path, format="wav")

        return processed_path, duration_sec
    except Exception as e:
        raise Exception(f"Error processing audio: {str(e)}")


def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    return str(datetime.timedelta(seconds=seconds)).split(".")[0]


def transcribe_audio(engine, audio_path, model_id, filename=None, file_size=None, save_to_history=True):
    """Unified transcription method for all STT engines."""
    total_start_time = time.time()

    # Extract filename if not provided
    if filename is None:
        filename = Path(audio_path).name

    # Get file size if not provided
    if file_size is None:
        try:
            file_size = Path(audio_path).stat().st_size
        except:
            file_size = 0

    # Generate transcription ID for history
    transcription_id = transcription_manager.generate_transcription_id() if save_to_history else None

    # Log transcription start
    transcription_logger.log_transcription_start(engine, model_id, filename, file_size)

    try:
        # Load the appropriate model (this time is not counted in transcription duration)
        model_load_start = time.time()
        model_result = load_model(engine, model_id)
        if model_result is None:
            raise Exception("Failed to load model")
        model_load_time = time.time() - model_load_start

        # Process audio (this time is not counted in transcription duration)
        audio_process_start = time.time()
        processed_path, duration_sec = process_audio(audio_path)
        if processed_path is None:
            raise Exception("Failed to process audio")
        audio_process_time = time.time() - audio_process_start

        try:
            # Start measuring actual transcription time
            transcription_start_time = time.time()
            
            if engine == "whisper":
                model = model_result
                result = model.transcribe(processed_path, word_timestamps=True)

                if not result or "segments" not in result:
                    raise Exception("Whisper transcription failed")

                # Generate CSV with timestamps
                csv_data = [
                    [
                        "From (s)",
                        "To (s)",
                        "From (time)",
                        "To (time)",
                        "Duration",
                        "Transcription",
                    ]
                ]

                for segment in result["segments"]:
                    start_s = segment["start"]
                    end_s = segment["end"]
                    start_formatted = format_time(start_s)
                    end_formatted = format_time(end_s)
                    duration = end_s - start_s

                    csv_data.append(
                        [
                            f"{start_s:.2f}",
                            f"{end_s:.2f}",
                            start_formatted,
                            end_formatted,
                            f"{duration:.2f}",
                            segment["text"].strip(),
                        ]
                    )

            elif engine == "voxtral":
                model, processor = model_result
                inputs = processor.apply_transcription_request(
                    language="en", audio=processed_path, model_id=model_id
                )
                inputs = inputs.to(
                    "cuda" if torch.cuda.is_available() else "cpu",
                    dtype=torch.bfloat16,
                )

                outputs = model.generate(**inputs, max_new_tokens=500)
                decoded_outputs = processor.batch_decode(
                    outputs[:, inputs.input_ids.shape[1] :],
                    skip_special_tokens=True,
                )

                transcription_text = decoded_outputs[0]

                # Simple CSV format (no timestamps from Voxtral)
                csv_data = [["Start (s)", "End (s)", "Duration (s)", "Transcription"]]
                csv_data.append(
                    [
                        "0.00",
                        f"{duration_sec:.2f}",
                        f"{duration_sec:.2f}",
                        transcription_text.strip(),
                    ]
                )

            elif engine == "nvidia":
                model = model_result

                if "parakeet" in model_id:
                    output = model.transcribe([processed_path], timestamps=True)

                    if not output or len(output) == 0:
                        raise Exception("Parakeet transcription failed")

                    # Generate CSV with timestamps if available
                    csv_data = [
                        [
                            "From (s)",
                            "To (s)",
                            "From (time)",
                            "To (time)",
                            "Duration",
                            "Transcription",
                        ]
                    ]

                    result = output[0]

                    # Check if timestamps are available
                    if hasattr(result, "timestamp") and result.timestamp:
                        segment_timestamps = result.timestamp.get("segment", [])

                        for stamp in segment_timestamps:
                            start_s = stamp["start"]
                            end_s = stamp["end"]
                            start_formatted = format_time(start_s)
                            end_formatted = format_time(end_s)
                            duration = end_s - start_s

                            csv_data.append(
                                [
                                    f"{start_s:.2f}",
                                    f"{end_s:.2f}",
                                    start_formatted,
                                    end_formatted,
                                    f"{duration:.2f}",
                                    stamp["segment"].strip(),
                                ]
                            )
                    else:
                        # Fallback to simple format without timestamps
                        csv_data.append(
                            [
                                "0.00",
                                f"{duration_sec:.2f}",
                                "00:00:00",
                                format_time(duration_sec),
                                f"{duration_sec:.2f}",
                                result.text.strip(),
                            ]
                        )

                elif "canary" in model_id:
                    prompts = [
                        [
                            {
                                "role": "user",
                                "content": f"Transcribe the following: {model.audio_locator_tag}",
                                "audio": [processed_path],
                            }
                        ]
                    ]

                    answer_ids = model.generate(
                        prompts=prompts,
                        max_new_tokens=128,
                    )

                    transcription_text = model.tokenizer.ids_to_text(
                        answer_ids[0].cpu()
                    )

                    if not transcription_text or transcription_text.strip() == "":
                        raise Exception("Canary-Qwen transcription failed")

                    # Simple CSV format (no detailed timestamps from Canary)
                    csv_data = [
                        ["Start (s)", "End (s)", "Duration (s)", "Transcription"]
                    ]
                    csv_data.append(
                        [
                            "0.00",
                            f"{duration_sec:.2f}",
                            f"{duration_sec:.2f}",
                            transcription_text.strip(),
                        ]
                    )

            else:
                raise Exception(f"Unknown engine: {engine}")

            # Calculate actual transcription time (excluding model load and audio processing)
            transcription_time = time.time() - transcription_start_time
            total_processing_time = time.time() - total_start_time

            # Log successful transcription
            transcription_logger.log_transcription_complete(
                engine, model_id, filename, total_processing_time, duration_sec, True
            )

            # Save to transcription history if requested
            if save_to_history and transcription_id:
                transcription_manager.save_transcription(
                    transcription_id=transcription_id,
                    engine=engine,
                    model_id=model_id,
                    audio_filename=filename,
                    audio_duration=duration_sec,
                    transcription_duration=transcription_time,
                    csv_data=csv_data,
                    success=True
                )

            return {
                "success": True,
                "csv_data": csv_data,
                "processing_time": total_processing_time,
                "transcription_time": transcription_time,
                "duration": duration_sec,
                "transcription_id": transcription_id,
            }

        finally:
            # Cleanup
            if os.path.exists(processed_path):
                os.remove(processed_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        total_processing_time = time.time() - total_start_time
        error_msg = str(e)

        # Log failed transcription
        transcription_logger.log_transcription_complete(
            engine, model_id, filename, total_processing_time, 0, False, error_msg
        )

        # Save failed transcription to history if requested
        if save_to_history and transcription_id:
            transcription_manager.save_transcription(
                transcription_id=transcription_id,
                engine=engine,
                model_id=model_id,
                audio_filename=filename,
                audio_duration=0,
                transcription_duration=0,
                csv_data=[],
                success=False,
                error=error_msg
            )

        return {
            "success": False, 
            "error": error_msg,
            "transcription_id": transcription_id
        }


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
                engine, audio_path, model_id, file.filename, len(content), save_to_history=True
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


class DependencyRequest(BaseModel):
    dependency: str

    class Config:
        schema_extra = {"example": {"dependency": "voxtral"}}


@app.post("/api/install-dependency")
async def install_dependency(request: DependencyRequest):
    """Install missing dependencies with WebSocket progress updates."""

    dependency = request.dependency
    print(f"Installing dependency: {dependency}")  # Debug log

    try:
        result = await install_deps_with_websocket(dependency)
        return result
    except Exception as e:
        # Fallback to synchronous installation
        logger.warning(f"WebSocket installation failed, falling back to sync: {e}")
        return install_deps(dependency)


@app.delete("/api/models/{engine}/{model_id}")
async def delete_model_cache(engine: str, model_id: str):
    """Delete a cached model."""
    
    # Validate engine and model
    if engine not in MODEL_REGISTRY or model_id not in MODEL_REGISTRY[engine]:
        raise HTTPException(status_code=400, detail="Invalid engine or model_id")
    
    # Check if model is currently being used (downloading)
    if websocket_manager.is_downloading(engine, model_id):
        raise HTTPException(
            status_code=400, 
            detail="Cannot delete model while it's being downloaded"
        )
    
    # Check if model is actually cached
    if not model_manager.is_model_cached(engine, model_id):
        raise HTTPException(
            status_code=404, 
            detail="Model is not cached"
        )
    
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
                "message": f"Model {engine}/{model_id} deleted successfully"
            }
        else:
            # Log failed deletion
            transcription_logger.log_model_delete(engine, model_id, False, "Failed to delete model cache")
            raise HTTPException(
                status_code=500, 
                detail="Failed to delete model cache"
            )
            
    except Exception as e:
        error_msg = str(e)
        # Log failed deletion
        transcription_logger.log_model_delete(engine, model_id, False, error_msg)
        logger.error(f"Error deleting model cache {engine}/{model_id}: {e}")
        raise HTTPException(status_code=500, detail=error_msg)


async def install_deps_with_websocket(dependency: str, engine: str = None, model_id: str = None):
    """Install dependencies with WebSocket progress updates."""
    # Validate dependency type first
    if dependency not in ["voxtral", "nvidia"]:
        error_msg = f"Unknown dependency: {dependency}. Supported: 'voxtral', 'nvidia'"
        transcription_logger.log_dependency_install_complete(
            dependency, False, error_msg
        )
        await websocket_manager.send_dependency_complete(
            dependency, False, error_msg, engine, model_id
        )
        raise HTTPException(status_code=400, detail=error_msg)

    # Log installation start
    transcription_logger.log_dependency_install_start(dependency)
    install_start_time = time.time()

    try:
        await websocket_manager.send_dependency_progress(
            dependency, 0, "preparing", "Preparing environment for dependency installation...", engine, model_id
        )

        if dependency == "voxtral":
            await websocket_manager.send_dependency_progress(
                dependency, 20, "installing", "Installing transformers 4.56.0+...", engine, model_id
            )
            
            logger.info("Installing transformers 4.56.0")
            command = [sys.executable, "-m", "pip", "install", "transformers>=4.56.0"]
            
            # Run pip install in executor to avoid blocking
            process = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: subprocess.run(command, capture_output=True, text=True, check=True)
            )

            await websocket_manager.send_dependency_progress(
                dependency, 80, "configuring", "Refreshing module cache...", engine, model_id
            )

            # Clear cache and refresh imports
            _clear_module_cache_and_refresh(["transformers"])

            message = "Transformers 4.56.0 installed successfully"

        elif dependency == "nvidia":
            await websocket_manager.send_dependency_progress(
                dependency, 20, "installing", "Installing transformers 4.51.0...", engine, model_id
            )
            
            logger.info("Installing transformers 4.51.0")
            command = [sys.executable, "-m", "pip", "install", "transformers==4.51.0"]
            
            # Run pip install in executor to avoid blocking
            process = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: subprocess.run(command, capture_output=True, text=True, check=True)
            )

            await websocket_manager.send_dependency_progress(
                dependency, 80, "configuring", "Refreshing module cache...", engine, model_id
            )

            # Clear cache and refresh imports
            _clear_module_cache_and_refresh(["transformers"])

            message = "Transformers 4.51.0 installed successfully"

        await websocket_manager.send_dependency_progress(
            dependency, 100, "complete", message, engine, model_id
        )

        install_time = time.time() - install_start_time
        transcription_logger.log_dependency_install_complete(
            dependency, True, None, install_time
        )

        await websocket_manager.send_dependency_complete(
            dependency, True, None, engine, model_id
        )

        return {"success": True, "message": message}

    except subprocess.CalledProcessError as e:
        install_time = time.time() - install_start_time
        error_msg = f"Installation failed: {e.stderr}"
        transcription_logger.log_dependency_install_complete(
            dependency, False, error_msg, install_time
        )
        
        await websocket_manager.send_dependency_progress(
            dependency, 0, "error", f"Installation failed: {error_msg}", engine, model_id
        )
        await websocket_manager.send_dependency_complete(
            dependency, False, error_msg, engine, model_id
        )
        
        raise HTTPException(status_code=500, detail=error_msg)
    except Exception as e:
        install_time = time.time() - install_start_time
        error_msg = f"Unexpected error: {str(e)}"
        transcription_logger.log_dependency_install_complete(
            dependency, False, error_msg, install_time
        )
        
        await websocket_manager.send_dependency_progress(
            dependency, 0, "error", error_msg, engine, model_id
        )
        await websocket_manager.send_dependency_complete(
            dependency, False, error_msg, engine, model_id
        )
        
        raise HTTPException(status_code=500, detail=error_msg)


def install_deps(dependency: str):
    """Synchronous dependency installation (fallback)."""
    # Validate dependency type first
    if dependency not in ["voxtral", "nvidia"]:
        error_msg = f"Unknown dependency: {dependency}. Supported: 'voxtral', 'nvidia'"
        transcription_logger.log_dependency_install_complete(
            dependency, False, error_msg
        )
        raise HTTPException(status_code=400, detail=error_msg)

    # Log installation start
    transcription_logger.log_dependency_install_start(dependency)
    install_start_time = time.time()

    try:
        if dependency == "voxtral":
            logger.info("Installing transformers 4.56.0")
            command = [sys.executable, "-m", "pip", "install", "transformers>=4.56.0"]
            subprocess.run(command, capture_output=True, text=True, check=True)

            # Clear cache and refresh imports
            _clear_module_cache_and_refresh(["transformers"])

            message = "Transformers 4.56.0 installed successfully"

        elif dependency == "nvidia":
            logger.info("Installing transformers 4.51.0")
            command = [sys.executable, "-m", "pip", "install", "transformers==4.51.0"]
            subprocess.run(command, capture_output=True, text=True, check=True)

            # Clear cache and refresh imports
            _clear_module_cache_and_refresh(["transformers"])

            message = "Transformers 4.51.0 installed successfully"

        install_time = time.time() - install_start_time
        transcription_logger.log_dependency_install_complete(
            dependency, True, None, install_time
        )

        return {"success": True, "message": message}

    except subprocess.CalledProcessError as e:
        install_time = time.time() - install_start_time
        error_msg = f"Installation failed: {e.stderr}"
        transcription_logger.log_dependency_install_complete(
            dependency, False, error_msg, install_time
        )
        raise HTTPException(status_code=500, detail=error_msg)


@app.get("/api/transcriptions")
async def get_transcriptions(limit: int = 50, offset: int = 0):
    """Get list of transcription history."""
    try:
        result = transcription_manager.get_transcriptions_list(limit=limit, offset=offset)
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
            raise HTTPException(status_code=404, detail="No transcription data available")

        # Convert to CSV format
        import io
        import csv
        
        output = io.StringIO()
        writer = csv.writer(output)
        
        for row in transcription["csv_data"]:
            writer.writerow(row)
        
        csv_content = output.getvalue()
        output.close()

        # Create filename
        timestamp = transcription["timestamp"][:10]  # YYYY-MM-DD
        filename = f"transcription_{timestamp}_{transcription['engine']}_{transcription['model_id']}.csv"
        
        from fastapi.responses import Response
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
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
    logger.info("SpeechHub Starting Up")
    logger.info("=" * 50)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Supported formats: {SUPPORTED_FORMATS}")
    logger.info(f"Base models directory: {BASE_MODELS_DIR}")
    logger.info(f"Voxtral support: {_check_voxtral_support()}")
    logger.info(f"NeMo support: {_check_nemo_support()}")
    logger.info("=" * 50)

    uvicorn.run(app, host="0.0.0.0", port=8000)
