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
from pathlib import Path
from typing import List, Dict, Optional, Any
import datetime
import gc

import torch
import whisper
from pydub import AudioSegment
import numpy as np
import pandas as pd

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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


# Dependency checking functions
def _check_voxtral_support():
    """Internal function to check Voxtral support."""
    try:
        from transformers import VoxtralForConditionalGeneration, AutoProcessor
        import transformers

        version = transformers.__version__
        return version >= "4.56.0"
    except ImportError:
        return False
    except Exception:
        return False


def _check_nemo_support():
    """Internal function to check NeMo support."""
    try:
        import nemo.collections.asr as nemo_asr
        from nemo.collections.speechlm2.models import SALM

        return True
    except ImportError:
        return False
    except Exception:
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
def load_model(engine, model_id):
    """Unified model loading for all STT engines."""
    try:
        cache_dir = model_manager.get_cache_dir(engine, model_id)
        is_cached = model_manager.is_model_cached(engine, model_id)

        if engine == "whisper":
            model = whisper.load_model(
                model_id, download_root=str(cache_dir.parent), device=DEVICE
            )
            result = model
            
            # Ensure Whisper model is marked as cached after loading
            if not is_cached:
                model_manager.mark_model_cached(engine, model_id, cache_dir)

        elif engine == "voxtral":
            if not _check_voxtral_support():
                raise Exception("Voxtral requires transformers 4.56.0+")

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

        elif engine == "nvidia":
            if not _check_nemo_support():
                raise Exception("NeMo toolkit required")

            import nemo.collections.asr as nemo_asr
            from nemo.collections.speechlm2.models import SALM

            os.environ["NEMO_CACHE_DIR"] = str(cache_dir)

            if "parakeet" in model_id:
                model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
            elif "canary" in model_id:
                model = SALM.from_pretrained(model_id)

            result = model

        else:
            raise Exception(f"Unknown engine: {engine}")

        # Mark as cached if it wasn't before
        if not is_cached:
            model_manager.mark_model_cached(engine, model_id, cache_dir)

        return result

    except Exception as e:
        raise Exception(f"Error loading {engine} model: {str(e)}")


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


def transcribe_audio(engine, audio_path, model_id):
    """Unified transcription method for all STT engines."""
    start_time = time.time()

    try:
        # Load the appropriate model
        model_result = load_model(engine, model_id)
        if model_result is None:
            raise Exception("Failed to load model")

        # Process audio
        processed_path, duration_sec = process_audio(audio_path)
        if processed_path is None:
            raise Exception("Failed to process audio")

        try:
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

                    transcription_text = model.tokenizer.ids_to_text(answer_ids[0].cpu())

                    if not transcription_text or transcription_text.strip() == "":
                        raise Exception("Canary-Qwen transcription failed")

                    # Simple CSV format (no detailed timestamps from Canary)
                    csv_data = [["Start (s)", "End (s)", "Duration (s)", "Transcription"]]
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

            processing_time = time.time() - start_time

            return {
                "success": True,
                "csv_data": csv_data,
                "processing_time": processing_time,
                "duration": duration_sec,
            }

        finally:
            # Cleanup
            if os.path.exists(processed_path):
                os.remove(processed_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        return {"success": False, "error": str(e)}


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


@app.get("/api/models")
async def get_models():
    """Get available models for all engines."""
    models = []

    for engine, engine_models in MODEL_REGISTRY.items():
        for model_id, config in engine_models.items():
            models.append(
                {
                    "engine": engine,
                    "model_id": model_id,
                    "display_name": model_manager.get_display_name(engine, model_id),
                    "size": config["size"],
                    "cached": model_manager.is_model_cached(engine, model_id),
                }
            )

    return {"models": models}


@app.post("/api/transcribe")
async def transcribe_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    engine: str = "whisper",
    model_id: str = "base",
):
    """Transcribe audio file with specified engine and model."""

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

        # Transcribe
        result = transcribe_audio(engine, audio_path, model_id)

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
    engines: str = "[]",  # JSON string of engine configs
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

        # Run comparisons
        results = {}
        for config in engine_configs:
            engine = config.get("engine")
            model_id = config.get("model_id")

            if not engine or not model_id:
                continue

            result = transcribe_audio(engine, audio_path, model_id)
            results[f"{engine}_{model_id}"] = result

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
    """Install missing dependencies."""

    dependency = request.dependency
    print(f"Installing dependency: {dependency}")  # Debug log

    if dependency == "voxtral":
        try:
            command = [sys.executable, "-m", "pip", "install", "transformers==4.56.0"]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            return {
                "success": True,
                "message": "Transformers 4.56.0 installed successfully",
            }
        except subprocess.CalledProcessError as e:
            raise HTTPException(
                status_code=500, detail=f"Installation failed: {e.stderr}"
            )

    elif dependency == "nemo":
        try:
            # Uninstall transformers first
            uninstall_cmd = [
                sys.executable,
                "-m",
                "pip",
                "uninstall",
                "-y",
                "transformers",
            ]
            subprocess.run(uninstall_cmd, capture_output=True, text=True, check=True)

            # Install NeMo toolkit
            package_name = (
                "nemo_toolkit[asr,tts] @ git+https://github.com/NVIDIA/NeMo.git"
            )
            install_cmd = [sys.executable, "-m", "pip", "install", package_name]
            result = subprocess.run(
                install_cmd, capture_output=True, text=True, check=True
            )

            return {"success": True, "message": "NeMo toolkit installed successfully"}
        except subprocess.CalledProcessError as e:
            raise HTTPException(
                status_code=500, detail=f"Installation failed: {e.stderr}"
            )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown dependency: {dependency}. Supported: 'voxtral', 'nemo'",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
