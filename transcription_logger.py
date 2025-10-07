#!/usr/bin/env python3
"""
Transcription Logger for VoxScribe - Specialized logging for transcription activities
"""

import json
import datetime
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger("voxscribe")


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
        from config import DEVICE
        
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
        from config import DEVICE
        
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
        from config import DEVICE
        
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
        from config import DEVICE
        
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
        from config import DEVICE
        
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
        from config import DEVICE
        
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
        from config import DEVICE
        
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
        from config import DEVICE
        
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
        from config import DEVICE
        
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

    def log_model_delete(
        self, engine: str, model_id: str, success: bool, error: str = None
    ):
        """Log model cache deletion."""
        from config import DEVICE
        
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