#!/usr/bin/env python3
"""
Model Manager for VoxScribe - Unified model management system for all STT engines
"""

import json
import datetime
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger("voxscribe")


class ModelManager:
    """Unified model management system for all STT engines."""

    def __init__(self, registry: Dict, base_models_dir: Path):
        self.registry = registry
        self.base_models_dir = base_models_dir
        self.cache_info_file = base_models_dir / "cache_info.json"
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
        self.base_models_dir.mkdir(parents=True, exist_ok=True)
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
            # For NeMo models, check huggingface cache (where they're actually stored)
            hf_cache_dir = self.base_models_dir / "huggingface" / "hub"
            model_name_safe = model_id.replace("/", "--")
            hf_model_dir = hf_cache_dir / f"models--{model_name_safe}"
            is_cached = hf_model_dir.exists() and any(hf_model_dir.rglob("*"))

            # If cached but not in our cache_info, add it
            if is_cached and cache_key not in self.cache_info:
                cache_dir = self.registry[engine][model_id]["cache_dir"]
                self.mark_model_cached(engine, model_id, cache_dir)

            return is_cached
        elif engine == "granite":
            # For Granite models, check both the marker file and huggingface cache
            cache_dir = self.registry[engine][model_id]["cache_dir"]
            marker_file = cache_dir / ".model_loaded"
            
            # Check if we have a marker file indicating successful loading
            if marker_file.exists():
                if cache_key not in self.cache_info:
                    self.mark_model_cached(engine, model_id, cache_dir)
                return True
            
            # Also check huggingface cache (where they're actually stored)
            hf_cache_dir = self.base_models_dir / "huggingface" / "hub"
            model_name_safe = model_id.replace("/", "--")
            hf_model_dir = hf_cache_dir / f"models--{model_name_safe}"
            is_cached = hf_model_dir.exists() and any(hf_model_dir.rglob("*"))

            # If cached but not in our cache_info, add it
            if is_cached and cache_key not in self.cache_info:
                self.mark_model_cached(engine, model_id, cache_dir)

            return is_cached
        else:
            # For other engines (like voxtral), check cache_info first, then physical presence
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
        import shutil

        cache_key = f"{engine}:{model_id}"
        deletion_success = True

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
                # For NeMo models, delete from huggingface cache (where they're actually stored)
                hf_cache_dir = self.base_models_dir / "huggingface" / "hub"
                model_name_safe = model_id.replace("/", "--")
                hf_model_dir = hf_cache_dir / f"models--{model_name_safe}"

                # Delete HuggingFace cache
                try:
                    if hf_model_dir.exists():
                        shutil.rmtree(hf_model_dir)
                        logger.info(
                            f"Deleted HuggingFace cache directory: {hf_model_dir}"
                        )
                except Exception as e:
                    logger.error(f"Error deleting HF cache for {model_id}: {e}")
                    deletion_success = False

                # Also clean up any files in the designated cache directory
                try:
                    cache_dir = self.registry[engine][model_id]["cache_dir"]
                    if cache_dir.exists():
                        # Check if directory has contents before attempting deletion
                        try:
                            if any(cache_dir.iterdir()):
                                shutil.rmtree(cache_dir)
                                logger.info(
                                    f"Deleted NeMo local cache directory: {cache_dir}"
                                )
                        except OSError:
                            # Directory might be empty or already deleted, recreate it
                            cache_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.error(f"Error deleting local cache for {model_id}: {e}")
                    deletion_success = False

            elif engine == "granite":
                # For Granite models, delete from huggingface cache (where they're actually stored)
                hf_cache_dir = self.base_models_dir / "huggingface" / "hub"
                model_name_safe = model_id.replace("/", "--")
                hf_model_dir = hf_cache_dir / f"models--{model_name_safe}"

                # Delete HuggingFace cache
                try:
                    if hf_model_dir.exists():
                        shutil.rmtree(hf_model_dir)
                        logger.info(
                            f"Deleted HuggingFace cache directory: {hf_model_dir}"
                        )
                except Exception as e:
                    logger.error(f"Error deleting HF cache for {model_id}: {e}")
                    deletion_success = False

                # Also clean up any files in the designated cache directory
                try:
                    cache_dir = self.registry[engine][model_id]["cache_dir"]
                    if cache_dir.exists():
                        # Check if directory has contents before attempting deletion
                        try:
                            if any(cache_dir.iterdir()):
                                shutil.rmtree(cache_dir)
                                logger.info(
                                    f"Deleted Granite local cache directory: {cache_dir}"
                                )
                        except OSError:
                            # Directory might be empty or already deleted, recreate it
                            cache_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    logger.error(f"Error deleting local cache for {model_id}: {e}")
                    deletion_success = False
            else:
                # For other engines (like voxtral), delete the cache directory
                cache_dir = self.registry[engine][model_id]["cache_dir"]
                if cache_dir.exists():
                    shutil.rmtree(cache_dir)
                    logger.info(f"Deleted cache directory: {cache_dir}")

            if deletion_success:
                logger.info(f"Successfully deleted model cache for {engine}/{model_id}")
            else:
                logger.warning(
                    f"Partially deleted model cache for {engine}/{model_id} (some errors occurred)"
                )

            return deletion_success

        except Exception as e:
            logger.error(f"Error deleting model cache for {engine}/{model_id}: {e}")
            return False

    def get_cache_size(self, engine, model_id):
        """Get the actual disk size of a cached model."""
        def _safe_get_dir_size(directory):
            """Safely calculate directory size with error handling."""
            total_size = 0
            try:
                for file_path in directory.rglob("*"):
                    try:
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
                    except (OSError, PermissionError):
                        # Skip files that can't be accessed (e.g., being written to)
                        continue
            except (OSError, PermissionError):
                # Skip directories that can't be accessed
                pass
            return total_size

        try:
            # First check if the model is actually cached
            if not self.is_model_cached(engine, model_id):
                return 0

            cache_dir = self.get_cache_dir(engine, model_id)

            if engine == "whisper":
                model_file = cache_dir / f"{model_id}.pt"
                if model_file.exists():
                    try:
                        return model_file.stat().st_size
                    except (OSError, PermissionError):
                        logger.warning(f"Could not access Whisper model file: {model_file}")
                        return 0
            elif engine == "nvidia":
                # Check HF cache (primary location) and local cache
                total_size = 0

                # Check HF cache first (where models are actually stored)
                hf_cache_dir = self.base_models_dir / "huggingface" / "hub"
                model_name_safe = model_id.replace("/", "--")
                hf_model_dir = hf_cache_dir / f"models--{model_name_safe}"
                if hf_model_dir.exists():
                    total_size += _safe_get_dir_size(hf_model_dir)

                # Also check local cache directory
                if cache_dir.exists():
                    total_size += _safe_get_dir_size(cache_dir)

                return total_size
            elif engine == "granite":
                # Check HF cache (primary location) and local cache for Granite models
                total_size = 0

                # Check HF cache first (where models are actually stored)
                hf_cache_dir = self.base_models_dir / "huggingface" / "hub"
                model_name_safe = model_id.replace("/", "--")
                hf_model_dir = hf_cache_dir / f"models--{model_name_safe}"
                if hf_model_dir.exists():
                    total_size += _safe_get_dir_size(hf_model_dir)

                # Also check local cache directory
                if cache_dir.exists():
                    total_size += _safe_get_dir_size(cache_dir)

                return total_size
            else:
                # For other engines (like voxtral)
                if cache_dir.exists():
                    return _safe_get_dir_size(cache_dir)

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

                # Check if model exists physically using the same logic as is_model_cached
                if engine == "whisper":
                    cache_dir = config["cache_dir"]
                    model_file = cache_dir / f"{model_id}.pt"
                    if model_file.exists():
                        self.mark_model_cached(engine, model_id, cache_dir)
                elif engine == "nvidia":
                    # Check huggingface cache for NeMo models
                    hf_cache_dir = self.base_models_dir / "huggingface" / "hub"
                    model_name_safe = model_id.replace("/", "--")
                    hf_model_dir = hf_cache_dir / f"models--{model_name_safe}"
                    if hf_model_dir.exists() and any(hf_model_dir.rglob("*")):
                        cache_dir = config["cache_dir"]
                        self.mark_model_cached(engine, model_id, cache_dir)
                elif engine == "granite":
                    # Check huggingface cache for Granite models
                    hf_cache_dir = self.base_models_dir / "huggingface" / "hub"
                    model_name_safe = model_id.replace("/", "--")
                    hf_model_dir = hf_cache_dir / f"models--{model_name_safe}"
                    if hf_model_dir.exists() and any(hf_model_dir.rglob("*")):
                        cache_dir = config["cache_dir"]
                        self.mark_model_cached(engine, model_id, cache_dir)
                else:
                    # For other engines (like voxtral), check designated cache directory
                    cache_dir = config["cache_dir"]
                    if cache_dir.exists() and any(cache_dir.iterdir()):
                        self.mark_model_cached(engine, model_id, cache_dir)
