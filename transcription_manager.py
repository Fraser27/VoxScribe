#!/usr/bin/env python3
"""
Transcription Manager for VoxScribe - Manages transcription storage and retrieval
"""

import json
import datetime
import logging
from pathlib import Path
from typing import List, Dict

logger = logging.getLogger("voxscribe")


class TranscriptionManager:
    """Manages transcription storage and retrieval."""

    def __init__(self, transcriptions_dir: Path):
        self.transcriptions_dir = transcriptions_dir
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
        rtfx: float = None,
    ):
        """Save transcription results and metadata."""
        from config import DEVICE
        
        try:
            # Create transcription entry
            transcription_entry = {
                "id": transcription_id,
                "timestamp": datetime.datetime.now().isoformat(),
                "engine": engine,
                "model_id": model_id,
                "model_display_name": self._get_model_display_name(engine, model_id),
                "audio_filename": audio_filename,
                "audio_duration_seconds": audio_duration,
                "transcription_duration_seconds": transcription_duration,
                "rtfx": rtfx,
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
                transcription_entry["segments_count"] = (
                    len(csv_data) - 1
                )  # Subtract header row

            # Add to metadata
            self.metadata["transcriptions"].insert(
                0, transcription_entry
            )  # Most recent first

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
            transcriptions = self.metadata["transcriptions"][offset : offset + limit]

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
                "has_more": offset + limit < total,
            }

        except Exception as e:
            logger.error(f"Failed to get transcriptions list: {e}")
            return {
                "transcriptions": [],
                "total": 0,
                "limit": limit,
                "offset": offset,
                "has_more": False,
            }

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

    def _get_model_display_name(self, engine: str, model_id: str):
        """Get display name for a model."""
        from config import MODEL_REGISTRY
        
        model_config = MODEL_REGISTRY[engine][model_id]
        if "display_name" in model_config:
            return model_config["display_name"]
        return model_id