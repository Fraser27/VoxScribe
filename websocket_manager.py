#!/usr/bin/env python3
"""
WebSocket Manager for VoxScribe - Handles real-time updates and model downloads
"""

import asyncio
import datetime
import logging
from typing import Set, Dict
from fastapi import WebSocket

logger = logging.getLogger("voxscribe")


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
    
    def get_connection_count(self) -> int:
        """Get the number of active WebSocket connections."""
        return len(self.active_connections)
    
    def has_active_connections(self) -> bool:
        """Check if there are any active WebSocket connections."""
        return len(self.active_connections) > 0

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
                # Check if connection is still open before sending
                if hasattr(connection, 'client_state') and connection.client_state.name != 'CONNECTED':
                    logger.warning(f"WebSocket connection not in CONNECTED state: {connection.client_state}")
                    disconnected.add(connection)
                    continue
                    
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

    async def send_transcription_progress(
        self,
        engine: str,
        model_id: str,
        filename: str,
        stage: str,
        message: str = "",
        progress: float = 0,
    ):
        """Send transcription progress update."""
        logger.info(
            f"Sending transcription progress: {engine}/{model_id} - {stage} - {message}"
        )
        
        # Only send if we have active connections
        if not self.active_connections:
            logger.warning("No active WebSocket connections for transcription progress")
            return
            
        await self.send_to_all(
            {
                "type": "transcription_progress",
                "engine": engine,
                "model_id": model_id,
                "filename": filename,
                "stage": stage,
                "message": message,
                "progress": progress,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

    async def send_transcription_complete(
        self,
        engine: str,
        model_id: str,
        filename: str,
        success: bool,
        duration: float = 0,
        processing_time: float = 0,
        rtfx: float = 0,
        error: str = None,
    ):
        """Send transcription completion notification."""
        
        # Only send if we have active connections
        if not self.active_connections:
            logger.warning("No active WebSocket connections for transcription complete")
            return
            
        await self.send_to_all(
            {
                "type": "transcription_complete",
                "engine": engine,
                "model_id": model_id,
                "filename": filename,
                "success": success,
                "duration": duration,
                "processing_time": processing_time,
                "rtfx": rtfx,
                "error": error,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

    async def send_synthesis_progress(
        self,
        engine: str,
        model_id: str,
        text: str,
        stage: str,
        message: str = "",
        progress: float = 0,
    ):
        """Send TTS synthesis progress update."""
        logger.info(
            f"Sending synthesis progress: {engine}/{model_id} - {stage} - {message}"
        )
        
        if not self.active_connections:
            logger.warning("No active WebSocket connections for synthesis progress")
            return
            
        await self.send_to_all(
            {
                "type": "synthesis_progress",
                "engine": engine,
                "model_id": model_id,
                "text": text[:50] + "..." if len(text) > 50 else text,
                "stage": stage,
                "message": message,
                "progress": progress,
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

    async def send_synthesis_complete(
        self,
        engine: str,
        model_id: str,
        text: str,
        success: bool,
        audio_duration: float = 0,
        processing_time: float = 0,
        error: str = None,
    ):
        """Send TTS synthesis completion notification."""
        
        if not self.active_connections:
            logger.warning("No active WebSocket connections for synthesis complete")
            return
            
        await self.send_to_all(
            {
                "type": "synthesis_complete",
                "engine": engine,
                "model_id": model_id,
                "text": text[:50] + "..." if len(text) > 50 else text,
                "success": success,
                "audio_duration": audio_duration,
                "processing_time": processing_time,
                "error": error,
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
                engine, model_id, 10, "starting", "Initializing download..."
            )

            # Get global manager instances
            from global_managers import get_model_manager
            from config import get_manager_type_for_engine
            
            # Determine manager type based on engine
            mgr_type = get_manager_type_for_engine(engine)
            model_manager = get_model_manager(mgr_type)
            
            # Check if model is already cached
            if model_manager.is_model_cached(engine, model_id):
                logger.info(f"Model {engine}/{model_id} already cached")
                await self.send_download_progress(
                    engine, model_id, 75, "downloading", f"Loading model {engine}/{model_id} from cache..."
                )
                await self.send_download_complete(engine, model_id, True)
                return

            await self.send_download_progress(
                engine, model_id, 25, "downloading", "Downloading model files..."
            )
            # Create a wrapper function that provides progress updates
            from model_loader import load_model
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