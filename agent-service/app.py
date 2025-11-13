#!/usr/bin/env python3
"""
FastAPI Agent Service - Real-time Voice Agent Orchestrator
"""
import asyncio
import json
import base64
import time
from typing import Optional
import aiohttp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import (
    STT_WS_URL,
    TTS_WS_URL,
    SESSION_TIMEOUT_SECONDS,
    DEFAULT_STT_ENGINE,
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_ENGINE,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_DESCRIPTION,
)
from strands_client import StrandsClient
from session_manager import SessionManager, AgentSession
from audio_processor import AudioProcessor

# Initialize FastAPI
app = FastAPI(
    title="VoxScribe Agent API",
    description="Real-time Voice Agent Orchestration Service with Strands Agents",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
strands_client = StrandsClient()
session_manager = SessionManager()
audio_processor = AudioProcessor()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "VoxScribe Agent",
        "version": "2.0.0",
        "framework": "Strands Agents",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "active_sessions": session_manager.get_active_session_count()
    }


@app.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket):
    """
    Main agent WebSocket endpoint
    Handles bidirectional streaming for voice conversations
    """
    await websocket.accept()
    session_id = session_manager.create_session(websocket)
    session = session_manager.get_session(session_id)
    
    print(f"New agent session created: {session_id}")
    
    # Send session info to client
    await websocket.send_json({
        "type": "session_created",
        "session_id": session_id,
        "default_models": {
            "stt_engine": DEFAULT_STT_ENGINE,
            "stt_model": DEFAULT_STT_MODEL,
            "tts_engine": DEFAULT_TTS_ENGINE,
            "tts_model": DEFAULT_TTS_MODEL
        }
    })
    
    try:
        while True:
            # Receive message from frontend
            data = await websocket.receive_text()
            message = json.loads(data)
            
            session.update_activity()
            
            msg_type = message.get("type")
            
            if msg_type == "configure":
                # Update session configuration
                await handle_configure(session, message)
            
            elif msg_type == "audio_chunk":
                # Handle incoming audio chunk
                await handle_audio_chunk(session, message)
            
            elif msg_type == "end_speech":
                # User finished speaking, process the utterance
                await handle_end_speech(session)
            
            elif msg_type == "ping":
                await websocket.send_json({"type": "pong"})
            
            elif msg_type == "clear_history":
                # Clear Strands session history
                strands_client.clear_session(session_id)
                await websocket.send_json({
                    "type": "history_cleared",
                    "session_id": session_id
                })
            
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": f"Unknown message type: {msg_type}"
                })
    
    except WebSocketDisconnect:
        print(f"Session disconnected: {session_id}")
        session_manager.remove_session(session_id)
    except Exception as e:
        print(f"Error in agent websocket: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })
        session_manager.remove_session(session_id)


async def handle_configure(session: AgentSession, message: dict):
    """Handle configuration updates"""
    config = message.get("config", {})
    
    if "stt_engine" in config:
        session.stt_engine = config["stt_engine"]
    if "stt_model" in config:
        session.stt_model = config["stt_model"]
    if "tts_engine" in config:
        session.tts_engine = config["tts_engine"]
    if "tts_model" in config:
        session.tts_model = config["tts_model"]
    if "tts_description" in config:
        session.tts_description = config["tts_description"]
    
    await session.frontend_ws.send_json({
        "type": "configured",
        "config": {
            "stt_engine": session.stt_engine,
            "stt_model": session.stt_model,
            "tts_engine": session.tts_engine,
            "tts_model": session.tts_model
        }
    })


async def handle_audio_chunk(session: AgentSession, message: dict):
    """Handle incoming audio chunk from frontend"""
    audio_data = message.get("data")
    if not audio_data:
        return
    
    # Decode and buffer audio
    audio_bytes = audio_processor.decode_audio_chunk(audio_data)
    session.audio_buffer.append(audio_bytes)


async def handle_end_speech(session: AgentSession):
    """Process complete user utterance"""
    if session.is_processing:
        await session.frontend_ws.send_json({
            "type": "error",
            "message": "Already processing a request"
        })
        return
    
    if not session.audio_buffer:
        await session.frontend_ws.send_json({
            "type": "error",
            "message": "No audio to process"
        })
        return
    
    session.is_processing = True
    
    try:
        # Step 1: Transcribe audio
        await session.frontend_ws.send_json({"type": "status", "status": "transcribing"})
        
        transcription = await transcribe_audio(session)
        
        if not transcription:
            await session.frontend_ws.send_json({
                "type": "error",
                "message": "Transcription failed"
            })
            session.is_processing = False
            return
        
        # Send transcription to frontend
        await session.frontend_ws.send_json({
            "type": "transcription_final",
            "text": transcription,
            "session_id": session.session_id
        })
        
        # Step 2: Get LLM response (Strands handles conversation history automatically)
        await session.frontend_ws.send_json({"type": "status", "status": "thinking"})
        
        await process_llm_response(session, transcription)
        
    except Exception as e:
        print(f"Error processing speech: {e}")
        await session.frontend_ws.send_json({
            "type": "error",
            "message": str(e)
        })
    finally:
        session.is_processing = False
        session.clear_audio_buffer()


async def transcribe_audio(session: AgentSession) -> Optional[str]:
    """Send audio to STT service and get transcription"""
    try:
        # Concatenate audio buffer
        full_audio = audio_processor.concatenate_audio_chunks(session.audio_buffer)
        audio_base64 = audio_processor.encode_audio_chunk(full_audio)
        
        # Connect to STT WebSocket
        async with aiohttp.ClientSession() as http_session:
            async with http_session.ws_connect(f"{STT_WS_URL}/ws/stt/stream") as stt_ws:
                # Send configuration
                await stt_ws.send_json({
                    "type": "configure",
                    "engine": session.stt_engine,
                    "model_id": session.stt_model
                })
                
                # Send audio
                await stt_ws.send_json({
                    "type": "audio_data",
                    "data": audio_base64
                })
                
                # Send finalize
                await stt_ws.send_json({
                    "type": "finalize"
                })
                
                # Wait for transcription
                async for msg in stt_ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        
                        if data.get("type") == "transcription":
                            return data.get("text", "")
                        elif data.get("type") == "error":
                            print(f"STT error: {data.get('message')}")
                            return None
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print(f"STT WebSocket error")
                        return None
        
        return None
        
    except Exception as e:
        print(f"Error in transcribe_audio: {e}")
        return None


async def process_llm_response(session: AgentSession, user_message: str):
    """Get LLM response using Strands and stream to TTS"""
    try:
        # Stream response from Strands agent
        # Strands automatically manages conversation history via session store
        full_response = ""
        sentence_buffer = ""
        
        async for chunk in strands_client.stream_response(
            session_id=session.session_id,
            user_message=user_message
        ):
            full_response += chunk
            sentence_buffer += chunk
            
            # Send chunk to frontend for display
            await session.frontend_ws.send_json({
                "type": "agent_response_chunk",
                "text": chunk
            })
            
            # Check if we have a complete sentence to send to TTS
            if any(punct in sentence_buffer for punct in ['. ', '! ', '? ', '\n']):
                # Send to TTS for audio generation
                await generate_and_stream_audio(session, sentence_buffer.strip())
                sentence_buffer = ""
        
        # Send any remaining text to TTS
        if sentence_buffer.strip():
            await generate_and_stream_audio(session, sentence_buffer.strip())
        
        # Notify completion
        await session.frontend_ws.send_json({
            "type": "response_complete",
            "session_id": session.session_id
        })
        
    except Exception as e:
        print(f"Error in process_llm_response: {e}")
        await session.frontend_ws.send_json({
            "type": "error",
            "message": f"LLM error: {str(e)}"
        })


async def generate_and_stream_audio(session: AgentSession, text: str):
    """Generate audio from text and stream to frontend"""
    try:
        async with aiohttp.ClientSession() as http_session:
            async with http_session.ws_connect(f"{TTS_WS_URL}/ws/tts/stream") as tts_ws:
                # Send configuration
                await tts_ws.send_json({
                    "type": "configure",
                    "engine": session.tts_engine,
                    "model_id": session.tts_model,
                    "description": session.tts_description
                })
                
                # Send text
                await tts_ws.send_json({
                    "type": "text",
                    "text": text
                })
                
                # Send finalize
                await tts_ws.send_json({
                    "type": "finalize"
                })
                
                # Stream audio chunks to frontend
                async for msg in tts_ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        
                        if data.get("type") == "audio_chunk":
                            # Forward audio chunk to frontend
                            await session.frontend_ws.send_json({
                                "type": "audio_chunk",
                                "data": data.get("data"),
                                "chunk_index": data.get("chunk_index", 0)
                            })
                        elif data.get("type") == "audio_complete":
                            await session.frontend_ws.send_json({
                                "type": "audio_complete"
                            })
                            break
                        elif data.get("type") == "error":
                            print(f"TTS error: {data.get('message')}")
                            break
                    elif msg.type == aiohttp.WSMsgType.ERROR:
                        print(f"TTS WebSocket error")
                        break
    
    except Exception as e:
        print(f"Error in generate_and_stream_audio: {e}")


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 50)
    print("VoxScribe Agent Service Starting")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8003)
