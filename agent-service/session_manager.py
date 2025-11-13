"""
Session management for voice agent conversations
"""
import uuid
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from fastapi import WebSocket


@dataclass
class AgentSession:
    """Represents an active voice agent session"""
    session_id: str
    frontend_ws: WebSocket
    audio_buffer: List[bytes] = field(default_factory=list)
    is_processing: bool = False
    last_activity: float = field(default_factory=time.time)
    stt_engine: str = "whisper"
    stt_model: str = "large-v3"
    tts_engine: str = "parler"
    tts_model: str = "parler_tts_mini_v1"
    tts_description: str = "A clear, friendly voice with neutral tone"
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
    
    def clear_audio_buffer(self):
        """Clear the audio buffer"""
        self.audio_buffer = []


class SessionManager:
    """Manages active agent sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, AgentSession] = {}
    
    def create_session(self, websocket: WebSocket) -> str:
        """Create a new session"""
        session_id = str(uuid.uuid4())
        session = AgentSession(
            session_id=session_id,
            frontend_ws=websocket
        )
        self.sessions[session_id] = session
        return session_id
    
    def get_session(self, session_id: str) -> Optional[AgentSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def remove_session(self, session_id: str):
        """Remove a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def cleanup_inactive_sessions(self, timeout_seconds: int = 300):
        """Remove sessions that have been inactive"""
        current_time = time.time()
        inactive_sessions = [
            sid for sid, session in self.sessions.items()
            if current_time - session.last_activity > timeout_seconds
        ]
        for sid in inactive_sessions:
            self.remove_session(sid)
    
    def get_active_session_count(self) -> int:
        """Get count of active sessions"""
        return len(self.sessions)
