"""
Configuration for Agent Service
"""
import os

# AWS Bedrock Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0")

# Service URLs
STT_SERVICE_URL = os.getenv("STT_SERVICE_URL", "http://stt:8001")
TTS_SERVICE_URL = os.getenv("TTS_SERVICE_URL", "http://tts:8002")

# STT/TTS WebSocket URLs
STT_WS_URL = STT_SERVICE_URL.replace("http://", "ws://").replace("https://", "wss://")
TTS_WS_URL = TTS_SERVICE_URL.replace("http://", "ws://").replace("https://", "wss://")

# Conversation Settings
MAX_CONVERSATION_HISTORY = int(os.getenv("MAX_CONVERSATION_HISTORY", "10"))
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful voice assistant. Keep responses concise and conversational."
)

# Audio Processing
SILENCE_THRESHOLD_MS = int(os.getenv("SILENCE_THRESHOLD_MS", "1500"))
MIN_AUDIO_LENGTH_MS = int(os.getenv("MIN_AUDIO_LENGTH_MS", "500"))

# Default Models
DEFAULT_STT_ENGINE = os.getenv("DEFAULT_STT_ENGINE", "whisper")
DEFAULT_STT_MODEL = os.getenv("DEFAULT_STT_MODEL", "large-v3")
DEFAULT_TTS_ENGINE = os.getenv("DEFAULT_TTS_ENGINE", "parler")
DEFAULT_TTS_MODEL = os.getenv("DEFAULT_TTS_MODEL", "parler_tts_mini_v1")
DEFAULT_TTS_DESCRIPTION = os.getenv(
    "DEFAULT_TTS_DESCRIPTION",
    "A clear, friendly voice with neutral tone"
)

# Session Management
SESSION_TIMEOUT_SECONDS = int(os.getenv("SESSION_TIMEOUT_SECONDS", "300"))

# LangChain Memory Storage
MEMORY_STORAGE_PATH = os.getenv("MEMORY_STORAGE_PATH", "./session_memory")

# LLM Provider Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "bedrock")  # bedrock, openai, anthropic, etc.
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
