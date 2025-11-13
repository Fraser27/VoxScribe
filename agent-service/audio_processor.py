"""
Audio processing utilities for voice agent
"""
import base64
import numpy as np
from typing import List, Tuple


class AudioProcessor:
    """Handles audio buffering and basic processing"""
    
    def __init__(self, silence_threshold_ms: int = 1500):
        self.silence_threshold_ms = silence_threshold_ms
        self.sample_rate = 16000  # Standard for speech
    
    def decode_audio_chunk(self, base64_data: str) -> bytes:
        """Decode base64 audio chunk"""
        try:
            return base64.b64decode(base64_data)
        except Exception as e:
            print(f"Error decoding audio: {e}")
            return b""
    
    def encode_audio_chunk(self, audio_bytes: bytes) -> str:
        """Encode audio chunk to base64"""
        return base64.b64encode(audio_bytes).decode('utf-8')
    
    def concatenate_audio_chunks(self, chunks: List[bytes]) -> bytes:
        """Concatenate multiple audio chunks"""
        return b"".join(chunks)
    
    def is_silence(self, audio_chunk: bytes, threshold: float = 0.01) -> bool:
        """
        Simple silence detection based on audio amplitude
        
        Args:
            audio_chunk: Raw audio bytes
            threshold: Amplitude threshold for silence
        
        Returns:
            True if chunk is considered silence
        """
        try:
            # Convert bytes to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # Normalize to [-1, 1]
            normalized = audio_array.astype(np.float32) / 32768.0
            
            # Calculate RMS (root mean square) energy
            rms = np.sqrt(np.mean(normalized ** 2))
            
            return rms < threshold
        except Exception as e:
            print(f"Error in silence detection: {e}")
            return False
    
    def get_audio_duration_ms(self, audio_bytes: bytes, sample_rate: int = 16000, channels: int = 1, sample_width: int = 2) -> float:
        """
        Calculate audio duration in milliseconds
        
        Args:
            audio_bytes: Raw audio data
            sample_rate: Sample rate in Hz
            channels: Number of audio channels
            sample_width: Bytes per sample (2 for 16-bit)
        
        Returns:
            Duration in milliseconds
        """
        num_samples = len(audio_bytes) / (channels * sample_width)
        duration_seconds = num_samples / sample_rate
        return duration_seconds * 1000
    
    def should_finalize_speech(
        self,
        audio_buffer: List[bytes],
        recent_silence_chunks: int,
        chunk_duration_ms: float = 100
    ) -> bool:
        """
        Determine if we should finalize speech based on silence duration
        
        Args:
            audio_buffer: Current audio buffer
            recent_silence_chunks: Number of recent silent chunks
            chunk_duration_ms: Duration of each chunk in ms
        
        Returns:
            True if speech should be finalized
        """
        if not audio_buffer:
            return False
        
        silence_duration_ms = recent_silence_chunks * chunk_duration_ms
        return silence_duration_ms >= self.silence_threshold_ms
