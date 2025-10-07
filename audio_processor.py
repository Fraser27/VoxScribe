#!/usr/bin/env python3
"""
Audio processing utilities for VoxScribe
"""

import os
import tempfile
import datetime
from pydub import AudioSegment


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