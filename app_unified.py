import streamlit as st
import torch
import os
import tempfile
import whisper
from transformers import VoxtralForConditionalGeneration, AutoProcessor, infer_device
from pydub import AudioSegment
import numpy as np
import csv
import datetime
import pandas as pd
import time
import gc

# Try to import NeMo for Parakeet and Canary support
try:
    import nemo.collections.asr as nemo_asr
    from nemo.collections.speechlm2.models import SALM
    NVIDIA_STT_AVAILABLE = True
    CANARY_AVAILABLE = True
except ImportError:
    NVIDIA_STT_AVAILABLE = False
    CANARY_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Unified STT Demo",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main { padding: 1rem 2rem; }
    .stButton>button {
        width: 100%;
        background-color: #4F46E5;
        color: white;
        padding: 0.5rem 1rem;
        border: none;
        border-radius: 4px;
        font-size: 1rem;
        font-weight: 500;
    }
    .stButton>button:hover { background-color: #4338CA; }
    h1 {
        color: #4F46E5;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin: -1rem 0 1rem 0 !important;
    }
    .info-box {
        background-color: rgba(79, 70, 229, 0.1);
        border-left: 5px solid #4F46E5;
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .success-box {
        background-color: rgba(34, 197, 94, 0.1);
        border-left: 5px solid #22C55E;
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .warning-box {
        background-color: rgba(251, 146, 60, 0.1);
        border-left: 5px solid #FB923C;
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .error-box {
        background-color: rgba(239, 68, 68, 0.1);
        border-left: 5px solid #EF4444;
        padding: 0.5rem;
        border-radius: 4px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUPPORTED_FORMATS = ["wav", "mp3", "flac", "m4a", "ogg"]
MAX_RECOMMENDED_DURATION = 30 * 60

# Model directories
WHISPER_CACHE_DIR = os.path.join(os.getcwd(), "models")
VOXTRAL_CACHE_DIR = os.path.join(os.getcwd(), "models", "voxtral")
PARAKEET_CACHE_DIR = os.path.join(os.getcwd(), "models", "parakeet")
CANARY_CACHE_DIR = os.path.join(os.getcwd(), "models", "canary")

# Ensure directories exist
os.makedirs(WHISPER_CACHE_DIR, exist_ok=True)
os.makedirs(VOXTRAL_CACHE_DIR, exist_ok=True)
os.makedirs(PARAKEET_CACHE_DIR, exist_ok=True)
os.makedirs(CANARY_CACHE_DIR, exist_ok=True)

# Initialize session state
if "transcription_history" not in st.session_state:
    st.session_state.transcription_history = []

# Available models
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large"]
VOXTRAL_MODELS = {
    "mistralai/Voxtral-Mini-3B-2507": "Voxtral Mini 3B (~6GB)",
    "mistralai/Voxtral-Small-24B-2507": "Voxtral Small 24B (~48GB)",
}
PARAKEET_MODELS = {"nvidia/parakeet-tdt-0.6b-v2": "Parakeet TDT 0.6B V2 (~2.4GB)"}
CANARY_MODELS = {"nvidia/canary-qwen-2.5b": "Canary-Qwen 2.5B (~5GB)"}


def custom_info(text):
    st.markdown(f'<div class="info-box">{text}</div>', unsafe_allow_html=True)


def custom_success(text):
    st.markdown(f'<div class="success-box">{text}</div>', unsafe_allow_html=True)


def custom_warning(text):
    st.markdown(f'<div class="warning-box">{text}</div>', unsafe_allow_html=True)


def custom_error(text):
    st.markdown(f'<div class="error-box">{text}</div>', unsafe_allow_html=True)


@st.cache_resource
def load_whisper_model(model_size="base"):
    """Load Whisper model."""
    try:
        model_path = os.path.join(WHISPER_CACHE_DIR, f"{model_size}.pt")
        if os.path.exists(model_path):
            with st.spinner(f"Loading Whisper {model_size} model..."):
                model = whisper.load_model(
                    model_size, download_root=WHISPER_CACHE_DIR, device=DEVICE
                )
                custom_success(f"Whisper {model_size} loaded on {DEVICE.upper()}!")
        else:
            with st.spinner(f"Downloading Whisper {model_size} model..."):
                model = whisper.load_model(
                    model_size, download_root=WHISPER_CACHE_DIR, device=DEVICE
                )
                custom_success(f"Whisper {model_size} downloaded and loaded!")
        return model
    except Exception as e:
        custom_error(f"Error loading Whisper model: {str(e)}")
        return None


@st.cache_resource
def load_voxtral_model(model_id="mistralai/Voxtral-Mini-3B-2507"):
    """Load Voxtral model with proper caching."""
    try:
        # Check if model is already cached
        cached_model_path = os.path.join(VOXTRAL_CACHE_DIR, model_id.replace("/", "--"))

        if os.path.exists(cached_model_path):
            with st.spinner(f"Loading Voxtral model from cache..."):
                processor = AutoProcessor.from_pretrained(
                    model_id, cache_dir=VOXTRAL_CACHE_DIR
                )
                model = VoxtralForConditionalGeneration.from_pretrained(
                    model_id,
                    cache_dir=VOXTRAL_CACHE_DIR,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                custom_success(f"Voxtral model loaded from cache!")
        else:
            with st.spinner(
                f"Downloading and caching Voxtral model... This may take several minutes."
            ):
                custom_info(
                    f"First time loading {model_id} - downloading to local cache..."
                )
                processor = AutoProcessor.from_pretrained(
                    model_id, cache_dir=VOXTRAL_CACHE_DIR
                )
                model = VoxtralForConditionalGeneration.from_pretrained(
                    model_id,
                    cache_dir=VOXTRAL_CACHE_DIR,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
                custom_success(f"Voxtral model downloaded and cached successfully!")

        return model, processor
    except Exception as e:
        custom_error(f"Error loading Voxtral model: {str(e)}")
        return None, None


@st.cache_resource
def load_parakeet_model(model_id="nvidia/parakeet-tdt-0.6b-v2"):
    """Load Parakeet model with proper caching."""
    if not NVIDIA_STT_AVAILABLE:
        custom_error(
            "NeMo toolkit not installed. Install with: pip install nemo_toolkit[asr]"
        )
        return None

    try:
        # Set cache directory for NeMo
        os.environ["NEMO_CACHE_DIR"] = PARAKEET_CACHE_DIR

        with st.spinner(f"Loading Parakeet model..."):
            model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id)
            custom_success(f"Parakeet model loaded successfully!")

        return model
    except Exception as e:
        custom_error(f"Error loading Parakeet model: {str(e)}")
        return None


@st.cache_resource
def load_canary_model(model_id="nvidia/canary-qwen-2.5b"):
    """Load Canary-Qwen model with proper caching."""
    if not CANARY_AVAILABLE:
        custom_error(
            "NeMo toolkit with SALM support not installed. Install with: pip install nemo_toolkit[asr]"
        )
        return None

    try:
        # Set cache directory for NeMo
        os.environ["NEMO_CACHE_DIR"] = CANARY_CACHE_DIR

        with st.spinner(f"Loading Canary-Qwen model..."):
            model = SALM.from_pretrained(model_id)
            custom_success(f"Canary-Qwen model loaded successfully!")

        return model
    except Exception as e:
        custom_error(f"Error loading Canary-Qwen model: {str(e)}")
        return None


def process_audio(audio_path):
    """Process audio file."""
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_sec = audio.duration_seconds

        if duration_sec > MAX_RECOMMENDED_DURATION:
            custom_warning(f"Audio is very long ({duration_sec/60:.1f} minutes).")

        progress_bar = st.progress(0)

        # Convert to mono and resample
        if audio.channels > 1:
            audio = audio.set_channels(1)
        progress_bar.progress(0.5)

        if audio.frame_rate != 16000:
            audio = audio.set_frame_rate(16000)
        progress_bar.progress(0.8)

        # Save processed audio
        temp_dir = tempfile.gettempdir()
        processed_path = os.path.join(temp_dir, "processed_audio.wav")
        audio.export(processed_path, format="wav")

        progress_bar.progress(1.0)
        custom_success("Audio processed successfully!")

        return processed_path, duration_sec
    except Exception as e:
        custom_error(f"Error processing audio: {str(e)}")
        return None, None


def format_time(seconds):
    """Convert seconds to HH:MM:SS format."""
    return str(datetime.timedelta(seconds=seconds)).split(".")[0]


def transcribe_with_whisper(audio_path, model_size="base"):
    """Transcribe with Whisper."""
    start_time = time.time()
    try:
        model = load_whisper_model(model_size)
        if model is None:
            return None

        processed_path, duration_sec = process_audio(audio_path)
        if processed_path is None:
            return None

        try:
            with st.spinner("Transcribing with Whisper..."):
                result = model.transcribe(processed_path, word_timestamps=True)

            if not result or "segments" not in result:
                custom_error("Whisper transcription failed.")
                return None

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

            processing_time = time.time() - start_time
            custom_success(
                f"Whisper transcription completed in {processing_time:.1f} seconds!"
            )

            return csv_data

        finally:
            if os.path.exists(processed_path):
                os.remove(processed_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        custom_error(f"Error during Whisper transcription: {str(e)}")
        return None


def transcribe_with_voxtral(audio_path, model_id="mistralai/Voxtral-Mini-3B-2507"):
    """Transcribe with Voxtral."""
    start_time = time.time()
    try:
        model, processor = load_voxtral_model(model_id)
        if model is None or processor is None:
            return None

        processed_path, duration_sec = process_audio(audio_path)
        if processed_path is None:
            return None

        try:
            with st.spinner("Transcribing with Voxtral..."):
                inputs = processor.apply_transcription_request(
                    language="en", audio=processed_path, model_id=model_id
                )
                inputs = inputs.to(
                    "cuda" if torch.cuda.is_available() else "cpu", dtype=torch.bfloat16
                )

                outputs = model.generate(**inputs, max_new_tokens=500)
                decoded_outputs = processor.batch_decode(
                    outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
                )

                transcription_text = decoded_outputs[0]

            processing_time = time.time() - start_time
            custom_success(
                f"Voxtral transcription completed in {processing_time:.1f} seconds!"
            )

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

            return csv_data

        finally:
            if os.path.exists(processed_path):
                os.remove(processed_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        custom_error(f"Error during Voxtral transcription: {str(e)}")
        return None


def transcribe_with_parakeet(audio_path, model_id="nvidia/parakeet-tdt-0.6b-v2"):
    """Transcribe with Parakeet."""
    start_time = time.time()
    try:
        model = load_parakeet_model(model_id)
        if model is None:
            return None

        processed_path, duration_sec = process_audio(audio_path)
        if processed_path is None:
            return None

        try:
            with st.spinner("Transcribing with Parakeet..."):
                # Transcribe with timestamps
                output = model.transcribe([processed_path], timestamps=True)

                if not output or len(output) == 0:
                    custom_error("Parakeet transcription failed.")
                    return None

            processing_time = time.time() - start_time
            custom_success(
                f"Parakeet transcription completed in {processing_time:.1f} seconds!"
            )

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
                # Use segment-level timestamps
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

            return csv_data

        finally:
            if os.path.exists(processed_path):
                os.remove(processed_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        custom_error(f"Error during Parakeet transcription: {str(e)}")
        return None


def transcribe_with_canary(audio_path, model_id="nvidia/canary-qwen-2.5b"):
    """Transcribe with Canary-Qwen."""
    start_time = time.time()
    try:
        model = load_canary_model(model_id)
        if model is None:
            return None

        processed_path, duration_sec = process_audio(audio_path)
        if processed_path is None:
            return None

        try:
            with st.spinner("Transcribing with Canary-Qwen..."):
                # Use ASR mode for transcription
                prompts = [
                    [{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}", "audio": [processed_path]}]
                ]
                
                answer_ids = model.generate(
                    prompts=prompts,
                    max_new_tokens=128,
                )
                
                transcription_text = model.tokenizer.ids_to_text(answer_ids[0].cpu())

                if not transcription_text or transcription_text.strip() == "":
                    custom_error("Canary-Qwen transcription failed.")
                    return None

            processing_time = time.time() - start_time
            custom_success(
                f"Canary-Qwen transcription completed in {processing_time:.1f} seconds!"
            )

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

            return csv_data

        finally:
            if os.path.exists(processed_path):
                os.remove(processed_path)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    except Exception as e:
        custom_error(f"Error during Canary-Qwen transcription: {str(e)}")
        return None


def get_cached_whisper_models():
    """Get cached Whisper models."""
    cached_models = []
    if os.path.exists(WHISPER_CACHE_DIR):
        for file in os.listdir(WHISPER_CACHE_DIR):
            if file.endswith(".pt"):
                model_name = file.replace(".pt", "")
                file_path = os.path.join(WHISPER_CACHE_DIR, file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)
                cached_models.append({"name": model_name, "size_mb": file_size})
    return cached_models


def get_cached_voxtral_models():
    """Get cached Voxtral models."""
    cached_models = []
    if os.path.exists(VOXTRAL_CACHE_DIR):
        # Look for cached model directories
        for item in os.listdir(VOXTRAL_CACHE_DIR):
            item_path = os.path.join(VOXTRAL_CACHE_DIR, item)
            if os.path.isdir(item_path):
                # Check if it's a model directory (contains config.json)
                config_path = os.path.join(item_path, "config.json")
                if os.path.exists(config_path):
                    # Calculate directory size
                    total_size = 0
                    for dirpath, dirnames, filenames in os.walk(item_path):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            total_size += os.path.getsize(filepath)

                    size_gb = total_size / (1024 * 1024 * 1024)
                    model_name = item.replace(
                        "--", "/"
                    )  # Convert back from cache format
                    cached_models.append({"name": model_name, "size_gb": size_gb})
    return cached_models


def is_voxtral_model_cached(model_id):
    """Check if a specific Voxtral model is cached."""
    cached_models = get_cached_voxtral_models()
    return any(model["name"] == model_id for model in cached_models)


def get_cached_parakeet_models():
    """Get cached Parakeet models."""
    cached_models = []
    if NVIDIA_STT_AVAILABLE and os.path.exists(PARAKEET_CACHE_DIR):
        # NeMo models are typically cached with their full model names
        for item in os.listdir(PARAKEET_CACHE_DIR):
            item_path = os.path.join(PARAKEET_CACHE_DIR, item)
            if os.path.isdir(item_path):
                # Calculate directory size
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(item_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        total_size += os.path.getsize(filepath)

                size_gb = total_size / (1024 * 1024 * 1024)
                cached_models.append({"name": item, "size_gb": size_gb})
    return cached_models


def is_parakeet_model_cached(model_id):
    """Check if a specific Parakeet model is cached."""
    cached_models = get_cached_parakeet_models()
    # Check both full model ID and just the model name part
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id
    return any(model_name in model["name"] for model in cached_models)


def get_cached_canary_models():
    """Get cached Canary models."""
    cached_models = []
    if CANARY_AVAILABLE and os.path.exists(CANARY_CACHE_DIR):
        # NeMo models are typically cached with their full model names
        for item in os.listdir(CANARY_CACHE_DIR):
            item_path = os.path.join(CANARY_CACHE_DIR, item)
            if os.path.isdir(item_path):
                # Calculate directory size
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(item_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        total_size += os.path.getsize(filepath)

                size_gb = total_size / (1024 * 1024 * 1024)
                cached_models.append({"name": item, "size_gb": size_gb})
    return cached_models


def is_canary_model_cached(model_id):
    """Check if a specific Canary model is cached."""
    cached_models = get_cached_canary_models()
    # Check both full model ID and just the model name part
    model_name = model_id.split("/")[-1] if "/" in model_id else model_id
    return any(model_name in model["name"] for model in cached_models)


def export_to_formats(csv_data):
    """Create exportable data."""
    csv_string = "\n".join(
        [
            ",".join([f'"{cell}"' if "," in cell else cell for cell in row])
            for row in csv_data
        ]
    )
    text_string = "\n\n".join(
        [row[-1] for row in csv_data[1:]]
    )  # Last column is transcription
    return csv_string, text_string


# Main UI
st.title("🎙️ Unified Speech Transcription")
st.markdown("Compare Whisper and Voxtral models for speech-to-text transcription")

# Sidebar
with st.sidebar:
    st.title("Model Selection")

    available_engines = ["Whisper", "Voxtral"]
    if NVIDIA_STT_AVAILABLE:
        available_engines.append("Parakeet")
    if CANARY_AVAILABLE:
        available_engines.append("Canary-Qwen")

    engine = st.radio("Choose STT Engine", available_engines)

    if engine == "Whisper":
        st.subheader("Whisper Settings")
        cached_whisper = get_cached_whisper_models()
        cached_names = [m["name"] for m in cached_whisper]

        if cached_names:
            selected_whisper = st.selectbox("Whisper Model", cached_names)
            st.success(f"✅ {len(cached_names)} Whisper model(s) ready")
        else:
            st.warning("No Whisper models cached!")
            selected_whisper = st.selectbox("Download Whisper Model", WHISPER_MODELS)
            if st.button("Download Selected Model"):
                load_whisper_model(selected_whisper)
                st.rerun()

    elif engine == "Voxtral":
        st.subheader("Voxtral Settings")

        # Check cached Voxtral models
        cached_voxtral = get_cached_voxtral_models()

        selected_voxtral = st.selectbox(
            "Voxtral Model",
            list(VOXTRAL_MODELS.keys()),
            format_func=lambda x: VOXTRAL_MODELS[x],
        )

        # Show cache status for selected model
        if is_voxtral_model_cached(selected_voxtral):
            cached_model = next(
                m for m in cached_voxtral if m["name"] == selected_voxtral
            )
            st.success(f"✅ Model cached ({cached_model['size_gb']:.1f} GB)")
        else:
            model_size = "~6GB" if "Mini" in selected_voxtral else "~48GB"
            st.warning(f"⚠️ First load will download {model_size}")

        st.info("🔧 Requires GPU with sufficient VRAM")

        # Show all cached models if any
        if cached_voxtral:
            st.subheader("Cached Models")
            for model in cached_voxtral:
                model_display = "Mini 3B" if "Mini" in model["name"] else "Small 24B"
                st.write(f"• {model_display}: {model['size_gb']:.1f} GB")

    elif engine == "Parakeet":
        st.subheader("Parakeet Settings")

        if not NVIDIA_STT_AVAILABLE:
            st.error("❌ NeMo toolkit not installed!")
            st.code("pip install nemo_toolkit[asr]")
        else:
            # Check cached Parakeet models
            cached_parakeet = get_cached_parakeet_models()

            selected_parakeet = st.selectbox(
                "Parakeet Model",
                list(PARAKEET_MODELS.keys()),
                format_func=lambda x: PARAKEET_MODELS[x],
            )

            # Show cache status for selected model
            if is_parakeet_model_cached(selected_parakeet):
                st.success(f"✅ Model cached")
            else:
                st.warning(f"⚠️ First load will download ~2.4GB")

            st.info("🔧 Optimized for 16kHz audio")

            # Show all cached models if any
            if cached_parakeet:
                st.subheader("Cached Models")
                for model in cached_parakeet:
                    st.write(f"• {model['name']}: {model['size_gb']:.1f} GB")

    else:  # Canary-Qwen
        st.subheader("Canary-Qwen Settings")

        if not CANARY_AVAILABLE:
            st.error("❌ NeMo toolkit with SALM support not installed!")
            st.code("pip install nemo_toolkit[asr]")
        else:
            # Check cached Canary models
            cached_canary = get_cached_canary_models()

            selected_canary = st.selectbox(
                "Canary-Qwen Model",
                list(CANARY_MODELS.keys()),
                format_func=lambda x: CANARY_MODELS[x],
            )

            # Show cache status for selected model
            if is_canary_model_cached(selected_canary):
                st.success(f"✅ Model cached")
            else:
                st.warning(f"⚠️ First load will download ~5GB")

            st.info("🔧 Speech-Augmented Language Model (SALM)")
            st.info("🎯 State-of-the-art English ASR with PnC")

            # Show all cached models if any
            if cached_canary:
                st.subheader("Cached Models")
                for model in cached_canary:
                    st.write(f"• {model['name']}: {model['size_gb']:.1f} GB")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader("Upload an audio file", type=SUPPORTED_FORMATS)

with col2:
    if uploaded_file:
        st.markdown("### Audio Preview")
        st.audio(uploaded_file)

        file_info = f"**File:** {uploaded_file.name}<br>"
        file_info += f"**Size:** {uploaded_file.size / (1024*1024):.2f} MB"
        st.markdown(f'<div class="info-box">{file_info}</div>', unsafe_allow_html=True)

if uploaded_file:
    # Save uploaded file
    temp_dir = tempfile.gettempdir()
    audio_path = os.path.join(temp_dir, uploaded_file.name)
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Transcription button
    if st.button(f"🎯 Transcribe with {engine}", type="primary"):
        if engine == "Whisper":
            if not get_cached_whisper_models():
                st.error("No Whisper models available! Please download a model first.")
            else:
                csv_data = transcribe_with_whisper(audio_path, selected_whisper)
        elif engine == "Voxtral":
            csv_data = transcribe_with_voxtral(audio_path, selected_voxtral)
        elif engine == "Parakeet":
            if not NVIDIA_STT_AVAILABLE:
                st.error(
                    "NeMo toolkit not installed! Install with: pip install nemo_toolkit[asr]"
                )
            else:
                csv_data = transcribe_with_parakeet(audio_path, selected_parakeet)
        else:  # Canary-Qwen
            if not CANARY_AVAILABLE:
                st.error(
                    "NeMo toolkit with SALM support not installed! Install with: pip install nemo_toolkit[asr]"
                )
            else:
                csv_data = transcribe_with_canary(audio_path, selected_canary)

        if "csv_data" in locals() and csv_data:
            # Display results
            st.markdown("### 📝 Transcription Results")

            df = pd.DataFrame(csv_data[1:], columns=csv_data[0])
            st.dataframe(df, hide_index=True, use_container_width=True)

            # Export options
            csv_string, text_string = export_to_formats(csv_data)

            st.markdown("### 📥 Export Options")
            col1, col2 = st.columns(2)

            with col1:
                st.download_button(
                    "📄 Download CSV",
                    data=csv_string,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_{engine.lower()}_transcript.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with col2:
                st.download_button(
                    "📝 Download Text",
                    data=text_string,
                    file_name=f"{os.path.splitext(uploaded_file.name)[0]}_{engine.lower()}_transcript.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

    # Cleanup
    if os.path.exists(audio_path):
        os.remove(audio_path)

# Model comparison info
st.markdown("---")
st.markdown("### 🔍 Model Comparison")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        """
    **Whisper**
    - ✅ Detailed timestamps
    - ✅ Multiple model sizes
    - ✅ Fast processing
    - ✅ Lower resource requirements
    - ✅ Proven accuracy
    """
    )

with col2:
    st.markdown(
        """
    **Voxtral**
    - ✅ Advanced audio understanding
    - ✅ Multilingual support
    - ✅ Long-form context (30 min)
    - ✅ Audio Q&A capabilities
    - ⚠️ Higher resource requirements
    """
    )

with col3:
    st.markdown(
        """
    **Parakeet**
    - ✅ NVIDIA optimized
    - ✅ Fast inference
    - ✅ Timestamps support
    - ✅ Punctuation & capitalization
    - ✅ 600M parameters
    """
    )

with col4:
    st.markdown(
        """
    **Canary-Qwen**
    - ✅ State-of-the-art English ASR
    - ✅ 418 RTFx performance
    - ✅ Punctuation & capitalization
    - ✅ ASR + LLM modes
    - ✅ 2.5B parameters
    """
    )

st.markdown("---")
st.markdown(
    "Choose Whisper for detailed timestamps and efficiency, Voxtral for advanced audio understanding, Parakeet for NVIDIA-optimized performance, or Canary-Qwen for state-of-the-art English ASR."
)
