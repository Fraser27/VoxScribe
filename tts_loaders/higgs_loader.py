#!/usr/bin/env python3
"""
Higgs Audio V2 model loader
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Tuple
import numpy as np

from .base_tts_loader import BaseTTSLoader


class HiggsAudioLoader(BaseTTSLoader):
    """Loader for Higgs Audio V2 models."""
    
    def __init__(self):
        super().__init__("higgs")
        self.higgs_repo_path = None
    
    def check_dependencies(self) -> Tuple[bool, str]:
        """Check if Higgs Audio dependencies are available."""
        try:
            # Check if higgs-audio repo is cloned
            models_dir = Path(os.getcwd()) / "models"
            higgs_path = models_dir / "higgs-audio"
            
            if not higgs_path.exists():
                print("Higgs Audio repository not found. Installing automatically...")
                
                # Create models directory if it doesn't exist
                models_dir.mkdir(parents=True, exist_ok=True)
                
                # Clone the repository
                print("Cloning Higgs Audio repository...")
                clone_result = subprocess.run(
                    ["git", "clone", "https://github.com/boson-ai/higgs-audio.git", str(higgs_path)],
                    cwd=str(models_dir),
                    capture_output=True,
                    text=True
                )
                
                if clone_result.returncode != 0:
                    return False, (
                        f"Failed to clone Higgs Audio repository: {clone_result.stderr}\n"
                        "Please clone manually:\n"
                        "cd models && git clone https://github.com/boson-ai/higgs-audio.git"
                    )
                
                print("✓ Repository cloned successfully")
                
                # Install requirements
                print("Installing dependencies...")
                requirements_file = higgs_path / "requirements.txt"
                if requirements_file.exists():
                    install_result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                        capture_output=True,
                        text=True
                    )
                    
                    if install_result.returncode != 0:
                        return False, (
                            f"Failed to install requirements: {install_result.stderr}\n"
                            "Please install manually:\n"
                            "cd models/higgs-audio && pip install -r requirements.txt"
                        )
                    
                    print("✓ Dependencies installed")
                
                # Install package in editable mode
                print("Installing Higgs Audio package...")
                package_install_result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-e", str(higgs_path)],
                    capture_output=True,
                    text=True
                )
                
                if package_install_result.returncode != 0:
                    return False, (
                        f"Failed to install Higgs Audio package: {package_install_result.stderr}\n"
                        "Please install manually:\n"
                        "cd models/higgs-audio && pip install -e ."
                    )
                
                print("✓ Higgs Audio installed successfully!")
            
            # Add higgs-audio to Python path if not already there
            if str(higgs_path) not in sys.path:
                sys.path.insert(0, str(higgs_path))
            
            self.higgs_repo_path = higgs_path
            
            # Try importing the required modules
            from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
            from boson_multimodal.data_types import ChatMLSample, Message
            import torch
            import torchaudio
            
            return True, ""
        except ImportError as e:
            return False, (
                f"Higgs Audio dependencies not available: {str(e)}. "
                "Please install manually:\n"
                "cd models/higgs-audio && pip install -r requirements.txt && pip install -e ."
            )
        except Exception as e:
            return False, (
                f"Unexpected error during Higgs Audio setup: {str(e)}\n"
                "Please try manual installation:\n"
                "cd models && git clone https://github.com/boson-ai/higgs-audio.git && "
                "cd higgs-audio && pip install -r requirements.txt && pip install -e ."
            )
    
    def setup_cache_environment(self, cache_dir: Path) -> None:
        """Setup cache environment for Higgs Audio."""
        os.environ["HF_HOME"] = str(cache_dir.parent)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir.parent / "hub")
    
    def load_model(self, model_id: str, cache_dir: Path, device: str) -> dict:
        """
        Load Higgs Audio model.
        
        Returns:
            dict: Dictionary containing serve_engine and device
        """
        from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
        
        self.setup_cache_environment(cache_dir)
        
        print(f"Loading Higgs Audio model: {model_id}")
        
        # Extract model path and tokenizer path from model_id
        # Format: "model_path|tokenizer_path" or just "model_path" (uses default tokenizer)
        if "|" in model_id:
            model_path, tokenizer_path = model_id.split("|", 1)
        else:
            model_path = model_id
            tokenizer_path = "bosonai/higgs-audio-v2-tokenizer"
        
        # Load the serve engine
        serve_engine = HiggsAudioServeEngine(
            model_path,
            tokenizer_path,
            device=device
        )
        
        return {
            "serve_engine": serve_engine,
            "device": device
        }
    
    def synthesize(
        self, 
        model_dict: dict, 
        text: str, 
        description: str = None,
        **kwargs
    ) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech using Higgs Audio.
        
        Args:
            model_dict: Dictionary with serve_engine
            text: Text to synthesize
            description: Scene description (optional, defaults to quiet room)
            **kwargs: Additional parameters (max_new_tokens, temperature, top_p, top_k)
            
        Returns:
            Tuple[audio_array, sample_rate]
        """
        from boson_multimodal.data_types import ChatMLSample, Message
        from boson_multimodal.serve.serve_engine import HiggsAudioResponse
        import torch
        
        serve_engine = model_dict["serve_engine"]
        
        # Default scene description if none provided
        if description is None:
            description = "Audio is recorded from a quiet room."
        
        # Build system prompt with scene description
        system_prompt = (
            f"Generate audio following instruction.\n\n"
            f"<|scene_desc_start|>\n{description}\n<|scene_desc_end|>"
        )
        
        # Create messages
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=text),
        ]
        
        # Extract generation parameters
        max_new_tokens = kwargs.get("max_new_tokens", 1024)
        temperature = kwargs.get("temperature", 0.3)
        top_p = kwargs.get("top_p", 0.95)
        top_k = kwargs.get("top_k", 50)
        
        # Generate audio
        output: HiggsAudioResponse = serve_engine.generate(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )
        
        # Convert to numpy array
        audio_arr = output.audio
        sample_rate = output.sampling_rate
        
        return audio_arr, sample_rate
    
    def get_model_info(self, model_id: str) -> dict:
        """Get Higgs Audio model information."""
        return {
            "engine": "higgs",
            "model_id": model_id,
            "type": "text-to-speech",
            "multilingual": True,
            "supports_voice_cloning": False,
            "supports_scene_description": True,
            "description": "Higgs Audio V2: Redefining Expressiveness in Audio Generation"
        }
