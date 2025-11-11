#!/usr/bin/env python3
"""
Parler-TTS model loader
"""

import os
from pathlib import Path
from typing import Any, Tuple
import torch
import numpy as np

from .base_tts_loader import BaseTTSLoader


class ParlerTTSLoader(BaseTTSLoader):
    """Loader for Parler-TTS models."""
    
    def __init__(self):
        super().__init__("parler")
    
    def check_dependencies(self) -> Tuple[bool, str]:
        """Check if Parler-TTS dependencies are available."""
        try:
            import parler_tts
            from transformers import AutoTokenizer
            return True, ""
        except ImportError as e:
            return False, f"Parler-TTS not available: {str(e)}. Install with: pip install git+https://github.com/huggingface/parler-tts.git"
    
    def setup_cache_environment(self, cache_dir: Path) -> None:
        """Setup cache environment for Parler-TTS."""
        os.environ["HF_HOME"] = str(cache_dir.parent)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir.parent / "hub")
    
    def load_model(self, model_id: str, cache_dir: Path, device: str) -> dict:
        """
        Load Parler-TTS model.
        
        Returns:
            dict: Dictionary containing model, tokenizer, and description_tokenizer
        """
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        
        self.setup_cache_environment(cache_dir)
        
        print(f"Loading Parler-TTS model: {model_id}")
        
        # Load model
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir=cache_dir
        ).to(device)
        
        # Load tokenizers
        tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        description_tokenizer = AutoTokenizer.from_pretrained(
            model.config.text_encoder._name_or_path,
            cache_dir=cache_dir
        )
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "description_tokenizer": description_tokenizer,
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
        Synthesize speech using Parler-TTS.
        
        Args:
            model_dict: Dictionary with model, tokenizer, and description_tokenizer
            text: Text to synthesize
            description: Voice description (e.g., "A female speaker with a calm voice")
            **kwargs: Additional parameters
            
        Returns:
            Tuple[audio_array, sample_rate]
        """
        model = model_dict["model"]
        tokenizer = model_dict["tokenizer"]
        description_tokenizer = model_dict["description_tokenizer"]
        device = model_dict["device"]
        
        # Default description if none provided
        if description is None:
            description = "A clear and neutral voice with moderate speed and pitch."
        
        # Tokenize inputs
        input_ids = description_tokenizer(
            description, 
            return_tensors="pt"
        ).input_ids.to(device)
        
        prompt_input_ids = tokenizer(
            text, 
            return_tensors="pt"
        ).input_ids.to(device)
        
        # Generate audio
        generation = model.generate(
            input_ids=input_ids, 
            prompt_input_ids=prompt_input_ids
        )
        
        # Convert to numpy array
        audio_arr = generation.cpu().numpy().squeeze()
        sample_rate = model.config.sampling_rate
        
        return audio_arr, sample_rate
    
    def get_model_info(self, model_id: str) -> dict:
        """Get Parler-TTS model information."""
        return {
            "engine": "parler",
            "model_id": model_id,
            "type": "text-to-speech",
            "multilingual": True,
            "supports_voice_cloning": False,
            "supports_voice_description": True
        }
