#!/usr/bin/env python3
"""
Script to check cached models status
"""
import os


def check_whisper_cache():
    """Check Whisper model cache."""
    print("Whisper Model Cache")
    print("=" * 30)

    whisper_cache_dir = os.path.join(os.getcwd(), "models")

    if not os.path.exists(whisper_cache_dir):
        print("❌ No Whisper cache directory found")
        return

    cached_models = []
    total_size = 0

    for file in os.listdir(whisper_cache_dir):
        if file.endswith(".pt"):
            model_name = file.replace(".pt", "")
            file_path = os.path.join(whisper_cache_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            cached_models.append({"name": model_name, "size_mb": file_size})
            total_size += file_size

    if cached_models:
        print(f"✅ Found {len(cached_models)} cached Whisper model(s):")
        for model in cached_models:
            print(f"  • {model['name']}: {model['size_mb']:.1f} MB")
        print(f"Total Whisper cache size: {total_size:.1f} MB")
    else:
        print("❌ No Whisper models cached")

    print()


def check_voxtral_cache():
    """Check Voxtral model cache."""
    print("Voxtral Model Cache")
    print("=" * 30)

    voxtral_cache_dir = os.path.join(os.getcwd(), "models", "voxtral")

    if not os.path.exists(voxtral_cache_dir):
        print("❌ No Voxtral cache directory found")
        return

    cached_models = []
    total_size = 0

    # Look for cached model directories
    for item in os.listdir(voxtral_cache_dir):
        item_path = os.path.join(voxtral_cache_dir, item)
        if os.path.isdir(item_path):
            # Check if it's a model directory (contains config.json)
            config_path = os.path.join(item_path, "config.json")
            if os.path.exists(config_path):
                # Calculate directory size
                dir_size = 0
                for dirpath, dirnames, filenames in os.walk(item_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        try:
                            dir_size += os.path.getsize(filepath)
                        except (OSError, FileNotFoundError):
                            continue

                size_gb = dir_size / (1024 * 1024 * 1024)
                model_name = item.replace("--", "/")  # Convert back from cache format
                cached_models.append({"name": model_name, "size_gb": size_gb})
                total_size += size_gb

    if cached_models:
        print(f"✅ Found {len(cached_models)} cached Voxtral model(s):")
        for model in cached_models:
            model_display = "Mini 3B" if "Mini" in model["name"] else "Small 24B"
            print(f"  • {model_display}: {model['size_gb']:.1f} GB")
        print(f"Total Voxtral cache size: {total_size:.1f} GB")
    else:
        print("❌ No Voxtral models cached")

    print()


def check_system_info():
    """Check system information."""
    print("System Information")
    print("=" * 30)

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"CUDA available: Yes")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
        else:
            print("CUDA available: No (CPU only)")
    except ImportError:
        print("PyTorch not installed")

    try:
        import transformers

        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("Transformers not installed")

    try:
        import whisper

        print(f"Whisper available: Yes")
    except ImportError:
        print("Whisper not installed")

    print()


def main():
    """Check all cache status."""
    print("Model Cache Status Check")
    print("=" * 50)
    print()

    check_system_info()
    check_whisper_cache()
    check_voxtral_cache()

    print("Cache Directories:")
    print(f"  Whisper: {os.path.join(os.getcwd(), 'models')}")
    print(f"  Voxtral: {os.path.join(os.getcwd(), 'models', 'voxtral')}")


if __name__ == "__main__":
    main()
