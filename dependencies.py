#!/usr/bin/env python3
"""
Dependency management and version checking for VoxScribe
"""

import sys
import subprocess
import logging
from packaging import version

logger = logging.getLogger("voxscribe")

UNIFIED_TRANSFORMERS_VERSION = "4.57.0"


def _clear_module_cache_and_refresh(modules_to_clear=None):
    """Clear module cache and refresh imports after installation."""
    import importlib

    if modules_to_clear is None:
        modules_to_clear = ["transformers"]

    # Remove specified modules from cache if they exist
    modules_to_remove = [
        name
        for name in sys.modules.keys()
        if any(name.startswith(prefix) for prefix in modules_to_clear)
    ]
    for module_name in modules_to_remove:
        del sys.modules[module_name]

    # Invalidate import caches
    importlib.invalidate_caches()
    print(f"Cleared module cache for: {', '.join(modules_to_clear)}")


def ensure_transformers_version():
    """Ensure transformers and tokenizers are at compatible versions before startup."""
    try:
        import transformers
        current_version = transformers.__version__

        if version.parse(current_version) >= version.parse(UNIFIED_TRANSFORMERS_VERSION):
            print(
                f"✓ Transformers version {current_version} meets requirement (>= {UNIFIED_TRANSFORMERS_VERSION})"
            )
            return current_version
        else:
            print(
                f"⚠ Transformers version {current_version} is below required {UNIFIED_TRANSFORMERS_VERSION}"
            )
            print("Upgrading transformers and compatible tokenizers...")
            
            # Upgrade transformers
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        f"transformers=={UNIFIED_TRANSFORMERS_VERSION}",
                        "tokenizers>=0.22,<0.23",
                        "--upgrade",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                print("✓ Transformers and tokenizers upgrade completed")
                print("Refreshing module cache...")
                
                # Clear module cache and refresh
                _clear_module_cache_and_refresh(["transformers", "tokenizers"])
                
                # Re-import and verify
                import transformers
                new_version = transformers.__version__
                if version.parse(new_version) >= version.parse(UNIFIED_TRANSFORMERS_VERSION):
                    print(f"✓ Transformers successfully upgraded to {new_version}")
                    return new_version
                else:
                    print(f"✗ Upgrade failed: still at version {new_version}")
                    return None
                    
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to upgrade transformers: {e}")
                print(f"stdout: {e.stdout}")
                print(f"stderr: {e.stderr}")
                return None
                
    except ImportError as e:
        # Handle tokenizers compatibility error specifically
        if "tokenizers" in str(e):
            print("⚠ Tokenizers compatibility issue detected, fixing...")
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        f"transformers=={UNIFIED_TRANSFORMERS_VERSION}",
                        "tokenizers>=0.22,<0.23",
                        "--upgrade",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                print("Transformers and tokenizers compatibility fixed")
                
                # Clear module cache and refresh
                _clear_module_cache_and_refresh(["transformers", "tokenizers"])
                
                # Import and verify
                import transformers
                new_version = transformers.__version__
                print(f"Transformers installed at version {new_version}")
                return new_version
                
            except subprocess.CalledProcessError as e:
                print(f"x Failed to fix tokenizers compatibility: {e}")
                return None
        else:
            print("Transformers not installed, installing...")
        
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    f"transformers=={UNIFIED_TRANSFORMERS_VERSION}",
                    "tokenizers>=0.22,<0.23",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            print("✓ Transformers and tokenizers installation completed")
            
            # Import and verify
            import transformers
            new_version = transformers.__version__
            print(f"✓ Transformers installed at version {new_version}")
            return new_version
            
        except subprocess.CalledProcessError as e:
            print(f"x Failed to install transformers: {e}")
            return None


def _check_voxtral_support():
    """Internal function to check Voxtral support."""
    try:
        import transformers

        version_str = transformers.__version__
        supported = version_str >= UNIFIED_TRANSFORMERS_VERSION

        if not supported:
            logger.warning(
                f"Voxtral not supported: transformers version {version_str} < {UNIFIED_TRANSFORMERS_VERSION}"
            )
        else:
            logger.info(f"Voxtral supported: transformers version {version_str}")

        return supported
    except ImportError as e:
        logger.warning(f"Voxtral not supported: transformers not installed - {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking Voxtral support: {e}")
        return False


def _check_nemo_support():
    """Internal function to check NeMo support."""
    try:
        # Check transformers version first
        import transformers

        version_str = transformers.__version__
        supported = version_str >= UNIFIED_TRANSFORMERS_VERSION

        if not supported:
            logger.warning(
                f"NeMo not supported: transformers version {version_str} < {UNIFIED_TRANSFORMERS_VERSION}"
            )
            return False

        # Try to import NeMo components
        import nemo.collections.asr as nemo_asr
        from nemo.collections.speechlm2.models import SALM

        logger.info(f"NeMo toolkit supported: transformers version {version_str}")
        return True
    except ImportError as e:
        logger.warning(f"NeMo not supported: NeMo toolkit not installed - {e}")
        return False
    except Exception as e:
        # Handle resolver registration errors specifically
        if "resolver" in str(e) and "already registered" in str(e):
            logger.info(
                "NeMo toolkit supported (resolver already registered from previous import)"
            )
            return True
        else:
            logger.error(f"Error checking NeMo support: {e}")
            return False


def _check_granite_support():
    """Internal function to check Granite Speech support."""
    try:
        # Check transformers version first
        import transformers

        version_str = transformers.__version__
        supported = version_str >= UNIFIED_TRANSFORMERS_VERSION

        if not supported:
            logger.warning(
                f"Granite not supported: transformers version {version_str} < {UNIFIED_TRANSFORMERS_VERSION}"
            )
            return False

        # Try to import required components
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        import torch
        import torchaudio

        logger.info(f"Granite Speech supported: transformers version {version_str}")
        return True
    except ImportError as e:
        logger.warning(f"Granite not supported: required dependencies not installed - {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking Granite support: {e}")
        return False


def get_transformers_version():
    """Get transformers version if available."""
    try:
        import transformers
        return transformers.__version__
    except ImportError:
        return None
