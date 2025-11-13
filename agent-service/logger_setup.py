"""
Logger setup for Agent Service
"""
import logging
import sys


def setup_logger(name: str = "agent-service", level: int = logging.INFO):
    """Setup and configure logger"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Add handler
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger
