#!/usr/bin/env python3
"""
SpeechHub startup script
"""

import uvicorn
import os
from pathlib import Path

if __name__ == "__main__":
    # Ensure public directory exists
    public_dir = Path("public")
    if not public_dir.exists():
        print("❌ Public directory not found!")
        print("Please make sure the public/ folder with HTML, CSS, and JS files exists.")
        exit(1)
    
    # Check required files
    required_files = ["public/index.html", "public/styles.css", "public/app.js"]
    missing_files = [f for f in required_files if not Path(f).exists()]
    
    if missing_files:
        print("❌ Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        exit(1)
    
    print("🎤 Starting SpeechHub server...")
    print("📁 Serving static files from: public/")
    print("🌐 API available at: http://localhost:8000/api/")
    print("🎯 Web interface at: http://localhost:8000/")
    print()
    
    # Start the server
    uvicorn.run(
        "backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[".", "public"],
        log_level="info"
    )