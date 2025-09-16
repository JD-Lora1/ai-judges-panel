#!/usr/bin/env python3
"""
Railway startup script for AI Judges Panel
"""

import os
import sys
import uvicorn

def main():
    """Start the application with proper configuration"""
    
    # Set environment variables
    os.environ.setdefault("ENVIRONMENT", "production")
    os.environ.setdefault("PYTHONPATH", ".")
    
    # Get port from Railway
    port = int(os.environ.get("PORT", 8000))
    
    print(f"🚀 Starting AI Judges Panel (GPT-2) on port {port}")
    print(f"📍 Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    print(f"🤖 Model: OpenAI GPT-2 (124M parameters)")
    
    # Start with uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
        timeout_keep_alive=120,
        access_log=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
