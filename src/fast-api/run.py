"""Simple runner script for Sortify FastAPI."""

from __future__ import annotations

import sys
from pathlib import Path

# Add fast-api folder to sys.path
FAST_API_DIR = Path(__file__).resolve().parent
if str(FAST_API_DIR) not in sys.path:
    sys.path.insert(0, str(FAST_API_DIR))

import uvicorn
from core.config import settings

if __name__ == "__main__":
    print(f"Starting Sortify ML API at http://localhost:{settings.PORT}")
    print(f"Interactive Docs: http://localhost:{settings.PORT}/docs")
    print(f"Interactive Frontend: http://localhost:{settings.PORT}/")
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        app_dir=str(FAST_API_DIR),
    )
