"""Main FastAPI application for Sortify Multi-Model ML Service."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure this package directory is on sys.path for direct script execution
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Import models package to trigger model registration
import models  # noqa: F401
from core.config import settings
from core.registry import model_registry
from routers.health import router as health_router
from routers.yolov11 import router as yolov11_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("sortify.api")


def create_app() -> FastAPI:
    """Build and configure the Sortify FastAPI application."""
    app = FastAPI(
        title="Sortify ML API",
        description=(
            "Robust multi-model computer vision backend for Sortify waste detection and disposal guidance.\n\n"
            "Features:\n"
            "- **YOLOv11**: 10-class waste detection mapped to 5 Sortify disposal categories\n"
            "- **Extensible Architecture**: Easily add future models (YOLOv8, ConvNeXt, CLIP, etc.) "
            "using modular services and routers\n"
            "- **Interactive Test UI**: Hosted frontend for live image testing, confidence adjustment, and visual inspection."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API Routers
    app.include_router(health_router)
    app.include_router(yolov11_router)

    # Mount frontend static directory if present
    frontend_dir = settings.FRONTEND_DIR
    if frontend_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_index():
            """Serve test frontend interface at root."""
            index_file = frontend_dir / "index.html"
            if index_file.is_file():
                return FileResponse(index_file)
            return {"message": "Sortify ML API running. Visit /docs for API documentation."}

    @app.on_event("startup")
    async def on_startup():
        logger.info("Sortify ML API starting up...")
        logger.info("Available models: %s", model_registry.list_available_models())
        logger.info("YOLOv11 default weights: %s", settings.DEFAULT_YOLO11_WEIGHTS)
        logger.info("API documentation available at http://%s:%d/docs", settings.HOST, settings.PORT)

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        app_dir=str(CURRENT_DIR),
    )
