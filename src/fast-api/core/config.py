"""Application Configuration for Sortify FastAPI Service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

# Determine project root directory (Sortify-ML)
# src/fast-api/core/config.py -> parents[3] is project root
CURRENT_FILE = Path(__file__).resolve()
FAST_API_DIR = CURRENT_FILE.parents[1]
SRC_DIR = FAST_API_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

class Settings:
    """Sortify backend service configuration settings."""

    # Project paths
    ROOT_DIR: Path = PROJECT_ROOT
    SRC_DIR: Path = SRC_DIR
    FAST_API_DIR: Path = FAST_API_DIR
    FRONTEND_DIR: Path = SRC_DIR / "frontend"

    # Weights and mappings for available models
    DEPLOYED_YOLO11_WEIGHTS: Path = (
        PROJECT_ROOT / "YOLOv11" / "model" / "best.pt"
    )
    DEPLOYED_YOLO11_MAPPING: Path = (
        PROJECT_ROOT / "YOLOv11" / "model" / "sortify_class_mapping.json"
    )

    STREAMLINED_YOLO11S_WEIGHTS: Path = (
        PROJECT_ROOT / "YOLOv11" / "runs" / "sortify_yolo11s_8cls_streamlined" / "weights" / "best.pt"
    )
    STREAMLINED_YOLO11S_MAPPING: Path = (
        PROJECT_ROOT / "YOLOv11" / "runs" / "sortify_yolo11s_8cls_streamlined" / "sortify_class_mapping.json"
    )

    LEGACY_YOLO11S_10CLS_WEIGHTS: Path = (
        PROJECT_ROOT / "YOLOv11" / "runs" / "sortify_yolo11s_10cls" / "weights" / "best.pt"
    )
    LEGACY_YOLO11S_10CLS_MAPPING: Path = (
        PROJECT_ROOT / "YOLOv11" / "runs" / "sortify_yolo11s_10cls" / "sortify_class_mapping.json"
    )

    CALIBRATED_YOLO11M_WEIGHTS: Path = (
        PROJECT_ROOT / "YOLOv11" / "runs" / "sortify_yolo11m_calibrated_aug" / "weights" / "best.pt"
    )
    CALIBRATED_YOLO11M_MAPPING: Path = (
        PROJECT_ROOT / "YOLOv11" / "runs" / "sortify_yolo11m_calibrated_aug" / "sortify_class_mapping.json"
    )

    BASE_YOLO11_WEIGHTS: Path = PROJECT_ROOT / "yolo11s.pt"

    @property
    def DEFAULT_YOLO11_WEIGHTS(self) -> Path:
        """Resolve best available default YOLOv11 weights."""
        env_weights = os.getenv("YOLO11_WEIGHTS_PATH")
        if env_weights and Path(env_weights).is_file():
            return Path(env_weights)
        if self.DEPLOYED_YOLO11_WEIGHTS.is_file():
            return self.DEPLOYED_YOLO11_WEIGHTS
        if self.STREAMLINED_YOLO11S_WEIGHTS.is_file():
            return self.STREAMLINED_YOLO11S_WEIGHTS
        if self.LEGACY_YOLO11S_10CLS_WEIGHTS.is_file():
            return self.LEGACY_YOLO11S_10CLS_WEIGHTS
        if self.BASE_YOLO11_WEIGHTS.is_file():
            return self.BASE_YOLO11_WEIGHTS
        return self.DEPLOYED_YOLO11_WEIGHTS

    # Inference defaults
    DEFAULT_CONF_THRESHOLD: float = float(os.getenv("DEFAULT_CONF_THRESHOLD", "0.25"))
    DEFAULT_IOU_THRESHOLD: float = float(os.getenv("DEFAULT_IOU_THRESHOLD", "0.45"))
    DEFAULT_IMGSZ: int = int(os.getenv("DEFAULT_IMGSZ", "640"))
    DEFAULT_DEVICE: str = os.getenv("DEVICE", "")  # Empty string allows auto-detection (cuda/cpu)

    # Server settings
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    RELOAD: bool = os.getenv("RELOAD", "true").lower() in ("true", "1", "yes")

    # CORS settings
    CORS_ORIGINS: List[str] = [
        "http://localhost",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:8000",
        "*",
    ]


settings = Settings()
