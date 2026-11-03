"""Models package for Sortify.

Registers available vision model implementations in the ModelRegistry.
"""

from core.config import settings
from core.registry import model_registry
from models.yolov11.service import YOLOv11Model

# 1. Default Production Model (Deployed 8-Class Streamlined YOLOv11s)
model_registry.register(
    "yolov11",
    lambda: YOLOv11Model(
        name="yolov11",
        version="11.0.0-prod",
        description="YOLOv11s Streamlined (8-Class Production - Deployed)",
        weights_path=settings.DEPLOYED_YOLO11_WEIGHTS,
        mapping_path=settings.DEPLOYED_YOLO11_MAPPING,
    ),
)

# 2. Explicit Streamlined 8-Class Model
model_registry.register(
    "yolov11s_8cls",
    lambda: YOLOv11Model(
        name="yolov11s_8cls",
        version="11.0.0-8cls",
        description="YOLOv11s Streamlined (8-Class Waste Detector)",
        weights_path=settings.STREAMLINED_YOLO11S_WEIGHTS,
        mapping_path=settings.STREAMLINED_YOLO11S_MAPPING,
    ),
)

# 3. Legacy 10-Class Model
model_registry.register(
    "yolov11s_10cls",
    lambda: YOLOv11Model(
        name="yolov11s_10cls",
        version="11.0.0-10cls",
        description="YOLOv11s Legacy (10-Class Waste Detector)",
        weights_path=settings.LEGACY_YOLO11S_10CLS_WEIGHTS,
        mapping_path=settings.LEGACY_YOLO11S_10CLS_MAPPING,
    ),
)

# 4. Calibrated Medium Model (if weights available)
if settings.CALIBRATED_YOLO11M_WEIGHTS.is_file():
    model_registry.register(
        "yolov11m_calibrated",
        lambda: YOLOv11Model(
            name="yolov11m_calibrated",
            version="11.0.0-calibrated-m",
            description="YOLOv11m Calibrated (10-Class Medium Detector)",
            weights_path=settings.CALIBRATED_YOLO11M_WEIGHTS,
            mapping_path=settings.CALIBRATED_YOLO11M_MAPPING,
        ),
    )

__all__ = ["YOLOv11Model"]
