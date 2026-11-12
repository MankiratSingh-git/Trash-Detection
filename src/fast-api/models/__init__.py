"""Models package for Sortify.

Registers available vision model implementations in the ModelRegistry with explicit dataset provenance.
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
        description="YOLOv11s v3 Generalization (8-Class Production Default)",
        weights_path=settings.DEPLOYED_YOLO11_WEIGHTS,
        mapping_path=settings.DEPLOYED_YOLO11_MAPPING,
    ),
)

# 2. Production Candidate: v3 Generalization (Negative-Supervised 8-Class)
if settings.GENERALIZATION_V3_WEIGHTS.is_file():
    model_registry.register(
        "yolo11s_8cls_v3_generalization",
        lambda: YOLOv11Model(
            name="yolo11s_8cls_v3_generalization",
            version="11.0.0-8cls-v3",
            description="YOLOv11s Generalization v3 (8-Class Negative-Supervised Detector)",
            weights_path=settings.GENERALIZATION_V3_WEIGHTS,
            mapping_path=settings.GENERALIZATION_V3_MAPPING,
        ),
    )

# 3. Curated v2 (8-Class Curated Dataset)
if settings.CURATED_V2_WEIGHTS.is_file():
    model_registry.register(
        "yolo11s_8cls_v2_curated",
        lambda: YOLOv11Model(
            name="yolo11s_8cls_v2_curated",
            version="11.0.0-8cls-v2",
            description="YOLOv11s v2 Curated (8-Class Refined Dataset)",
            weights_path=settings.CURATED_V2_WEIGHTS,
            mapping_path=settings.CURATED_V2_MAPPING,
        ),
    )

# 4. Real Organic Augmentation (8-Class Real Organic Dataset)
if settings.REAL_ORGANIC_WEIGHTS.is_file():
    model_registry.register(
        "yolo11s_8cls_real_organic",
        lambda: YOLOv11Model(
            name="yolo11s_8cls_real_organic",
            version="11.0.0-8cls-organic",
            description="YOLOv11s Real Organic (8-Class Organic-Augmented Dataset)",
            weights_path=settings.REAL_ORGANIC_WEIGHTS,
            mapping_path=settings.REAL_ORGANIC_MAPPING,
        ),
    )

# 5. Old 8-Class Streamlined Baseline (Old 8-Class Dataset)
if settings.STREAMLINED_YOLO11S_WEIGHTS.is_file():
    model_registry.register(
        "yolo11s_8cls_v1_streamlined_old_dataset",
        lambda: YOLOv11Model(
            name="yolo11s_8cls_v1_streamlined_old_dataset",
            version="11.0.0-8cls-v1",
            description="YOLOv11s v1 Streamlined (Old 8-Class Dataset)",
            weights_path=settings.STREAMLINED_YOLO11S_WEIGHTS,
            mapping_path=settings.STREAMLINED_YOLO11S_MAPPING,
        ),
    )

# 6. Legacy 10-Class Baseline (Old 10-Class Dataset)
if settings.LEGACY_YOLO11S_10CLS_WEIGHTS.is_file():
    model_registry.register(
        "yolo11s_10cls_legacy_old_dataset",
        lambda: YOLOv11Model(
            name="yolo11s_10cls_legacy_old_dataset",
            version="11.0.0-10cls",
            description="YOLOv11s Legacy (Old 10-Class Dataset)",
            weights_path=settings.LEGACY_YOLO11S_10CLS_WEIGHTS,
            mapping_path=settings.LEGACY_YOLO11S_10CLS_MAPPING,
        ),
    )

# 7. Calibrated Medium Model (Old 10-Class Dataset)
if settings.CALIBRATED_YOLO11M_WEIGHTS.is_file():
    model_registry.register(
        "yolo11m_10cls_calibrated_old_dataset",
        lambda: YOLOv11Model(
            name="yolo11m_10cls_calibrated_old_dataset",
            version="11.0.0-calibrated-m",
            description="YOLOv11m Calibrated (Old 10-Class Dataset)",
            weights_path=settings.CALIBRATED_YOLO11M_WEIGHTS,
            mapping_path=settings.CALIBRATED_YOLO11M_MAPPING,
        ),
    )

__all__ = ["YOLOv11Model"]
