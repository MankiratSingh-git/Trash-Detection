"""YOLOv11 Model Package for Sortify."""

from .service import YOLOv11Model
from .categories import (
    SORTIFY_CLASSES,
    SORTIFY_CATEGORIES,
    get_class_info,
)

__all__ = [
    "YOLOv11Model",
    "SORTIFY_CLASSES",
    "SORTIFY_CATEGORIES",
    "get_class_info",
]
