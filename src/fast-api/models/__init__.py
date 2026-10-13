"""Models package for Sortify.

Registers available vision model implementations in the ModelRegistry.
"""

from core.registry import model_registry
from models.yolov11.service import YOLOv11Model

# Register YOLOv11 factory in central registry
model_registry.register("yolov11", lambda: YOLOv11Model())

__all__ = ["YOLOv11Model"]
