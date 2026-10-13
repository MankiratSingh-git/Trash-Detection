"""Abstract base class interface for all computer vision models in Sortify."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from .schemas import ModelMetadata, PredictionResponse


class BaseVisionModel(ABC):
    """Abstract Base Model for vision tasks (detection, classification, segmentation).

    Subclasses must implement:
    - load(): Load weights into memory.
    - predict(): Run inference on input image and return PredictionResponse.
    - annotate(): Run inference and return (PredictionResponse, annotated_image_np).
    - get_metadata(): Return ModelMetadata description.
    """

    def __init__(self, name: str, version: str = "1.0.0", description: str = ""):
        self.name = name
        self.version = version
        self.description = description
        self.is_loaded: bool = False

    @abstractmethod
    def load(self, weights_path: Optional[str] = None) -> None:
        """Load model weights into memory.

        Args:
            weights_path: Optional explicit path to model weights file.
        """
        pass

    @abstractmethod
    def predict(
        self,
        image: Union[np.ndarray, Image.Image, bytes],
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        return_image: bool = False,
        **kwargs: Any,
    ) -> PredictionResponse:
        """Execute inference on a single image.

        Args:
            image: Input image as numpy array (BGR/RGB), PIL Image, or raw bytes.
            conf: Confidence threshold (0.0 to 1.0).
            iou: NMS IoU threshold (0.0 to 1.0).
            imgsz: Input resolution.
            return_image: If True, include base64 annotated image in response.
            **kwargs: Extra model-specific parameters.

        Returns:
            PredictionResponse containing structured detections and summary.
        """
        pass

    @abstractmethod
    def annotate(
        self,
        image: Union[np.ndarray, Image.Image, bytes],
        conf: float = 0.25,
        iou: float = 0.45,
        imgsz: int = 640,
        **kwargs: Any,
    ) -> Tuple[PredictionResponse, np.ndarray]:
        """Execute inference and return the annotated image as a numpy array.

        Args:
            image: Input image.
            conf: Confidence threshold.
            iou: IoU threshold.
            imgsz: Input resolution.
            **kwargs: Extra parameters.

        Returns:
            Tuple of (PredictionResponse, annotated_image_bgr_numpy).
        """
        pass

    @abstractmethod
    def get_metadata(self) -> ModelMetadata:
        """Return metadata describing model architecture, classes, and status."""
        pass

    def unload(self) -> None:
        """Unload model from memory if supported."""
        self.is_loaded = False
