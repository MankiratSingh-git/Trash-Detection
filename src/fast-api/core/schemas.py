"""Pydantic request and response schemas for computer vision models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Bounding box coordinates and dimensions."""

    x1: float = Field(..., description="Top-left X coordinate in pixels")
    y1: float = Field(..., description="Top-left Y coordinate in pixels")
    x2: float = Field(..., description="Bottom-right X coordinate in pixels")
    y2: float = Field(..., description="Bottom-right Y coordinate in pixels")
    width: float = Field(..., description="Box width in pixels")
    height: float = Field(..., description="Box height in pixels")
    normalized: Optional[Dict[str, float]] = Field(
        None,
        description="Normalized coordinates (0.0 to 1.0): x1, y1, x2, y2, width, height",
    )


class DetectionResult(BaseModel):
    """Single object detection result."""

    class_id: int = Field(..., description="Class identifier integer")
    class_name: str = Field(..., description="Name of detected class (e.g., 'plastic_bottle')")
    confidence: float = Field(..., description="Prediction confidence score between 0.0 and 1.0")
    category: Optional[str] = Field(
        None,
        description="Sortify disposal category (e.g., 'Recyclable', 'Organic', 'General waste')",
    )
    guidance: Optional[str] = Field(
        None,
        description="Disposal guidance or instructions for this item",
    )
    color: Optional[str] = Field(
        None,
        description="Hex color code associated with the category/class",
    )
    box: BoundingBox = Field(..., description="Bounding box location in image")


class InferenceSummary(BaseModel):
    """High-level summary of model inference."""

    total_detections: int = Field(0, description="Total number of items detected")
    category_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown of detection count per Sortify disposal category",
    )
    inference_time_ms: float = Field(..., description="Inference latency in milliseconds")
    image_width: int = Field(..., description="Width of input image in pixels")
    image_height: int = Field(..., description="Height of input image in pixels")


class PredictionResponse(BaseModel):
    """Full prediction response payload."""

    success: bool = Field(True, description="Whether the inference succeeded")
    model_name: str = Field(..., description="Identifier of the model used")
    model_version: str = Field("1.0.0", description="Version of the model checkpoint")
    summary: InferenceSummary = Field(..., description="Inference performance and count summary")
    detections: List[DetectionResult] = Field(
        default_factory=list,
        description="List of detected objects with bounding boxes and guidance",
    )
    annotated_image_base64: Optional[str] = Field(
        None,
        description="Base64 encoded data URI of the image with rendered bounding boxes",
    )


class BatchPredictionResponse(BaseModel):
    """Batch prediction response payload."""

    success: bool = True
    model_name: str
    total_images: int
    total_inference_time_ms: float
    results: List[PredictionResponse]


class ModelMetadata(BaseModel):
    """Metadata describing a registered vision model."""

    name: str = Field(..., description="Model unique name identifier")
    version: str = Field(..., description="Model version or tag")
    description: str = Field(..., description="Brief description of the model architecture")
    task: str = Field("object-detection", description="Task type (e.g. object-detection, classification)")
    weights_path: str = Field(..., description="Path to checkpoint weights file")
    device: str = Field(..., description="Device used for inference (cpu, cuda, mps)")
    is_loaded: bool = Field(..., description="Whether weights are currently loaded in memory")
    num_classes: int = Field(..., description="Number of distinct classes")
    classes: Dict[int, str] = Field(default_factory=dict, description="Mapping of class IDs to class names")
    categories: Optional[List[str]] = Field(
        default=None,
        description="Disposal categories supported by the model (if applicable)",
    )


class HealthResponse(BaseModel):
    """API health status and model availability."""

    status: str = "healthy"
    version: str = "1.0.0"
    available_models: List[str] = Field(default_factory=list)
    loaded_models: List[str] = Field(default_factory=list)
