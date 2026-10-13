"""YOLOv11 Dedicated API Endpoints for Sortify Waste Detection."""

from __future__ import annotations

import io
import time
from typing import List, Optional
import cv2
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field

from core.config import settings
from core.registry import model_registry
from core.schemas import (
    BatchPredictionResponse,
    ModelMetadata,
    PredictionResponse,
)

router = APIRouter(prefix="/api/v1/yolov11", tags=["YOLOv11 Waste Detector"])


def get_yolo11_model():
    """Dependency helper to fetch active YOLOv11 model instance."""
    try:
        return model_registry.get_model("yolov11", load_if_needed=True)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize YOLOv11 model: {str(exc)}",
        ) from exc


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Detect waste items and get disposal recommendations",
    response_description="Detailed detections with bounding boxes and Sortify categories",
)
async def predict_waste(
    file: UploadFile = File(..., description="Image file (JPG, PNG, WebP)"),
    conf: float = Form(
        default=settings.DEFAULT_CONF_THRESHOLD,
        ge=0.01,
        le=1.0,
        description="Confidence threshold (0.01 to 1.0)",
    ),
    iou: float = Form(
        default=settings.DEFAULT_IOU_THRESHOLD,
        ge=0.01,
        le=1.0,
        description="NMS IoU threshold (0.01 to 1.0)",
    ),
    imgsz: int = Form(
        default=settings.DEFAULT_IMGSZ,
        ge=160,
        le=1280,
        description="Inference image resolution",
    ),
    return_image: bool = Form(
        default=False,
        description="If True, includes base64-encoded annotated image in response",
    ),
) -> PredictionResponse:
    """Upload an image to run Sortify YOLOv11 waste detection.

    Returns:
    - Bounding boxes (absolute & normalized)
    - Classification label and confidence score
    - Sortify category (Recyclable, General waste, Organic, E-waste, Hazardous)
    - Specific disposal instructions (e.g., 'Clean dry paper bin')
    - Inference latency and aggregate category count summary
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        # Still attempt decoding if extension looks like an image
        valid_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        if not file.filename or not file.filename.lower().endswith(valid_exts):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{file.filename}' is not a supported image format.",
            )

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        model = get_yolo11_model()
        return model.predict(
            image=contents,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            return_image=return_image,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}",
        ) from exc


@router.post(
    "/annotate",
    summary="Detect waste and return rendered annotated image directly",
    responses={
        200: {
            "content": {"image/jpeg": {}},
            "description": "Annotated image with bounding boxes, category tags, and guidance.",
        }
    },
)
async def annotate_image(
    file: UploadFile = File(..., description="Image file"),
    conf: float = Form(
        default=settings.DEFAULT_CONF_THRESHOLD,
        ge=0.01,
        le=1.0,
    ),
    iou: float = Form(
        default=settings.DEFAULT_IOU_THRESHOLD,
        ge=0.01,
        le=1.0,
    ),
    imgsz: int = Form(default=settings.DEFAULT_IMGSZ, ge=160, le=1280),
) -> Response:
    """Upload an image and receive the annotated JPEG directly."""
    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty.",
            )

        model = get_yolo11_model()
        _, annotated_cv = model.annotate(
            image=contents,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
        )

        success, buffer = cv2.imencode(".jpg", annotated_cv)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to encode annotated image to JPEG.",
            )

        return Response(content=buffer.tobytes(), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Annotation error: {str(exc)}",
        ) from exc


@router.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    summary="Batch inference on multiple images",
)
async def predict_batch(
    files: List[UploadFile] = File(..., description="Multiple image files"),
    conf: float = Form(default=settings.DEFAULT_CONF_THRESHOLD, ge=0.01, le=1.0),
    iou: float = Form(default=settings.DEFAULT_IOU_THRESHOLD, ge=0.01, le=1.0),
    imgsz: int = Form(default=settings.DEFAULT_IMGSZ, ge=160, le=1280),
) -> BatchPredictionResponse:
    """Run YOLOv11 waste detection over a batch of uploaded images."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided in request.",
        )

    model = get_yolo11_model()
    start_total = time.perf_counter()
    results: List[PredictionResponse] = []

    for file in files:
        try:
            contents = await file.read()
            if len(contents) == 0:
                continue
            pred = model.predict(
                image=contents,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                return_image=False,
            )
            results.append(pred)
        except Exception as exc:
            # Continue processing remaining batch images
            continue

    total_time_ms = round((time.perf_counter() - start_total) * 1000, 2)

    return BatchPredictionResponse(
        success=True,
        model_name=model.name,
        total_images=len(results),
        total_inference_time_ms=total_time_ms,
        results=results,
    )


@router.get(
    "/info",
    response_model=ModelMetadata,
    summary="Get YOLOv11 model metadata, class mapping, and status",
)
async def get_model_info() -> ModelMetadata:
    """Retrieve metadata, weights path, classes, and categories for YOLOv11."""
    model = get_yolo11_model()
    return model.get_metadata()


class ReloadModelRequest(BaseModel):
    weights_path: Optional[str] = Field(
        None, description="Optional custom weights path to load"
    )
    device: Optional[str] = Field(
        None, description="Optional compute device (cpu, cuda, etc.)"
    )


@router.post(
    "/reload",
    summary="Reload YOLOv11 model weights or switch checkpoint",
)
async def reload_model(request: Optional[ReloadModelRequest] = None):
    """Force reload of the YOLOv11 model checkpoint."""
    try:
        model = get_yolo11_model()
        weights = request.weights_path if request else None
        if request and request.device:
            model.device = request.device
        model.load(weights_path=weights)
        return {
            "success": True,
            "message": f"Successfully reloaded YOLOv11 weights from {model.weights_path}",
            "device": model.device or "auto",
            "weights_path": str(model.weights_path),
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {str(exc)}",
        ) from exc
