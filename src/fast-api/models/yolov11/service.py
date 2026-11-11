"""YOLOv11 Inference Service implementing BaseVisionModel interface."""

from __future__ import annotations

import base64
import io
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from core.base_model import BaseVisionModel
from core.config import settings
from core.schemas import (
    BoundingBox,
    DetectionResult,
    InferenceSummary,
    ModelMetadata,
    PredictionResponse,
)
from models.yolov11.categories import (
    CategoryInfo,
    SORTIFY_CATEGORIES,
    SORTIFY_CLASSES,
    get_category_color,
    get_class_info,
)

logger = logging.getLogger("sortify.yolov11")


class YOLOv11Model(BaseVisionModel):
    """Sortify YOLOv11 Object Detection & Waste Categorization Service.

    Supports configurable checkpoints, automated class mapping resolution,
    5 Sortify disposal bins, and GPU/CPU device dispatch.
    """

    def __init__(
        self,
        name: str = "yolov11",
        version: str = "11.0.0-sortify",
        description: str = "Ultralytics YOLO11 model for Sortify waste detection.",
        weights_path: Optional[Union[str, Path]] = None,
        mapping_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            version=version,
            description=description,
        )
        self.weights_path: Path = Path(weights_path) if weights_path else settings.DEFAULT_YOLO11_WEIGHTS
        self.mapping_path: Optional[Path] = Path(mapping_path) if mapping_path else None
        self.device: str = device or settings.DEFAULT_DEVICE
        self.model = None
        self._class_names: Dict[int, str] = {}
        self._class_info_map: Dict[int, CategoryInfo] = {}

        # Pre-resolve class mapping metadata from JSON without loading heavy weights
        self._resolve_class_mapping()

    def load(self, weights_path: Optional[str] = None) -> None:
        """Load YOLO11 model weights and resolve category taxonomy."""
        if weights_path:
            self.weights_path = Path(weights_path)

        if not self.weights_path.is_file():
            # If default trained weights not found, fallback to root base weights
            fallback = settings.BASE_YOLO11_WEIGHTS
            if fallback.is_file():
                logger.warning(
                    "Specified weights not found at %s. Falling back to %s",
                    self.weights_path,
                    fallback,
                )
                self.weights_path = fallback
            else:
                raise FileNotFoundError(
                    f"YOLOv11 weights not found at {self.weights_path} or {fallback}"
                )

        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics package is missing. Install with 'pip install ultralytics'."
            ) from exc

        logger.info("[%s] Loading checkpoint: %s", self.name, self.weights_path)
        self.model = YOLO(str(self.weights_path))

        # Populate class names from model checkpoint
        if hasattr(self.model, "names") and isinstance(self.model.names, dict):
            self._class_names = {int(k): str(v) for k, v in self.model.names.items()}
        else:
            self._class_names = {cid: info["name"] for cid, info in SORTIFY_CLASSES.items()}

        # Resolve category mapping JSON
        self._resolve_class_mapping()

        self.is_loaded = True
        logger.info(
            "[%s] Loaded successfully. Classes: %d (%s), Device: %s",
            self.name,
            len(self._class_names),
            ", ".join(list(self._class_names.values())[:4]) + ("..." if len(self._class_names) > 4 else ""),
            self.device or "auto",
        )

    def _resolve_class_mapping(self) -> None:
        """Resolve per-class Sortify disposal categories and guidance."""
        self._class_info_map.clear()

        # Potential mapping file locations
        candidates: List[Path] = []
        if self.mapping_path:
            candidates.append(self.mapping_path)
        candidates.append(self.weights_path.parent / "sortify_class_mapping.json")
        candidates.append(self.weights_path.parent.parent / "sortify_class_mapping.json")
        candidates.append(settings.ROOT_DIR / "YOLOv11" / "model" / "sortify_class_mapping.json")

        loaded_from_file = False
        for cand in candidates:
            if cand.is_file():
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for cid_str, info_dict in data.items():
                        cid = int(cid_str)
                        cname = info_dict.get("name", self._class_names.get(cid, f"class_{cid}"))
                        if cid not in self._class_names:
                            self._class_names[cid] = cname
                        category = info_dict.get("category", "General waste")
                        guidance = info_dict.get("guidance", "Inspect item carefully")
                        color, bgr = get_category_color(category, cname)
                        self._class_info_map[cid] = {
                            "name": cname,
                            "category": category,
                            "guidance": guidance,
                            "color": color,
                            "bgr": bgr,
                        }
                    loaded_from_file = True
                    logger.info("[%s] Loaded class mapping from %s", self.name, cand)
                    break
                except Exception as exc:
                    logger.warning("[%s] Failed loading mapping from %s: %s", self.name, cand, exc)

        # Ensure all model classes have full info
        for cid, cname in self._class_names.items():
            if cid not in self._class_info_map:
                self._class_info_map[cid] = get_class_info(cid, cname)

    def _ensure_loaded(self) -> None:
        """Helper to ensure weights are loaded before inference."""
        if not self.is_loaded or self.model is None:
            self.load()

    def _prepare_image(
        self, image: Union[np.ndarray, Image.Image, bytes]
    ) -> Tuple[np.ndarray, int, int]:
        """Convert input image into a BGR numpy array and return (img, width, height)."""
        if isinstance(image, bytes):
            img_array = np.frombuffer(image, dtype=np.uint8)
            cv_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if cv_img is None:
                raise ValueError("Could not decode image from provided bytes.")
        elif isinstance(image, Image.Image):
            rgb_np = np.array(image.convert("RGB"))
            cv_img = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            cv_img = image.copy()
            if len(cv_img.shape) == 2:  # Grayscale
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
            elif cv_img.shape[2] == 4:  # RGBA
                cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        height, width = cv_img.shape[:2]
        return cv_img, width, height

    def predict(
        self,
        image: Union[np.ndarray, Image.Image, bytes],
        conf: Optional[float] = None,
        iou: float = 0.45,
        imgsz: int = 640,
        return_image: bool = False,
        **kwargs: Any,
    ) -> PredictionResponse:
        """Run YOLOv11 waste detection on an image with calibrated confidence gating."""
        self._ensure_loaded()
        cv_img, orig_width, orig_height = self._prepare_image(image)

        start_time = time.perf_counter()

        # Resolve lowest class threshold so model doesn't truncate candidate detections early
        min_class_conf = (
            min(settings.CLASS_MIN_CONFIDENCE.values())
            if settings.CLASS_MIN_CONFIDENCE
            else settings.DEFAULT_CONF_THRESHOLD
        )
        base_yolo_conf = (
            min(conf, min_class_conf)
            if conf is not None
            else min(settings.DEFAULT_CONF_THRESHOLD, min_class_conf)
        )

        predict_kwargs: Dict[str, Any] = {
            "source": cv_img,
            "conf": base_yolo_conf,
            "iou": iou,
            "imgsz": imgsz,
            "verbose": False,
        }
        if self.device:
            predict_kwargs["device"] = self.device

        results = self.model.predict(**predict_kwargs)
        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        detections: List[DetectionResult] = []
        category_counts: Dict[str, int] = {cat: 0 for cat in SORTIFY_CATEGORIES}

        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes

            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()

                for box_coord, score, cls_id in zip(xyxy, confs, classes):
                    cid = int(cls_id)
                    score_val = float(score)

                    class_name = self._class_names.get(cid, f"class_{cid}")

                    # Class-specific confidence calibration:
                    # Prunes low-confidence noise & hallucinations on complex/natural backgrounds
                    class_min_conf = settings.CLASS_MIN_CONFIDENCE.get(
                        class_name, settings.DEFAULT_CONF_THRESHOLD
                    )
                    effective_threshold = (
                        max(conf, class_min_conf)
                        if conf is not None
                        else class_min_conf
                    )
                    if score_val < effective_threshold:
                        continue

                    x1, y1, x2, y2 = [float(v) for v in box_coord]

                    # Clamp pixel bounds
                    x1 = max(0.0, min(x1, float(orig_width)))
                    y1 = max(0.0, min(y1, float(orig_height)))
                    x2 = max(0.0, min(x2, float(orig_width)))
                    y2 = max(0.0, min(y2, float(orig_height)))
                    bw = max(0.0, x2 - x1)
                    bh = max(0.0, y2 - y1)

                    normalized = {
                        "x1": round(x1 / orig_width, 4) if orig_width > 0 else 0.0,
                        "y1": round(y1 / orig_height, 4) if orig_height > 0 else 0.0,
                        "x2": round(x2 / orig_width, 4) if orig_width > 0 else 0.0,
                        "y2": round(y2 / orig_height, 4) if orig_height > 0 else 0.0,
                        "width": round(bw / orig_width, 4) if orig_width > 0 else 0.0,
                        "height": round(bh / orig_height, 4) if orig_height > 0 else 0.0,
                    }

                    class_name = self._class_names.get(cid, f"class_{cid}")
                    info = self._class_info_map.get(cid) or get_class_info(
                        cid, class_name, self._class_info_map
                    )
                    cat_name = info["category"]

                    if cat_name in category_counts:
                        category_counts[cat_name] += 1
                    else:
                        category_counts[cat_name] = 1

                    bbox = BoundingBox(
                        x1=round(x1, 1),
                        y1=round(y1, 1),
                        x2=round(x2, 1),
                        y2=round(y2, 1),
                        width=round(bw, 1),
                        height=round(bh, 1),
                        normalized=normalized,
                    )

                    detections.append(
                        DetectionResult(
                            class_id=cid,
                            class_name=info["name"],
                            confidence=round(score_val, 4),
                            category=cat_name,
                            guidance=info["guidance"],
                            color=info["color"],
                            box=bbox,
                        )
                    )

        # Filter out 0 counts for clean summary response
        clean_category_counts = {k: v for k, v in category_counts.items() if v > 0}

        summary = InferenceSummary(
            total_detections=len(detections),
            category_counts=clean_category_counts,
            inference_time_ms=inference_time_ms,
            image_width=orig_width,
            image_height=orig_height,
        )

        annotated_b64: Optional[str] = None
        if return_image:
            annotated_cv = self._draw_annotations(cv_img, detections)
            success_enc, buffer = cv2.imencode(".jpg", annotated_cv)
            if success_enc:
                encoded = base64.b64encode(buffer).decode("utf-8")
                annotated_b64 = f"data:image/jpeg;base64,{encoded}"

        return PredictionResponse(
            success=True,
            model_name=self.name,
            model_version=self.version,
            summary=summary,
            detections=detections,
            annotated_image_base64=annotated_b64,
        )

    def annotate(
        self,
        image: Union[np.ndarray, Image.Image, bytes],
        conf: Optional[float] = None,
        iou: float = 0.45,
        imgsz: int = 640,
        **kwargs: Any,
    ) -> Tuple[PredictionResponse, np.ndarray]:
        """Run detection and return the annotated OpenCV image (BGR)."""
        cv_img, _, _ = self._prepare_image(image)
        pred_resp = self.predict(
            image=cv_img,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            return_image=False,
            **kwargs,
        )
        annotated_cv = self._draw_annotations(cv_img, pred_resp.detections)
        return pred_resp, annotated_cv

    def _draw_annotations(
        self, image: np.ndarray, detections: List[DetectionResult]
    ) -> np.ndarray:
        """Render crisp bounding boxes, category tags, and guidance on image."""
        annotated = image.copy()
        for det in detections:
            x1, y1, x2, y2 = (
                int(det.box.x1),
                int(det.box.y1),
                int(det.box.x2),
                int(det.box.y2),
            )
            cid = det.class_id
            info = self._class_info_map.get(cid) or get_class_info(
                cid, det.class_name, self._class_info_map
            )
            bgr_color = info["bgr"]

            # Draw outer rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr_color, 2)

            # Label text: e.g. "plastic_bottle [Recyclable] 94%"
            label = f"{det.class_name} [{det.category}] {int(det.confidence * 100)}%"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            thickness = 1
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, thickness
            )

            # Draw label background filled rectangle
            tag_y1 = max(0, y1 - text_h - baseline - 6)
            tag_y2 = y1
            cv2.rectangle(
                annotated,
                (x1, tag_y1),
                (x1 + text_w + 10, tag_y2),
                bgr_color,
                -1,
            )

            # Draw text
            cv2.putText(
                annotated,
                label,
                (x1 + 5, tag_y2 - baseline - 2),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return annotated

    def get_metadata(self) -> ModelMetadata:
        """Return model metadata and configuration."""
        classes_map = self._class_names or {
            cid: info["name"] for cid, info in self._class_info_map.items()
        }
        categories = list({info["category"] for info in self._class_info_map.values()}) or SORTIFY_CATEGORIES
        return ModelMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            task="object-detection",
            weights_path=str(self.weights_path),
            device=self.device or "auto",
            is_loaded=self.is_loaded,
            num_classes=len(classes_map),
            classes=classes_map,
            categories=sorted(categories),
        )
