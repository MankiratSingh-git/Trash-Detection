#!/usr/bin/env python3
"""Test inference and calibrated gating on real internet photos.

Compares:
1) Raw uncalibrated predictions at conf=0.25 / 0.35
2) Calibrated FastAPI YOLOv11 service with class-specific minimum confidence gating
"""

import sys
from pathlib import Path

# Add fast-api directory to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "fast-api"))

from core.config import settings
from models.yolov11.service import YOLOv11Model
from ultralytics import YOLO

test_img_dir = ROOT / "scratch" / "internet_test_images"
images = sorted(test_img_dir.glob("*.jpg"))

print("=" * 80)
print(f"Internet Test Images Verification ({len(images)} images in {test_img_dir})")
print(f"Config DEFAULT_CONF_THRESHOLD: {settings.DEFAULT_CONF_THRESHOLD}")
print(f"Config CLASS_MIN_CONFIDENCE:   {settings.CLASS_MIN_CONFIDENCE}")
print("=" * 80)

# 1. Raw uncalibrated inference (what was failing before)
print("\n--- 1. Raw YOLOv11 Model (conf=0.25, uncalibrated) ---")
raw_model = YOLO("YOLOv11/model/best.pt")
for p in images:
    res = raw_model.predict(str(p), conf=0.25, verbose=False)[0]
    preds = []
    for b in res.boxes:
        cid = int(b.cls.item())
        conf = float(b.conf.item())
        cname = res.names[cid]
        preds.append(f"{cname} ({conf*100:.1f}%)")
    pred_str = ", ".join(preds) if preds else "NO DETECTION"
    print(f"  [{p.name:<20}] -> {pred_str}")

# 2. FastAPI calibrated service inference (with class-specific gating)
print("\n--- 2. FastAPI YOLOv11 Service with Calibrated Confidence Gating ---")
service = YOLOv11Model()
service.load()

for p in images:
    with open(p, "rb") as f:
        img_bytes = f.read()
    resp = service.predict(img_bytes)
    dets = []
    for d in resp.detections:
        dets.append(f"{d.class_name} [{d.category}] ({d.confidence*100:.1f}%)")
    det_str = ", ".join(dets) if dets else "NO DETECTION (Noise suppressed)"
    print(f"  [{p.name:<20}] -> {det_str}")

print("\n" + "=" * 80)
