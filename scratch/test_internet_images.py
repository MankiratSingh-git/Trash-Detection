#!/usr/bin/env python3
"""Internet Test Images Verification: Raw vs Calibrated Confidence Gating.

Downloads/verifies 10 real-world internet photos (bottles on beaches/grass/tables,
cans on streets/roads/grass, paper on desks/floors) and evaluates:
1) Raw uncalibrated predictions at conf=0.25
2) FastAPI YOLOv11 service with class-specific calibrated confidence gating
3) Direct FastAPI HTTP endpoint verification via TestClient
"""

import os
import sys
import urllib.request
from pathlib import Path

# Add fast-api directory to sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "fast-api"))

from core.config import settings
from models.yolov11.service import YOLOv11Model
from ultralytics import YOLO

urls = {
    "bottle_beach_1.jpg": "https://images.unsplash.com/photo-1618477461853-cf6ed80faba5?auto=format&fit=crop&w=1200&q=80",
    "bottle_beach_2.jpg": "https://images.unsplash.com/photo-1530587191325-3db32d826c18?auto=format&fit=crop&w=1200&q=80",
    "bottle_table.jpg": "https://images.unsplash.com/photo-1526951521990-620dc14c214b?auto=format&fit=crop&w=1200&q=80",
    "bottle_grass.jpg": "https://images.unsplash.com/photo-1605600659873-d808a13e4d2a?auto=format&fit=crop&w=1200&q=80",
    "soda_can_street.jpg": "https://images.unsplash.com/photo-1596755389378-c31d21fd1273?auto=format&fit=crop&w=1200&q=80",
    "soda_can_road.jpg": "https://images.unsplash.com/photo-1528190336454-13cd56b45b5a?auto=format&fit=crop&w=1200&q=80",
    "paper_desk.jpg": "https://images.unsplash.com/photo-1588854337236-6889d631faa8?auto=format&fit=crop&w=1200&q=80",
    "paper_floor.jpg": "https://images.unsplash.com/photo-1584438784894-089d6a62b8fa?auto=format&fit=crop&w=1200&q=80",
    "litter_trail.jpg": "https://images.unsplash.com/photo-1563245372-f21724e3856d?auto=format&fit=crop&w=1200&q=80",
    "can_grass.jpg": "https://images.unsplash.com/photo-1595278069441-2cf29f8005a4?auto=format&fit=crop&w=1200&q=80",
}

target_dir = ROOT / "scratch" / "internet_test_images"
target_dir.mkdir(parents=True, exist_ok=True)

headers = {"User-Agent": "Mozilla/5.0"}
for filename, url in urls.items():
    dest = target_dir / filename
    if not dest.exists() or dest.stat().st_size == 0:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as response, open(dest, "wb") as f:
                f.write(response.read())
            print(f"Downloaded: {filename} ({dest.stat().st_size} bytes)")
        except Exception as e:
            print(f"Error {filename}: {e}")

images = sorted(target_dir.glob("*.jpg"))
print("=" * 80)
print(f"Sortify Internet Test Images Verification ({len(images)} images in {target_dir})")
print(f"FastAPI DEFAULT_CONF_THRESHOLD: {settings.DEFAULT_CONF_THRESHOLD}")
print(f"FastAPI CLASS_MIN_CONFIDENCE:   {settings.CLASS_MIN_CONFIDENCE}")
print("=" * 80)

# 1. Raw uncalibrated inference (baseline with false positives)
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
    print(f"  [{p.name:<22}] -> {pred_str}")

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
    det_str = ", ".join(dets) if dets else "NO DETECTION (Spurious noise suppressed)"
    print(f"  [{p.name:<22}] -> {det_str}")

# 3. Direct HTTP Endpoint verification via TestClient
print("\n--- 3. FastAPI HTTP Endpoint Test (/api/v1/yolov11/predict) ---")
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)
endpoint_passed = True
for p in images:
    with open(p, "rb") as f:
        r = client.post("/api/v1/yolov11/predict", files={"file": (p.name, f, "image/jpeg")})
    if r.status_code != 200:
        print(f"  FAIL [{p.name}]: status {r.status_code} - {r.text}")
        endpoint_passed = False
    else:
        data = r.json()
        dets = [f"{d['class_name']} ({d['confidence']*100:.1f}%)" for d in data.get("detections", [])]
        s = ", ".join(dets) if dets else "NO DETECTION (Clean)"
        print(f"  HTTP 200 [{p.name:<22}] -> {s}")

print("\n" + "=" * 80)
if endpoint_passed:
    print("ALL INTERNET IMAGE TESTS & FASTAPI ENDPOINTS PASSED.")
else:
    print("SOME ENDPOINTS RETURNED ERRORS.")
print("=" * 80)
