#!/usr/bin/env python3
"""Download and curate the real-world everyday waste TACO subset for Sortify.

Pillar 2: Real Everyday Litter Data:
- Ingest clean everyday waste images (clear plastic bottles on grass/sand/tables,
  crushed cans on asphalt, paper on desks/floors).
- Map clean bottles to plastic_bottle (1), cans to metal (2), and paper to paper_cardboard (0).
- Save to RAW Datasets/TACO/
"""

from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_taco_subset")

ROOT = Path(__file__).resolve().parents[1]
TACO_DIR = ROOT / "RAW Datasets" / "TACO"
IMAGES_DIR = TACO_DIR / "images"
ANN_URL = "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json"
ANN_FILE = TACO_DIR / "annotations.json"
CURATED_FILE = TACO_DIR / "curated_taco_samples.json"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

# TACO category ID to Sortify 8-class ID mapping:
# 0: paper_cardboard, 1: plastic_bottle, 2: metal
TACO_CLASS_MAP = {
    # Plastic bottles
    4: 1,   # Other plastic bottle -> plastic_bottle (1)
    5: 1,   # Clear plastic bottle -> plastic_bottle (1)
    # Metal cans
    10: 2,  # Food Can -> metal (2)
    11: 2,  # Aerosol -> metal (2)
    12: 2,  # Drink can -> metal (2)
    # Paper and cardboard
    13: 0,  # Toilet tube -> paper_cardboard (0)
    14: 0,  # Other carton -> paper_cardboard (0)
    15: 0,  # Egg carton -> paper_cardboard (0)
    16: 0,  # Drink carton -> paper_cardboard (0)
    17: 0,  # Corrugated carton -> paper_cardboard (0)
    18: 0,  # Meal carton -> paper_cardboard (0)
    19: 0,  # Pizza box -> paper_cardboard (0)
    20: 0,  # Paper cup -> paper_cardboard (0)
    30: 0,  # Magazine paper -> paper_cardboard (0)
    32: 0,  # Wrapping paper -> paper_cardboard (0)
    33: 0,  # Normal paper -> paper_cardboard (0)
    34: 0,  # Paper bag -> paper_cardboard (0)
    35: 0,  # Plastified paper bag -> paper_cardboard (0)
}


def download_annotations() -> Dict[str, Any]:
    if ANN_FILE.is_file() and ANN_FILE.stat().st_size > 10000:
        logger.info("Found existing annotations at %s", ANN_FILE)
        with open(ANN_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info("Downloading TACO annotations from %s...", ANN_URL)
    req = urllib.request.Request(ANN_URL, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()
    with open(ANN_FILE, "wb") as f:
        f.write(content)
    logger.info("Saved TACO annotations (%d bytes) to %s", len(content), ANN_FILE)
    return json.loads(content.decode("utf-8"))


def download_single_image(task: tuple[int, str, str, Path]) -> bool:
    img_id, url_orig, url_thumb, dest_path = task
    if dest_path.is_file() and dest_path.stat().st_size > 5000:
        return True

    # Try 640 thumb first for fast and efficient download, fallback to original
    for candidate_url in [url_thumb, url_orig]:
        if not candidate_url:
            continue
        try:
            req = urllib.request.Request(candidate_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = resp.read()
            if len(data) > 3000:
                with Image.open(io.BytesIO(data)) as im:
                    im = im.convert("RGB")
                    im.save(dest_path, format="JPEG", quality=90)
                return True
        except Exception:
            continue

    return False


def main() -> None:
    TACO_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    data = download_annotations()
    images_by_id = {img["id"]: img for img in data["images"]}

    # Group annotations by image
    img_boxes: Dict[int, List[Dict[str, Any]]] = {}
    for ann in data["annotations"]:
        taco_cid = ann["category_id"]
        if taco_cid in TACO_CLASS_MAP:
            target_cid = TACO_CLASS_MAP[taco_cid]
            bbox = ann["bbox"]  # [x, y, w, h] in image pixel coordinates
            img_id = ann["image_id"]
            if img_id not in img_boxes:
                img_boxes[img_id] = []
            img_boxes[img_id].append({
                "target_class": target_cid,
                "taco_class": taco_cid,
                "bbox": bbox,
            })

    logger.info("Found %d TACO images containing target bottles, cans, and paper/cardboard.", len(img_boxes))

    # Prepare download tasks
    tasks = []
    for img_id in sorted(img_boxes.keys()):
        img_info = images_by_id.get(img_id)
        if not img_info:
            continue
        dest_filename = f"taco_{img_id:05d}.jpg"
        dest_path = IMAGES_DIR / dest_filename
        url_orig = img_info.get("flickr_url", "")
        url_thumb = img_info.get("flickr_640_url", "")
        tasks.append((img_id, url_orig, url_thumb, dest_path))

    logger.info("Downloading %d TACO images to %s...", len(tasks), IMAGES_DIR)
    success = 0
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(download_single_image, t) for t in tasks]
        for fut in as_completed(futures):
            if fut.result():
                success += 1
                if success % 100 == 0 or success == len(tasks):
                    logger.info("Downloaded %d / %d TACO images...", success, len(tasks))

    # Build curated JSON with normalized bounding boxes
    curated_records = []
    for img_id, raw_boxes in img_boxes.items():
        dest_path = IMAGES_DIR / f"taco_{img_id:05d}.jpg"
        if not dest_path.is_file():
            continue

        try:
            with Image.open(dest_path) as im:
                actual_w, actual_h = im.size
        except Exception:
            continue

        orig_w = images_by_id[img_id].get("width", actual_w)
        orig_h = images_by_id[img_id].get("height", actual_h)

        scale_x = actual_w / orig_w if orig_w > 0 else 1.0
        scale_y = actual_h / orig_h if orig_h > 0 else 1.0

        yolo_boxes = []
        for b in raw_boxes:
            target_cls = b["target_class"]
            bx, by, bw, bh = b["bbox"]

            # Scale to actual downloaded image dimensions
            sbx = bx * scale_x
            sby = by * scale_y
            sbw = bw * scale_x
            sbh = bh * scale_y

            # Convert to normalized YOLO format (xc, yc, w, h)
            xc = (sbx + sbw / 2.0) / actual_w
            yc = (sby + sbh / 2.0) / actual_h
            nw = sbw / actual_w
            nh = sbh / actual_h

            # Clip bounds
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            nw = max(0.001, min(1.0, nw))
            nh = max(0.001, min(1.0, nh))

            yolo_boxes.append([target_cls, round(xc, 6), round(yc, 6), round(nw, 6), round(nh, 6)])

        if yolo_boxes:
            curated_records.append({
                "unique_id": f"taco_{img_id:05d}",
                "image_path": str(dest_path.relative_to(ROOT)).replace("\\", "/"),
                "boxes": yolo_boxes,
            })

    with open(CURATED_FILE, "w", encoding="utf-8") as f:
        json.dump(curated_records, f, indent=2)

    logger.info("TACO curation complete! Saved %d validated samples to %s", len(curated_records), CURATED_FILE)


if __name__ == "__main__":
    main()
