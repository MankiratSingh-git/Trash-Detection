#!/usr/bin/env python3
"""Supplement negative backgrounds to reach exactly ~500 clean images.

Extracts authentic empty background quadrants (e.g. clean conveyor surface,
clean tables, empty outdoor ground) from raw datasets to reach 500+ images.
"""

from __future__ import annotations

import logging
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("complete_to_500_backgrounds")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "RAW Datasets" / "negative_backgrounds"
PW_DIR = ROOT / "RAW Datasets" / "Packwise" / "data"
TARGET_TOTAL = 505

QUADRANTS = [
    ("tl", (0, 0, 2048, 2048), (0.0, 0.5, 0.0, 0.5)),
    ("tr", (2048, 0, 4096, 2048), (0.5, 1.0, 0.0, 0.5)),
    ("bl", (0, 2048, 2048, 4096), (0.0, 0.5, 0.5, 1.0)),
    ("br", (2048, 2048, 4096, 4096), (0.5, 1.0, 0.5, 1.0)),
]


def parse_yolo_boxes(label_path: Path):
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append((xc, yc, w, h))
    return boxes


def main():
    existing = list(OUT_DIR.glob("*.jpg"))
    current = len(existing)
    needed = TARGET_TOTAL - current
    logger.info("Current background count: %d, needed: %d", current, needed)

    if needed <= 0:
        logger.info("Target already reached.")
        return

    added = 0
    # Scan Packwise splits for completely empty quadrants (pure conveyor surface)
    for split in ("train", "val", "test"):
        if added >= needed:
            break
        sp_dir = PW_DIR / split
        if not sp_dir.is_dir():
            continue
        for lp in sorted(sp_dir.glob("*.txt")):
            if added >= needed:
                break
            raw_boxes = parse_yolo_boxes(lp)
            img_path = sp_dir / f"{lp.stem}.jpg"
            if not img_path.is_file():
                continue

            for qname, crop_box, (qx0, qx1, qy0, qy1) in QUADRANTS:
                if added >= needed:
                    break
                # Check if ANY box intersects this quadrant
                has_box = False
                for xc, yc, w, h in raw_boxes:
                    xmin, xmax = xc - w / 2.0, xc + w / 2.0
                    ymin, ymax = yc - h / 2.0, yc + h / 2.0
                    if max(xmin, qx0) < min(xmax, qx1) and max(ymin, qy0) < min(ymax, qy1):
                        has_box = True
                        break

                if not has_box:
                    # Clean empty quadrant!
                    dest = OUT_DIR / f"bg_empty_quad_{lp.stem}_{qname}.jpg"
                    if not dest.is_file():
                        try:
                            with Image.open(img_path) as im:
                                q_im = im.crop(crop_box).resize((800, 800), Image.Resampling.BILINEAR)
                                q_im.save(dest, format="JPEG", quality=90)
                                added += 1
                                if added % 25 == 0:
                                    logger.info("Added %d / %d authentic empty background crops...", added, needed)
                        except Exception:
                            pass

    final_total = len(list(OUT_DIR.glob("*.jpg")))
    logger.info("Supplement complete! Total clean negative background images: %d", final_total)


if __name__ == "__main__":
    main()
