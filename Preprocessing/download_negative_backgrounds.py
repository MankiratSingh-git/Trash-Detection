#!/usr/bin/env python3
"""Download and curate ~500 clean negative background images for Sortify.

Collects realistic empty backgrounds where litter commonly occurs:
- Empty tables, desks, countertops
- Wooden, tile, and concrete floors
- Sandy beaches and ocean/sky horizons
- Park grass, lawns, and dirt trails
- Roads, asphalt, and sidewalks

These images contain NO target objects (empty annotation files in YOLO).
Ultralytics YOLO uses them as negative background samples to eliminate false positives
and hallucinated detections on empty surfaces.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Set

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("download_negative_backgrounds")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "RAW Datasets" / "negative_backgrounds"
TARGET_COUNT = 520

SEARCH_QUERIES = [
    "sandy beach seashore",
    "sea ocean water horizon",
    "empty wooden table surface",
    "empty desk workspace",
    "hardwood floor texture",
    "green grass lawn park",
    "asphalt road texture street",
    "concrete sidewalk pavement",
    "marble granite kitchen countertop",
    "empty dirt gravel path",
    "blue sky fluffy clouds horizon",
    "ceramic tile floor texture",
    "cobblestone pavement road",
    "empty park field meadow",
    "clean laminate tabletop surface",
]

HEADERS = {
    "User-Agent": "SortifyNegativeBackgroundCollector/1.0 (waste-sorting research project; academic use; sortify@local)"
}


def fetch_candidate_urls(query: str, limit: int = 50) -> List[tuple[str, str]]:
    """Query Wikimedia Commons API for high quality bitmap photos."""
    encoded_query = urllib.parse.quote(query)
    url = (
        f"https://commons.wikimedia.org/w/api.php?action=query&format=json"
        f"&generator=search&gsrsearch=filetype:bitmap+{encoded_query}"
        f"&gsrnamespace=6&gsrlimit={limit}&prop=imageinfo&iiprop=url&iiurlwidth=800"
    )
    req = urllib.request.Request(url, headers=HEADERS)
    candidates: List[tuple[str, str]] = []
    try:
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", {})
            for pid, pinfo in pages.items():
                title = pinfo.get("title", f"img_{pid}")
                imageinfo = pinfo.get("imageinfo", [])
                if not imageinfo:
                    continue
                # Prefer 800px thumburl for faster download and good resolution
                img_url = imageinfo[0].get("thumburl") or imageinfo[0].get("url")
                if img_url and not any(bad in title.lower() for bad in ["logo", "icon", "flag", "map", "chart", "diagram", "trash", "bottle", "can"]):
                    candidates.append((title, img_url))
    except Exception as exc:
        logger.warning("Error fetching candidates for query %r: %s", query, exc)
    return candidates


def download_and_validate_image(task: tuple[str, str, Path]) -> bool:
    title, url, dest_path = task
    if dest_path.is_file() and dest_path.stat().st_size > 5000:
        return True

    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
        if len(content) < 5000:
            return False

        with Image.open(io.BytesIO(content)) as im:
            im = im.convert("RGB")
            w, h = im.size
            if w < 256 or h < 256:
                return False
            # Standardize max dimension to 800 for efficient storage & training
            if max(w, h) > 800:
                scale = 800 / max(w, h)
                nw, nh = int(w * scale), int(h * scale)
                im = im.resize((nw, nh), Image.Resampling.BILINEAR)
            im.save(dest_path, format="JPEG", quality=90)
        return True
    except Exception:
        if dest_path.is_file():
            try:
                dest_path.unlink()
            except Exception:
                pass
        return False


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_files = list(OUT_DIR.glob("*.jpg"))
    logger.info("Found %d existing background images in %s", len(existing_files), OUT_DIR)

    if len(existing_files) >= TARGET_COUNT:
        logger.info("Target count of %d already satisfied. Exiting.", TARGET_COUNT)
        return

    logger.info("Gathering candidate image URLs across %d diverse outdoor and indoor queries...", len(SEARCH_QUERIES))
    all_candidates: List[tuple[str, str]] = []
    seen_urls: Set[str] = set()

    for query in SEARCH_QUERIES:
        candidates = fetch_candidate_urls(query, limit=50)
        new_candidates = 0
        for title, url in candidates:
            if url not in seen_urls:
                seen_urls.add(url)
                all_candidates.append((title, url))
                new_candidates += 1
        logger.info("Query '%s' returned %d new candidates (Total: %d)", query, new_candidates, len(all_candidates))
        time.sleep(0.6)  # Respect API rate limits

    logger.info("Downloading images up to %d target count...", TARGET_COUNT)
    tasks = []
    curr_idx = len(existing_files)
    for title, url in all_candidates:
        if curr_idx >= TARGET_COUNT:
            break
        # Create safe unique filename
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", title.replace("File:", "").replace(".jpg", ""))[:40]
        dest = OUT_DIR / f"bg_{curr_idx:04d}_{safe_name}.jpg"
        tasks.append((title, url, dest))
        curr_idx += 1

    success_count = len(existing_files)
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(download_and_validate_image, t) for t in tasks]
        for fut in as_completed(futures):
            if fut.result():
                success_count += 1
                if success_count % 50 == 0 or success_count >= TARGET_COUNT:
                    logger.info("Downloaded %d / %d valid background images...", success_count, TARGET_COUNT)
            if success_count >= TARGET_COUNT:
                break

    valid_images = list(OUT_DIR.glob("*.jpg"))
    logger.info("Negative background collection complete! Total images in %s: %d", OUT_DIR, len(valid_images))


if __name__ == "__main__":
    main()
