#!/usr/bin/env python3
"""Complete negative background collection to reach ~500 clean images.

Collects from Wikimedia with proper delay and supplements with clean crops
from empty background areas to reach 500+ diverse negative background scenes.
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
logger = logging.getLogger("complete_negative_backgrounds")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "RAW Datasets" / "negative_backgrounds"
TARGET_TOTAL = 510

ADDITIONAL_QUERIES = [
    "empty sandy beach shoreline",
    "calm ocean sea horizon",
    "wooden tabletop texture empty",
    "office desk surface empty",
    "wooden parquet flooring",
    "green lawn grass texture",
    "clean asphalt road highway",
    "concrete pavement sidewalk path",
    "granite countertop kitchen marble",
    "clear blue sky clouds horizon",
    "dirt trail park path",
    "cobblestone sidewalk ground",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)"
}


def fetch_candidate_urls(query: str, limit: int = 40) -> List[tuple[str, str]]:
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
                img_url = imageinfo[0].get("thumburl") or imageinfo[0].get("url")
                if img_url and not any(bad in title.lower() for bad in ["logo", "icon", "flag", "map", "chart", "diagram", "trash", "bottle", "can"]):
                    candidates.append((title, img_url))
    except Exception as exc:
        logger.warning("Error fetching candidates for query %r: %s", query, exc)
    return candidates


def download_image(task: tuple[str, str, Path]) -> bool:
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
    current_count = len(existing_files)
    logger.info("Current background image count: %d. Target: %d", current_count, TARGET_TOTAL)

    if current_count >= TARGET_TOTAL:
        logger.info("Target already reached.")
        return

    logger.info("Querying Wikimedia with rate-limiting pauses...")
    all_candidates: List[tuple[str, str]] = []
    seen_urls: Set[str] = set()

    for query in ADDITIONAL_QUERIES:
        candidates = fetch_candidate_urls(query, limit=40)
        new_cnt = 0
        for title, url in candidates:
            if url not in seen_urls:
                seen_urls.add(url)
                all_candidates.append((title, url))
                new_cnt += 1
        logger.info("Query %r yielded %d new candidates (Total pooled: %d)", query, new_cnt, len(all_candidates))
        time.sleep(1.8)  # Polite sleep to avoid 429

    logger.info("Downloading up to target total...")
    tasks = []
    idx = current_count
    for title, url in all_candidates:
        if idx >= TARGET_TOTAL:
            break
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", title.replace("File:", "").replace(".jpg", ""))[:35]
        dest = OUT_DIR / f"bg_{idx:04d}_{safe_name}.jpg"
        tasks.append((title, url, dest))
        idx += 1

    success_cnt = current_count
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = [ex.submit(download_image, t) for t in tasks]
        for fut in as_completed(futures):
            if fut.result():
                success_cnt += 1
                if success_cnt % 25 == 0:
                    logger.info("Current total background images: %d / %d", success_cnt, TARGET_TOTAL)
            if success_cnt >= TARGET_TOTAL:
                break

    final_count = len(list(OUT_DIR.glob("*.jpg")))
    logger.info("Negative background gathering completed. Total images: %d", final_count)


if __name__ == "__main__":
    main()
