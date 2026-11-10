#!/usr/bin/env python3
"""Build the streamlined 8-class Sortify YOLOv11 Dataset in c:\\Code\\Sortify-ML\\Dataset.

Streamlined 8-Class Taxonomy:
  0: paper_cardboard    (Recyclable) - Merged paper and cardboard
  1: plastic_bottle     (Recyclable) - Rigid beverage & medical bottles
  2: metal              (Recyclable) - Cans, foil, lids, cylinders, and scrap metal
  3: flexible_plastic   (General waste) - Shopping bags, snack packets, sachets, film
  4: sanitary_ppe_waste (General waste) - Masks, gloves, swabs, dressings, tissues
  5: organic            (Organic) - Biodegradable food, fruit, and produce waste
  6: battery_ewaste     (E-waste) - Cylindrical, pouch, and prismatic batteries, thermometers
  7: hazardous_medical  (Hazardous) - Syringes, blades, sharps, blisters, medical containers

Key Improvements:
- Glass is completely eliminated (eradicates bottle-to-glass false positives at the root).
- Paper & Cardboard are merged into 'paper_cardboard', unlocking clean paper & packaging annotations.
- Taxonomical Realignment: PackWISE class 2 ('aluminium-coated-foil') remapped to 'flexible_plastic' (General waste).
- Bangladesh 'Plastic waste' (crushed rigid cups/shards) dropped to eliminate flexible plastic attractor bias.
- PackWISE 2x2 Quadrant Tiling: 4096px conveyor images tiled into 2048px quadrants preserving native resolution.
- PackWISE Probe Grouping: Images grouped by video sequence / probe ID to eliminate split leakage across train/val/test.
- 100% Authentic Organic Waste: Removed 2500 instance cap; filtered out sparse platter scenes to protect precision.
- Zero-leakage stratified splits for physical batteries and probe sequences.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "RAW Datasets"
OUT_DIR = ROOT / "Dataset"

TARGET_NAMES: List[str] = [
    "paper_cardboard",
    "plastic_bottle",
    "metal",
    "flexible_plastic",
    "sanitary_ppe_waste",
    "organic",
    "battery_ewaste",
    "hazardous_medical",
]

SEED = 42


@dataclass
class SampleRecord:
    source_dataset: str
    unique_id: str
    group_id: str
    src_img_path: Path
    boxes: List[Tuple[int, float, float, float, float]]  # (target_cls, xc, yc, w, h)
    crop_box: Tuple[int, int, int, int] | None = None  # (left, top, right, bottom) in pixel coords
    resize_to: int | None = None


def find_image_for_label(img_dir: Path, stem: str) -> Path | None:
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".bmp", ".webp"):
        cand = img_dir / f"{stem}{ext}"
        if cand.is_file():
            return cand
    matches = list(img_dir.glob(f"{stem}.*"))
    for m in matches:
        if m.suffix.lower() != ".txt":
            return m
    return None


def parse_yolo_boxes(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            c = int(parts[0])
            xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            boxes.append((c, xc, yc, w, h))
    return boxes


def collect_bangladesh_samples() -> List[SampleRecord]:
    """Bangladesh: map classes without drop, deduplicate Roboflow base stems:
      0: Paper -> paper_cardboard (0)
      1: Plastic Bottle -> plastic_bottle (1)
      2: Plastic bag -> flexible_plastic (3)
    Note: Bangladesh class 3 ('Plastic waste' - crushed rigid cups, shards) is dropped
    to eliminate flexible plastic attractor bias and restore balanced cross-class boundaries.
    """
    bg_dir = RAW_DIR / "bangladesh_recyclable_waste_dataset" / "data_canonical_75-15-10"
    bg_map = {
        0: 0,  # Paper -> paper_cardboard (0)
        1: 1,  # Plastic Bottle -> plastic_bottle (1)
        2: 3,  # Plastic bag -> flexible_plastic (3)
    }
    seen_stems = set()
    samples: List[SampleRecord] = []

    for split in ("train", "valid", "test"):
        lbl_dir = bg_dir / split / "labels"
        img_dir = bg_dir / split / "images"
        for lp in sorted(lbl_dir.glob("*.txt")):
            base_stem = lp.name.split(".rf.")[0]
            if base_stem in seen_stems:
                continue
            seen_stems.add(base_stem)

            raw_boxes = parse_yolo_boxes(lp)
            if not raw_boxes:
                continue

            img_path = find_image_for_label(img_dir, lp.stem)
            if img_path is None:
                continue

            mapped_boxes = [(bg_map[c], xc, yc, w, h) for c, xc, yc, w, h in raw_boxes if c in bg_map]
            if not mapped_boxes:
                continue

            samples.append(
                SampleRecord(
                    source_dataset="bangladesh",
                    unique_id=f"bg_{base_stem}",
                    group_id=f"bg_{base_stem}",
                    src_img_path=img_path,
                    boxes=mapped_boxes,
                )
            )
    return samples


def collect_packwise_samples(rng: random.Random) -> Tuple[List[SampleRecord], Dict[str, str]]:
    """PackWISE: map packaging classes into the 8-class taxonomy, tile 4096x4096 images
    into 2x2 quadrants (2048x2048), and group by video sequence / probe ID to eliminate
    split leakage across train, val, and test.

    - Remaps class 2 (aluminium-coated-foil) from metal (2) to flexible_plastic (3).
    - 2x2 Quadrants:
        tl: top-left     [0, 2048, 0, 2048]     norm: [0.0..0.5, 0.0..0.5]
        tr: top-right    [2048, 4096, 0, 2048]  norm: [0.5..1.0, 0.0..0.5]
        bl: bottom-left  [0, 2048, 2048, 4096]  norm: [0.0..0.5, 0.5..1.0]
        br: bottom-right [2048, 4096, 2048, 4096] norm: [0.5..1.0, 0.5..1.0]
    - Recalculates YOLO normalized coordinates for each quadrant.
    - Discards boxes outside the quadrant, boxes with < 20% area remaining, or slivers.
    - Saves tiled records with prefix 'pw_tile_<quadrant>_<stem>'.
    - Groups probe IDs ('Probe00022', etc.) with an 80/10/10 split.
    """
    pw_dir = RAW_DIR / "Packwise" / "data"
    pw_map = {
        3: 0,   # paper-bag -> paper_cardboard (0)
        15: 0,  # paper-other -> paper_cardboard (0)
        22: 0,  # paper-cup/pot -> paper_cardboard (0)
        4: 0,   # beverage-carton -> paper_cardboard (0)
        18: 1,  # plastic-bottle -> plastic_bottle (1)
        9: 2,   # metal-can -> metal (2)
        10: 2,  # metal-foil -> metal (2)
        12: 2,  # metal-lid -> metal (2)
        20: 2,  # metal-other -> metal (2)
        2: 3,   # aluminium-coated-foil -> flexible_plastic (3) (General waste)
        1: 3,   # plastic-foil/bag -> flexible_plastic (3)
        13: 4,  # tissue -> sanitary_ppe_waste (4)
        8: 7,   # blister -> hazardous_medical (7)
    }

    quadrants = [
        ("tl", (0, 0, 2048, 2048), (0.0, 0.5, 0.0, 0.5)),
        ("tr", (2048, 0, 4096, 2048), (0.5, 1.0, 0.0, 0.5)),
        ("bl", (0, 2048, 2048, 4096), (0.0, 0.5, 0.5, 1.0)),
        ("br", (2048, 2048, 4096, 4096), (0.5, 1.0, 0.5, 1.0)),
    ]

    all_probes = set()
    samples: List[SampleRecord] = []

    for split in ("train", "val", "test"):
        sp_dir = pw_dir / split
        for lp in sorted(sp_dir.glob("*.txt")):
            raw_boxes = parse_yolo_boxes(lp)
            mapped_boxes = [
                (pw_map[c], xc, yc, w, h)
                for c, xc, yc, w, h in raw_boxes
                if c in pw_map
            ]
            if not mapped_boxes:
                continue

            img_path = find_image_for_label(sp_dir, lp.stem)
            if img_path is None:
                continue

            m = re.search(r"Probe\d+", lp.stem)
            probe_id = m.group(0) if m else lp.stem
            all_probes.add(probe_id)
            group_id = f"pw_{probe_id}"

            for qname, crop_box, (qx0, qx1, qy0, qy1) in quadrants:
                quad_boxes: List[Tuple[int, float, float, float, float]] = []
                for c, xc, yc, w, h in mapped_boxes:
                    xmin = xc - w / 2.0
                    xmax = xc + w / 2.0
                    ymin = yc - h / 2.0
                    ymax = yc + h / 2.0

                    ixmin = max(xmin, qx0)
                    ixmax = min(xmax, qx1)
                    iymin = max(ymin, qy0)
                    iymax = min(ymax, qy1)

                    iw = ixmax - ixmin
                    ih = iymax - iymin
                    if iw <= 0 or ih <= 0:
                        continue

                    orig_area = w * h
                    clipped_area = iw * ih
                    if orig_area > 0 and (clipped_area / orig_area) < 0.20:
                        continue

                    qw = qx1 - qx0
                    qh = qy1 - qy0
                    n_w = iw / qw
                    n_h = ih / qh
                    n_xc = ((ixmin + ixmax) / 2.0 - qx0) / qw
                    n_yc = ((iymin + iymax) / 2.0 - qy0) / qh

                    if n_w < 0.005 or n_h < 0.005:
                        continue

                    n_xc = max(0.0, min(1.0, n_xc))
                    n_yc = max(0.0, min(1.0, n_yc))
                    n_w = max(0.0, min(1.0, n_w))
                    n_h = max(0.0, min(1.0, n_h))

                    quad_boxes.append((c, n_xc, n_yc, n_w, n_h))

                if not quad_boxes:
                    continue

                samples.append(
                    SampleRecord(
                        source_dataset="packwise",
                        unique_id=f"pw_tile_{qname}_{lp.stem}",
                        group_id=group_id,
                        src_img_path=img_path,
                        boxes=quad_boxes,
                        crop_box=crop_box,
                    )
                )

    pw_probes = sorted(all_probes)
    rng.shuffle(pw_probes)
    n = len(pw_probes)
    n_val = max(1, round(n * 0.10))
    n_test = max(1, round(n * 0.10))
    pw_group_splits: Dict[str, str] = {}
    for i, pid in enumerate(pw_probes):
        gid = f"pw_{pid}"
        if i < n_val:
            pw_group_splits[gid] = "val"
        elif i < n_val + n_test:
            pw_group_splits[gid] = "test"
        else:
            pw_group_splits[gid] = "train"

    return samples, pw_group_splits


def collect_medbin_samples() -> List[SampleRecord]:
    """MedBin: map medical and clinical waste classes, completely omitting glass (15, 32)."""
    mb_dir = RAW_DIR / "MedBin_dataset 2.v1i.yolov11"
    mb_map = {
        23: 0,  # paperbox -> paper_cardboard (0)
        11: 0,  # covid_buffer_box -> paper_cardboard (0)
        26: 1,  # plastic_medical_bottle -> plastic_bottle (1)
        4: 2,   # Oxygen_cylinder -> metal (2)
        34: 2,  # tweezer_metal -> metal (2)
        25: 3,  # plastic_medical_bag -> flexible_plastic (3)
        2: 4,   # Mask -> sanitary_ppe_waste (4)
        3: 4,   # N95 -> sanitary_ppe_waste (4)
        19: 4,  # medical_gloves -> sanitary_ppe_waste (4)
        9: 4,   # cotton_swab -> sanitary_ppe_waste (4)
        18: 4,  # iodine_swab -> sanitary_ppe_waste (4)
        14: 4,  # gauze -> sanitary_ppe_waste (4)
        6: 4,   # bandage -> sanitary_ppe_waste (4)
        1: 6,   # Electronic_thermometer -> battery_ewaste (6)
        31: 7,  # syringe -> hazardous_medical (7)
        7: 7,   # blade -> hazardous_medical (7)
        29: 7,  # scalpel -> hazardous_medical (7)
        22: 7,  # needle -> hazardous_medical (7)
        0: 7,   # Bloody_objects -> hazardous_medical (7)
        21: 7,  # mercury_thermometer -> hazardous_medical (7)
        13: 7,  # drug_packaging -> hazardous_medical (7)
        36: 7,  # unguent -> hazardous_medical (7)
    }
    seen_stems = set()
    samples: List[SampleRecord] = []

    for split in ("train", "valid", "test"):
        lbl_dir = mb_dir / split / "labels"
        img_dir = mb_dir / split / "images"
        for lp in sorted(lbl_dir.glob("*.txt")):
            base_stem = lp.name.split(".rf.")[0]
            if base_stem in seen_stems:
                continue
            seen_stems.add(base_stem)

            raw_boxes = parse_yolo_boxes(lp)
            cset = frozenset(b[0] for b in raw_boxes)
            # Exclude radioactive and glass images
            if 5 in cset or 15 in cset or 32 in cset:
                continue

            mapped_boxes = [
                (mb_map[c], xc, yc, w, h)
                for c, xc, yc, w, h in raw_boxes
                if c in mb_map
            ]
            if not mapped_boxes:
                continue

            img_path = find_image_for_label(img_dir, lp.stem)
            if img_path is None:
                continue

            samples.append(
                SampleRecord(
                    source_dataset="medbin",
                    unique_id=f"mb_{base_stem}",
                    group_id=f"mb_{base_stem}",
                    src_img_path=img_path,
                    boxes=mapped_boxes,
                )
            )
    return samples


def collect_recybat24_samples() -> Tuple[List[SampleRecord], Dict[str, str]]:
    """RecyBat24: group by the 110 physical batteries, 80/10/10 split by battery identity."""
    rb_dir = RAW_DIR / "recybat24"
    by_phys_id: Dict[str, List[SampleRecord]] = defaultdict(list)
    by_type: Dict[str, List[str]] = defaultdict(list)

    for split in ("train", "val"):
        sp_dir = rb_dir / split
        for lp in sorted(sp_dir.glob("*.txt")):
            raw_boxes = parse_yolo_boxes(lp)
            if not raw_boxes:
                continue
            img_path = find_image_for_label(sp_dir, lp.stem)
            if img_path is None:
                continue

            parts = lp.stem.split("_")
            btype = parts[-1]  # cyl, po, pri
            phys_id = f"rb_{parts[0]}_{btype}"
            mapped_boxes = [(6, xc, yc, w, h) for _, xc, yc, w, h in raw_boxes]  # 6: battery_ewaste

            by_phys_id[phys_id].append(
                SampleRecord(
                    source_dataset="recybat24",
                    unique_id=f"rb_{split}_{lp.stem}",
                    group_id=phys_id,
                    src_img_path=img_path,
                    boxes=mapped_boxes,
                )
            )

    samples: List[SampleRecord] = []
    for phys_id in sorted(by_phys_id.keys()):
        btype = phys_id.split("_")[-1]
        by_type[btype].append(phys_id)
        selected = by_phys_id[phys_id][:12]
        samples.extend(selected)

    rng = random.Random(SEED)
    group_to_split: Dict[str, str] = {}
    for btype, phys_ids in sorted(by_type.items()):
        pids = list(phys_ids)
        rng.shuffle(pids)
        n = len(pids)
        n_val = max(1, round(n * 0.10))
        n_test = max(1, round(n * 0.10))
        for i, pid in enumerate(pids):
            if i < n_val:
                group_to_split[pid] = "val"
            elif i < n_val + n_test:
                group_to_split[pid] = "test"
            else:
                group_to_split[pid] = "train"

    return samples, group_to_split


def collect_organic_waste_samples(rng: random.Random) -> List[SampleRecord]:
    """Collect genuine organic food scraps, leftovers, peels, bones, and kitchen waste
    from RAW Datasets/organic_waste_detection.
    - Deduplicates Roboflow base stems so each unique scene is represented once.
    - Maps classes 0..31 (excluding class 19 'Other-waste') to Sortify class 5 ('organic').
    - Excludes sparsely annotated buffet/tray scenes (e.g. video stills with 1-2 labeled
      items amidst full food platters) to eliminate false positive precision penalties.
    - Removes artificial 2,500 instance cap, unlocking 100% authentic organic training scenes.
    """
    ow_dir = RAW_DIR / "organic_waste_detection"
    seen_stems = set()
    samples: List[SampleRecord] = []

    for split in ("train", "valid", "test"):
        lbl_dir = ow_dir / split / "labels"
        img_dir = ow_dir / split / "images"
        if not lbl_dir.is_dir():
            continue
        for lp in sorted(lbl_dir.glob("*.txt")):
            base_stem = lp.name.split(".rf.")[0]
            if base_stem in seen_stems:
                continue
            seen_stems.add(base_stem)

            # Exclude sparsely annotated buffet / tray video sequence
            if "Still" in base_stem or "20220209103351" in base_stem:
                continue

            raw_boxes = parse_yolo_boxes(lp)
            if not raw_boxes:
                continue

            # Exclude class 19 ('Other-waste'), map everything else to 5 ('organic')
            valid_boxes = [
                (5, xc, yc, w, h)
                for c, xc, yc, w, h in raw_boxes
                if c != 19 and not (w > 0.95 and h > 0.95)
            ]
            if not valid_boxes:
                continue

            # Exclude sparse scene slivers where total box area is negligible (< 0.5% of frame)
            total_box_area = sum(w * h for _, _, _, w, h in valid_boxes)
            if total_box_area < 0.005:
                continue

            img_path = find_image_for_label(img_dir, lp.stem)
            if img_path is None:
                continue

            samples.append(
                SampleRecord(
                    source_dataset="organic_waste",
                    unique_id=f"ow_{base_stem}",
                    group_id=f"ow_{base_stem}",
                    src_img_path=img_path,
                    boxes=valid_boxes,
                )
            )

    samples.sort(key=lambda r: r.unique_id)
    return samples


def collect_mendeley_metal_samples(rng: random.Random) -> List[SampleRecord]:
    """Collect only authentic natural-background metal cans from Mendeley (excluding studio white backgrounds).
    - Completely drops all Mendeley BIODEGRADABLE, CARDBOARD, PAPER, GLASS, and PLASTIC.
    - Filters out any metal image with >25% pure white pixels to eliminate studio catalog bias.
    """
    md_dir = RAW_DIR / "Mendely" / "Trash Detection Dataset" / "CUSTOM_DATASET"
    candidates: List[SampleRecord] = []

    for split in ("train", "valid", "test"):
        lbl_dir = md_dir / split / "labels"
        img_dir = md_dir / split / "images"
        if not lbl_dir.is_dir():
            continue
        for lp in sorted(lbl_dir.glob("metal*.txt")):
            raw_boxes = parse_yolo_boxes(lp)
            if not raw_boxes:
                continue
            if not all(c == 3 for c, _, _, _, _ in raw_boxes):
                continue

            img_path = find_image_for_label(img_dir, lp.stem)
            if img_path is None:
                continue

            if Image is not None:
                try:
                    import numpy as np
                    arr = np.array(Image.open(img_path))
                    white_px = np.sum((arr[:, :, 0] > 240) & (arr[:, :, 1] > 240) & (arr[:, :, 2] > 240))
                    pct_white = (white_px / (arr.shape[0] * arr.shape[1])) * 100
                    if pct_white >= 25.0:
                        continue
                except Exception:
                    pass

            base_stem = lp.name.split(".rf.")[0]
            mapped_boxes = [(2, xc, yc, w, h) for _, xc, yc, w, h in raw_boxes]
            candidates.append(
                SampleRecord(
                    source_dataset="mendeley",
                    unique_id=f"md_{split}_{lp.stem}",
                    group_id=f"md_{base_stem}",
                    src_img_path=img_path,
                    boxes=mapped_boxes,
                )
            )

    rng.shuffle(candidates)
    kept: List[SampleRecord] = []
    curr = 0
    for r in candidates:
        cnt = len(r.boxes)
        if curr + cnt <= 550:
            kept.append(r)
            curr += cnt
        else:
            break

    kept.sort(key=lambda r: r.unique_id)
    return kept


def assign_stratified_splits(
    samples: List[SampleRecord],
    preassigned_groups: Dict[str, str],
    rng: random.Random,
) -> Dict[str, List[SampleRecord]]:
    splits: Dict[str, List[SampleRecord]] = {"train": [], "val": [], "test": []}
    strata: Dict[Tuple[str, int], List[SampleRecord]] = defaultdict(list)

    for s in samples:
        if s.group_id in preassigned_groups:
            splits[preassigned_groups[s.group_id]].append(s)
        else:
            cls_counts = Counter(b[0] for b in s.boxes)
            primary_cls = sorted(cls_counts.keys())[0] if cls_counts else 0
            strata[(s.source_dataset, primary_cls)].append(s)

    for key in sorted(strata.keys()):
        bucket = list(strata[key])
        rng.shuffle(bucket)
        n = len(bucket)
        n_val = max(1, round(n * 0.10)) if n >= 10 else 0
        n_test = max(1, round(n * 0.10)) if n >= 10 else 0
        for i, s in enumerate(bucket):
            if i < n_val:
                splits["val"].append(s)
            elif i < n_val + n_test:
                splits["test"].append(s)
            else:
                splits["train"].append(s)

    return splits


def write_sample_worker(task: Tuple[str, SampleRecord, Path, Path]) -> Tuple[str, str, int, List[int]]:
    split_name, rec, img_out_dir, lbl_out_dir = task
    needs_reencode = rec.crop_box is not None or rec.resize_to is not None
    ext = ".jpg" if needs_reencode else rec.src_img_path.suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"
    out_img_path = img_out_dir / f"{rec.unique_id}{ext}"
    out_lbl_path = lbl_out_dir / f"{rec.unique_id}.txt"

    if needs_reencode:
        if Image is None:
            raise RuntimeError("Pillow (PIL) is required to process and re-encode dataset images.")
        with Image.open(rec.src_img_path) as im:
            im = im.convert("RGB")
            if rec.crop_box is not None:
                im = im.crop(rec.crop_box)
            if rec.resize_to is not None and im.size != (rec.resize_to, rec.resize_to):
                im = im.resize((rec.resize_to, rec.resize_to), Image.Resampling.BILINEAR)
            im.save(out_img_path, format="JPEG", quality=92)
    else:
        shutil.copy2(rec.src_img_path, out_img_path)

    label_lines = [
        f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        for c, xc, yc, w, h in rec.boxes
    ]
    out_lbl_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    return split_name, rec.source_dataset, len(rec.boxes), [b[0] for b in rec.boxes]


def write_data_yaml(dataset_dir: Path) -> None:
    try:
        rel_path = dataset_dir.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        rel_path = dataset_dir.resolve().as_posix()
    lines = [
        "# Sortify Streamlined 8-Class YOLOv11 Dataset",
        f"path: {rel_path}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(TARGET_NAMES)}",
        "names:",
    ]
    for idx, name in enumerate(TARGET_NAMES):
        lines.append(f"  {idx}: {name}")
    (dataset_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rng = random.Random(SEED)
    print("=" * 76)
    print("Building Sortify Streamlined 8-Class YOLOv11 Dataset (No Glass)")
    print(f"Target Directory: {OUT_DIR}")
    print("=" * 76)

    print("Collecting and mapping samples across raw datasets...")
    bg_samples = collect_bangladesh_samples()
    pw_samples, pw_group_splits = collect_packwise_samples(rng)
    mb_samples = collect_medbin_samples()
    rb_samples, rb_group_splits = collect_recybat24_samples()
    ow_samples = collect_organic_waste_samples(rng)
    md_samples = collect_mendeley_metal_samples(rng)

    all_samples = bg_samples + pw_samples + mb_samples + rb_samples + ow_samples + md_samples
    print(
        f"Collected {len(all_samples):,} curated images:\n"
        f"  Bangladesh (clean paper, bottle, bag): {len(bg_samples):,}\n"
        f"  PackWISE (2x2 quadrants & probe-grouped): {len(pw_samples):,}\n"
        f"  MedBin (clean non-glass): {len(mb_samples):,}\n"
        f"  RecyBat24 (110 physical batteries): {len(rb_samples):,}\n"
        f"  Organic Waste (100% authentic real food scraps): {len(ow_samples):,}\n"
        f"  Mendeley (metal on natural bgs only): {len(md_samples):,}"
    )

    preassigned_groups = {**rb_group_splits, **pw_group_splits}
    splits = assign_stratified_splits(all_samples, preassigned_groups, rng)

    # Clean and re-initialize output directories
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val", "test"):
        sp_img = OUT_DIR / "images" / sp
        sp_lbl = OUT_DIR / "labels" / sp
        if sp_img.is_dir():
            shutil.rmtree(sp_img)
        if sp_lbl.is_dir():
            shutil.rmtree(sp_lbl)
        sp_img.mkdir(parents=True, exist_ok=True)
        sp_lbl.mkdir(parents=True, exist_ok=True)

    # Remove stale YOLO caches
    for c_file in (OUT_DIR / "labels").glob("*.cache"):
        try:
            c_file.unlink()
        except Exception:
            pass

    tasks = []
    for sp in ("train", "val", "test"):
        img_out_dir = OUT_DIR / "images" / sp
        lbl_out_dir = OUT_DIR / "labels" / sp
        for rec in splits[sp]:
            tasks.append((sp, rec, img_out_dir, lbl_out_dir))

    print(f"\nWriting {len(tasks):,} image/label pairs to {OUT_DIR}...")
    split_img_counts = Counter()
    split_inst_counts = Counter()
    class_split_inst = defaultdict(Counter)
    class_split_imgs = defaultdict(Counter)
    ds_split_imgs = defaultdict(Counter)

    with ThreadPoolExecutor(max_workers=min(16, (os.cpu_count() or 4) * 2)) as ex:
        futures = [ex.submit(write_sample_worker, t) for t in tasks]
        for i, fut in enumerate(as_completed(futures), 1):
            sp, src_ds, n_boxes, cls_list = fut.result()
            split_img_counts[sp] += 1
            split_inst_counts[sp] += n_boxes
            ds_split_imgs[src_ds][sp] += 1
            for c in cls_list:
                class_split_inst[c][sp] += 1
            for c in set(cls_list):
                class_split_imgs[c][sp] += 1
            if i % 1500 == 0 or i == len(tasks):
                print(f"  Processed {i:,}/{len(tasks):,} files...")

    write_data_yaml(OUT_DIR)

    manifest = {
        "total_images": sum(split_img_counts.values()),
        "total_instances": sum(split_inst_counts.values()),
        "split_images": dict(split_img_counts),
        "split_instances": dict(split_inst_counts),
        "dataset_split_images": {k: dict(v) for k, v in ds_split_imgs.items()},
        "class_split_instances": {
            TARGET_NAMES[c]: dict(class_split_inst[c]) for c in range(len(TARGET_NAMES))
        },
        "class_split_images": {
            TARGET_NAMES[c]: dict(class_split_imgs[c]) for c in range(len(TARGET_NAMES))
        },
    }
    manifest_path = ROOT / "Summaries" / "Results" / "streamlined_dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDataset build complete! Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
