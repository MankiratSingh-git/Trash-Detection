#!/usr/bin/env python3
"""Build the unified 10-class Sortify YOLOv11 Dataset in c:\\Code\\Sortify-ML\\Dataset.

Implements the approved 10-class generalized taxonomy and strict cross-dataset
curation rules:
  0: paper               (Recyclable)
  1: cardboard           (Recyclable)
  2: glass               (Recyclable)
  3: metal               (Recyclable)
  4: plastic_bottle      (Recyclable)
  5: plastic_bag         (General waste / Film drop-off)
  6: sanitary_ppe_waste  (General waste)
  7: organic             (Organic)
  8: battery_ewaste      (E-waste)
  9: hazardous_medical   (Hazardous)

Key safeguards:
- Pure-image filtering in Bangladesh (excludes images with dropped Paper or Plastic waste)
  and Mendeley (excludes images with dropped broad PLASTIC).
- Deduplication of Roboflow base stems in Bangladesh and MedBin to prevent split leakage.
- Physical-battery-grouped 80/10/10 splitting for RecyBat24 (110 unique batteries,
  capped at 12 viewpoints per battery) so zero physical batteries leak across splits.
- Geometry & density filtering in Mendeley (removes full-image macro crops w>0.90 & h>0.90,
  excludes catalog produce piles with >15 biodegradable boxes, and subsamples pure-METAL
  images so Mendeley METAL is capped at ~2,200 instances without leaving unlabeled metal
  objects in kept multi-class images).
- Pre-resizes 4096x4096 PackWISE images to 1280x1280 for fast YOLOv11 dataloader I/O.
- Outputs standard Ultralytics YOLOv11 directory layout:
    Dataset/
      data.yaml
      images/{train,val,test}/
      labels/{train,val,test}/
"""

from __future__ import annotations

import json
import os
import random
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
    "paper",
    "cardboard",
    "glass",
    "metal",
    "plastic_bottle",
    "plastic_bag",
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
    """Bangladesh canonical: deduplicate stems, keep pure {1: Plastic Bottle, 2: Plastic bag}."""
    bg_dir = RAW_DIR / "bangladesh_recyclable_waste_dataset" / "data_canonical_75-15-10"
    bg_map = {1: 4, 2: 5}  # 1 -> plastic_bottle (4), 2 -> plastic_bag (5)
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
            cset = frozenset(b[0] for b in raw_boxes)
            # Pure {1, 2} only (no Paper=0 or Plastic waste=3)
            if not cset or not cset.issubset({1, 2}):
                continue

            img_path = find_image_for_label(img_dir, lp.stem)
            if img_path is None:
                continue

            mapped_boxes = [(bg_map[c], xc, yc, w, h) for c, xc, yc, w, h in raw_boxes]
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


def collect_mendeley_samples(rng: random.Random) -> List[SampleRecord]:
    """Mendeley: exclude images with PLASTIC (5), filter full-image boxes (w>0.9 & h>0.9),
    exclude images with >15 BIODEGRADABLE boxes, and cap METAL (3) to ~2,200 instances
    by subsampling pure-METAL images so kept multi-class images retain 100% of labels.
    """
    md_dir = RAW_DIR / "Mendely" / "Trash Detection Dataset" / "CUSTOM_DATASET"
    md_map = {
        0: 7,  # BIODEGRADABLE -> organic (7)
        1: 1,  # CARDBOARD -> cardboard (1)
        2: 2,  # GLASS -> glass (2)
        3: 3,  # METAL -> metal (3)
        4: 0,  # PAPER -> paper (0)
    }

    multi_or_non_metal: List[SampleRecord] = []
    pure_metal: List[SampleRecord] = []
    metal_in_multi = 0

    for split in ("train", "valid", "test"):
        lbl_dir = md_dir / split / "labels"
        img_dir = md_dir / split / "images"
        for lp in sorted(lbl_dir.glob("*.txt")):
            raw_boxes = parse_yolo_boxes(lp)
            cset = frozenset(b[0] for b in raw_boxes)
            if 5 in cset:  # Exclude images containing broad PLASTIC
                continue

            valid_boxes = [
                (c, xc, yc, w, h)
                for c, xc, yc, w, h in raw_boxes
                if c in md_map and not (w > 0.90 and h > 0.90)
            ]
            if not valid_boxes:
                continue

            bio_cnt = sum(1 for c, _, _, _, _ in valid_boxes if c == 0)
            if bio_cnt > 15:
                continue

            img_path = find_image_for_label(img_dir, lp.stem)
            if img_path is None:
                continue

            base_stem = lp.name.split(".rf.")[0]
            mapped_boxes = [(md_map[c], xc, yc, w, h) for c, xc, yc, w, h in valid_boxes]
            rec = SampleRecord(
                source_dataset="mendeley",
                unique_id=f"md_{split}_{lp.stem}",
                group_id=f"md_{base_stem}",
                src_img_path=img_path,
                boxes=mapped_boxes,
            )

            target_cset = frozenset(b[0] for b in mapped_boxes)
            if target_cset == {3}:  # Pure metal image
                pure_metal.append(rec)
            else:
                multi_or_non_metal.append(rec)
                metal_in_multi += sum(1 for b in mapped_boxes if b[0] == 3)

    # Shuffle pure_metal deterministically and keep enough to reach ~2,200 Mendeley metal instances
    rng.shuffle(pure_metal)
    kept_pure_metal: List[SampleRecord] = []
    current_metal = metal_in_multi
    for rec in pure_metal:
        m_cnt = len(rec.boxes)
        if current_metal + m_cnt <= 2200 or current_metal < 2190:
            kept_pure_metal.append(rec)
            current_metal += m_cnt

    all_md = multi_or_non_metal + kept_pure_metal
    all_md.sort(key=lambda r: r.unique_id)
    return all_md


def collect_packwise_samples() -> List[SampleRecord]:
    """PackWISE: map 9 clean classes and mark 4096x4096 images for 1280x1280 resizing."""
    pw_dir = RAW_DIR / "Packwise" / "data"
    pw_map = {
        3: 0,   # paper-bag -> paper (0)
        15: 0,  # paper-other -> paper (0)
        22: 0,  # paper-cup/pot -> paper (0)
        9: 3,   # metal-can -> metal (3)
        10: 3,  # metal-foil -> metal (3)
        18: 4,  # plastic-bottle -> plastic_bottle (4)
        1: 5,   # plastic-foil/bag -> plastic_bag (5)
        13: 6,  # tissue -> sanitary_ppe_waste (6)
        8: 9,   # blister -> hazardous_medical (9)
    }
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

            samples.append(
                SampleRecord(
                    source_dataset="packwise",
                    unique_id=f"pw_{split}_{lp.stem}",
                    group_id=f"pw_{lp.stem}",
                    src_img_path=img_path,
                    boxes=mapped_boxes,
                    resize_to=1280,
                )
            )
    return samples


def collect_medbin_samples() -> List[SampleRecord]:
    """MedBin: deduplicate base stems, exclude Radioactive_objects (5), map 22 kept classes."""
    mb_dir = RAW_DIR / "MedBin_dataset 2.v1i.yolov11"
    mb_map = {
        23: 1,  # paperbox -> cardboard (1)
        11: 1,  # covid_buffer_box -> cardboard (1)
        15: 2,  # glass_bottle -> glass (2)
        4: 3,   # Oxygen_cylinder -> metal (3)
        26: 4,  # plastic_medical_bottle -> plastic_bottle (4)
        25: 5,  # plastic_medical_bag -> plastic_bag (5)
        2: 6,   # Mask -> sanitary_ppe_waste (6)
        3: 6,   # N95 -> sanitary_ppe_waste (6)
        19: 6,  # medical_gloves -> sanitary_ppe_waste (6)
        9: 6,   # cotton_swab -> sanitary_ppe_waste (6)
        18: 6,  # iodine_swab -> sanitary_ppe_waste (6)
        14: 6,  # gauze -> sanitary_ppe_waste (6)
        6: 6,   # bandage -> sanitary_ppe_waste (6)
        1: 8,   # Electronic_thermometer -> battery_ewaste (8)
        31: 9,  # syringe -> hazardous_medical (9)
        7: 9,   # blade -> hazardous_medical (9)
        29: 9,  # scalpel -> hazardous_medical (9)
        22: 9,  # needle -> hazardous_medical (9)
        0: 9,   # Bloody_objects -> hazardous_medical (9)
        21: 9,  # mercury_thermometer -> hazardous_medical (9)
        13: 9,  # drug_packaging -> hazardous_medical (9)
        36: 9,  # unguent -> hazardous_medical (9)
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
            if 5 in cset:  # Exclude contrived Radioactive_objects
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
    """RecyBat24: group by the 110 unique physical batteries (num_type), cap at 12 views
    per battery, and assign each physical battery to train (80%), val (10%), or test (10%)
    stratified by battery type (cyl, po, pri) so zero physical batteries leak across splits.
    """
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
            mapped_boxes = [(8, xc, yc, w, h) for _, xc, yc, w, h in raw_boxes]

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

    # Assign each physical battery ID to train (80%), val (10%), test (10%) per battery type
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


def assign_stratified_splits(
    samples: List[SampleRecord],
    preassigned_groups: Dict[str, str],
    rng: random.Random,
) -> Dict[str, List[SampleRecord]]:
    """Assign samples into train (80%), val (10%), test (10%) stratified by (dataset, primary_class)
    while honoring preassigned group splits (for RecyBat24 physical batteries).
    """
    splits: Dict[str, List[SampleRecord]] = {"train": [], "val": [], "test": []}
    strata: Dict[Tuple[str, int], List[SampleRecord]] = defaultdict(list)

    for s in samples:
        if s.group_id in preassigned_groups:
            splits[preassigned_groups[s.group_id]].append(s)
        else:
            # Primary class = rarest target class in the image (or first box class)
            cls_counts = Counter(b[0] for b in s.boxes)
            primary_cls = sorted(cls_counts.keys())[0]
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
    ext = rec.src_img_path.suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        ext = ".jpg"
    out_img_path = img_out_dir / f"{rec.unique_id}{ext}"
    out_lbl_path = lbl_out_dir / f"{rec.unique_id}.txt"

    if rec.resize_to is not None and Image is not None:
        with Image.open(rec.src_img_path) as im:
            im = im.convert("RGB")
            if im.size != (rec.resize_to, rec.resize_to):
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
    lines = [
        "# Sortify Unified 10-Class YOLOv11 Dataset",
        "path: .",
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
    print("Collecting and filtering samples across the 5 raw datasets...")
    bg_samples = collect_bangladesh_samples()
    md_samples = collect_mendeley_samples(rng)
    pw_samples = collect_packwise_samples()
    mb_samples = collect_medbin_samples()
    rb_samples, rb_group_splits = collect_recybat24_samples()

    all_samples = bg_samples + md_samples + pw_samples + mb_samples + rb_samples
    print(
        f"Collected {len(all_samples):,} curated images: "
        f"Bangladesh={len(bg_samples):,}, Mendeley={len(md_samples):,}, "
        f"PackWISE={len(pw_samples):,}, MedBin={len(mb_samples):,}, RecyBat24={len(rb_samples):,}"
    )

    splits = assign_stratified_splits(all_samples, rb_group_splits, rng)

    # Prepare clean output directories
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for sp in ("train", "val", "test"):
        (OUT_DIR / "images" / sp).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / sp).mkdir(parents=True, exist_ok=True)

    tasks = []
    for sp in ("train", "val", "test"):
        img_out_dir = OUT_DIR / "images" / sp
        lbl_out_dir = OUT_DIR / "labels" / sp
        for rec in splits[sp]:
            tasks.append((sp, rec, img_out_dir, lbl_out_dir))

    print(f"Writing {len(tasks):,} image/label pairs to {OUT_DIR} using ThreadPoolExecutor...")
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
    manifest_path = ROOT / "Summaries" / "Results" / "built_dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Dataset build complete! Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()
