#!/usr/bin/env python3
"""Read-only summary script that computes the exact proposed 10-class Sortify YOLO11
dataset distribution (raw vs. post-filtering/deduplication) and writes a comprehensive
Markdown report to Summaries/Results/proposed_dataset_distribution.md.

Does NOT modify any dataset files.
"""

from __future__ import annotations

import glob
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "RAW Datasets"
OUT_MD = ROOT / "Summaries" / "Results" / "proposed_dataset_distribution.md"


def analyze_distribution():
    # Target 10-class taxonomy order
    target_classes = [
        (0, "paper", "Recyclable"),
        (1, "cardboard", "Recyclable"),
        (2, "glass", "Recyclable"),
        (3, "metal", "Recyclable"),
        (4, "plastic_bottle", "Recyclable"),
        (5, "plastic_bag", "General waste (or Film Drop-off)"),
        (6, "sanitary_ppe_waste", "General waste"),
        (7, "organic", "Organic"),
        (8, "battery_ewaste", "E-waste"),
        (9, "hazardous_medical", "Hazardous"),
    ]

    raw_inst_by_target = Counter()
    raw_imgs_by_target = defaultdict(set)
    clean_inst_by_target = Counter()
    clean_imgs_by_target = defaultdict(set)
    capped_inst_by_target = Counter()

    dataset_stats = {}

    # -------------------------------------------------------------------------
    # 1. Bangladesh Recyclable Waste (data_canonical_75-15-10)
    # 0: Paper (DROP), 1: Plastic Bottle -> plastic_bottle, 2: Plastic bag -> plastic_bag, 3: Plastic waste (DROP)
    # -------------------------------------------------------------------------
    bg_dir = RAW_DIR / "bangladesh_recyclable_waste_dataset" / "data_canonical_75-15-10"
    bg_map = {1: "plastic_bottle", 2: "plastic_bag"}
    bg_seen_stems = set()
    bg_raw_imgs = 0
    bg_clean_imgs = 0
    bg_src_raw = Counter()
    bg_src_clean = Counter()

    for split in ["train", "valid", "test"]:
        for f in sorted((bg_dir / split / "labels").glob("*.txt")):
            bg_raw_imgs += 1
            img_id = f"bg:{split}:{f.stem}"
            stem = f.name.split(".rf.")[0]
            classes = [int(l.split()[0]) for l in f.read_text().splitlines() if l.strip()]
            for c in classes:
                bg_src_raw[c] += 1
                if c in bg_map:
                    t = bg_map[c]
                    raw_inst_by_target[t] += 1
                    raw_imgs_by_target[t].add(img_id)

            # Deduplicate Roboflow stem and require pure {1, 2} (no Paper=0 or Plastic waste=3)
            if stem in bg_seen_stems:
                continue
            bg_seen_stems.add(stem)
            cset = frozenset(classes)
            if cset and cset.issubset({1, 2}):
                bg_clean_imgs += 1
                for c in classes:
                    bg_src_clean[c] += 1
                    t = bg_map[c]
                    clean_inst_by_target[t] += 1
                    capped_inst_by_target[t] += 1
                    clean_imgs_by_target[t].add(img_id)

    dataset_stats["Bangladesh"] = {
        "raw_imgs": bg_raw_imgs,
        "clean_imgs": bg_clean_imgs,
        "src_raw": bg_src_raw,
        "src_clean": bg_src_clean,
    }

    # -------------------------------------------------------------------------
    # 2. Mendeley Trash Detection Dataset (CUSTOM_DATASET)
    # 0: BIODEGRADABLE -> organic, 1: CARDBOARD -> cardboard, 2: GLASS -> glass,
    # 3: METAL -> metal, 4: PAPER -> paper, 5: PLASTIC (DROP)
    # -------------------------------------------------------------------------
    md_dir = RAW_DIR / "Mendely" / "Trash Detection Dataset" / "CUSTOM_DATASET"
    md_map = {
        0: "organic",
        1: "cardboard",
        2: "glass",
        3: "metal",
        4: "paper",
    }
    md_raw_imgs = 0
    md_clean_imgs = 0
    md_src_raw = Counter()
    md_src_clean = Counter()

    for split in ["train", "valid", "test"]:
        for f in sorted((md_dir / split / "labels").glob("*.txt")):
            md_raw_imgs += 1
            img_id = f"md:{split}:{f.stem}"
            boxes = []
            for l in f.read_text().splitlines():
                p = l.strip().split()
                if p:
                    boxes.append((int(p[0]), float(p[3]), float(p[4])))
            for c, _, _ in boxes:
                md_src_raw[c] += 1
                if c in md_map:
                    t = md_map[c]
                    raw_inst_by_target[t] += 1
                    raw_imgs_by_target[t].add(img_id)

            cset = frozenset(b[0] for b in boxes)
            # Exclude images containing dropped class 5 (PLASTIC)
            if 5 in cset:
                continue
            # Filter out full-image macro/clipart boxes (w > 0.90 and h > 0.90)
            valid_boxes = [(c, w, h) for c, w, h in boxes if not (w > 0.90 and h > 0.90)]
            if not valid_boxes:
                continue
            # Filter out dense produce catalog images (>15 biodegradable boxes in a single image)
            bio_count = sum(1 for c, _, _ in valid_boxes if c == 0)
            if bio_count > 15:
                continue

            md_clean_imgs += 1
            for c, _, _ in valid_boxes:
                md_src_clean[c] += 1
                t = md_map[c]
                clean_inst_by_target[t] += 1
                clean_imgs_by_target[t].add(img_id)

    # Capped Mendeley METAL at 2,200 instances (from 3,648) for balance
    for c, t in md_map.items():
        if t == "metal":
            capped_inst_by_target[t] += min(md_src_clean[c], 2200)
        else:
            capped_inst_by_target[t] += md_src_clean[c]

    dataset_stats["Mendeley"] = {
        "raw_imgs": md_raw_imgs,
        "clean_imgs": md_clean_imgs,
        "src_raw": md_src_raw,
        "src_clean": md_src_clean,
    }

    # -------------------------------------------------------------------------
    # 3. PackWISE Packaging Waste Dataset
    # -------------------------------------------------------------------------
    pw_dir = RAW_DIR / "Packwise" / "data"
    pw_map = {
        3: "paper",          # paper-bag (150)
        15: "paper",         # paper-other (485)
        22: "paper",         # paper-cup/pot (23)
        9: "metal",          # metal-can (104)
        10: "metal",         # metal-foil (73)
        18: "plastic_bottle",# plastic-bottle (143)
        1: "plastic_bag",    # plastic-foil/bag (2,229)
        13: "sanitary_ppe_waste", # tissue (457)
        8: "hazardous_medical",   # blister (95)
    }
    pw_raw_imgs = 0
    pw_clean_imgs = 0
    pw_src_raw = Counter()
    pw_src_clean = Counter()

    for split in ["train", "val", "test"]:
        for f in sorted((pw_dir / split).glob("*.txt")):
            pw_raw_imgs += 1
            img_id = f"pw:{split}:{f.stem}"
            classes = [int(l.split()[0]) for l in f.read_text().splitlines() if l.strip()]
            for c in classes:
                pw_src_raw[c] += 1
                if c in pw_map:
                    t = pw_map[c]
                    raw_inst_by_target[t] += 1
                    raw_imgs_by_target[t].add(img_id)

            kept_classes = [c for c in classes if c in pw_map]
            if kept_classes:
                pw_clean_imgs += 1
                for c in kept_classes:
                    pw_src_clean[c] += 1
                    t = pw_map[c]
                    clean_inst_by_target[t] += 1
                    capped_inst_by_target[t] += 1
                    clean_imgs_by_target[t].add(img_id)

    dataset_stats["PackWISE"] = {
        "raw_imgs": pw_raw_imgs,
        "clean_imgs": pw_clean_imgs,
        "src_raw": pw_src_raw,
        "src_clean": pw_src_clean,
    }

    # -------------------------------------------------------------------------
    # 4. MedBin Dataset 2
    # -------------------------------------------------------------------------
    mb_dir = RAW_DIR / "MedBin_dataset 2.v1i.yolov11"
    mb_map = {
        23: "cardboard",          # paperbox (372)
        11: "cardboard",          # covid_buffer_box (90)
        15: "glass",              # glass_bottle (112)
        4: "metal",               # Oxygen_cylinder (98)
        26: "plastic_bottle",     # plastic_medical_bottle (310)
        25: "plastic_bag",        # plastic_medical_bag (279)
        2: "sanitary_ppe_waste",  # Mask (242)
        3: "sanitary_ppe_waste",  # N95 (208)
        19: "sanitary_ppe_waste", # medical_gloves (336)
        9: "sanitary_ppe_waste",  # cotton_swab (210)
        18: "sanitary_ppe_waste", # iodine_swab (109)
        14: "sanitary_ppe_waste", # gauze (122)
        6: "sanitary_ppe_waste",  # bandage (105)
        1: "battery_ewaste",      # Electronic_thermometer (126)
        31: "hazardous_medical",  # syringe (266)
        7: "hazardous_medical",   # blade (104)
        29: "hazardous_medical",  # scalpel (99)
        22: "hazardous_medical",  # needle (98)
        0: "hazardous_medical",   # Bloody_objects (148)
        21: "hazardous_medical",  # mercury_thermometer (115)
        13: "hazardous_medical",  # drug_packaging (375)
        36: "hazardous_medical",  # unguent (146)
    }
    mb_raw_imgs = 0
    mb_clean_imgs = 0
    mb_seen_stems = set()
    mb_src_raw = Counter()
    mb_src_clean = Counter()

    for split in ["train", "valid", "test"]:
        for f in sorted((mb_dir / split / "labels").glob("*.txt")):
            mb_raw_imgs += 1
            img_id = f"mb:{split}:{f.stem}"
            stem = f.name.split(".rf.")[0]
            classes = [int(l.split()[0]) for l in f.read_text().splitlines() if l.strip()]
            for c in classes:
                mb_src_raw[c] += 1
                if c in mb_map:
                    t = mb_map[c]
                    raw_inst_by_target[t] += 1
                    raw_imgs_by_target[t].add(img_id)

            if stem in mb_seen_stems:
                continue
            mb_seen_stems.add(stem)
            if 5 in classes:  # Contrived Radioactive_objects
                continue
            kept_classes = [c for c in classes if c in mb_map]
            if kept_classes:
                mb_clean_imgs += 1
                for c in kept_classes:
                    mb_src_clean[c] += 1
                    t = mb_map[c]
                    clean_inst_by_target[t] += 1
                    capped_inst_by_target[t] += 1
                    clean_imgs_by_target[t].add(img_id)

    dataset_stats["MedBin"] = {
        "raw_imgs": mb_raw_imgs,
        "clean_imgs": mb_clean_imgs,
        "src_raw": mb_src_raw,
        "src_clean": mb_src_clean,
    }

    # -------------------------------------------------------------------------
    # 5. RecyBat24 Battery Dataset
    # 0: cylindric, 1: pouch, 2: prismatic -> battery_ewaste
    # -------------------------------------------------------------------------
    rb_dir = RAW_DIR / "recybat24"
    rb_raw_imgs = 0
    rb_src_raw = Counter()
    rb_by_phys_id = defaultdict(list)

    for split in ["train", "val"]:
        for f in sorted((rb_dir / split).glob("*.txt")):
            rb_raw_imgs += 1
            img_id = f"rb:{split}:{f.stem}"
            classes = [int(l.split()[0]) for l in f.read_text().splitlines() if l.strip()]
            for c in classes:
                rb_src_raw[c] += 1
                raw_inst_by_target["battery_ewaste"] += 1
                raw_imgs_by_target["battery_ewaste"].add(img_id)
                clean_inst_by_target["battery_ewaste"] += 1
                clean_imgs_by_target["battery_ewaste"].add(img_id)

            parts = f.stem.split("_")
            phys_id = f"{parts[0]}_{parts[-1]}"
            if classes:
                rb_by_phys_id[phys_id].append((img_id, classes))

    # Subsample max 12 images per unique physical battery (110 physical batteries -> ~1,320 images)
    rb_capped_imgs = 0
    rb_capped_inst = 0
    for phys_id, items in sorted(rb_by_phys_id.items()):
        selected = items[:12]
        rb_capped_imgs += len(selected)
        for _, clist in selected:
            rb_capped_inst += len(clist)
            capped_inst_by_target["battery_ewaste"] += len(clist)

    dataset_stats["RecyBat24"] = {
        "raw_imgs": rb_raw_imgs,
        "clean_imgs": rb_capped_imgs,
        "src_raw": rb_src_raw,
        "capped_inst": rb_capped_inst,
    }

    return target_classes, raw_inst_by_target, raw_imgs_by_target, clean_inst_by_target, clean_imgs_by_target, capped_inst_by_target, dataset_stats


def main():
    (
        target_classes,
        raw_inst,
        raw_imgs,
        clean_inst,
        clean_imgs,
        capped_inst,
        ds_stats,
    ) = analyze_distribution()

    lines = [
        "# Sortify YOLOv11 — Proposed 10-Class Generalized Taxonomy & Dataset Distribution",
        "",
        "## 1. Executive Summary & Curation Principles",
        "",
        "This document defines the proposed **10-class generalized object-detection taxonomy** for **Sortify**, distilled from the **73 source classes** across our 5 YOLOv11-compatible raw datasets (`13,108` raw images, `56,162` raw bounding boxes).",
        "",
        "### Core Curation Rules Applied",
        "1. **Strict `Specific → General` Hierarchy (Zero Broad-into-Specific Contamination):**",
        "   - Because `paper` and `cardboard` are separate target classes, **Bangladesh `Paper`** (`1,575` boxes—which mixes cardboard boxes and paper) is **dropped**.",
        "   - Because `plastic_bottle` and `plastic_bag` are separate target classes, **Mendeley `PLASTIC`** (`3,950` boxes—which mixes rigid bottles and shopping bags) and **Bangladesh `Plastic waste`** (`1,830` catch-all boxes) are **dropped**.",
        "   - Ambiguous composite materials like **PackWISE `aluminium-coated-foil`** (`391` boxes), **`beverage-carton`** (`179` boxes), and **`undefined`** (`521` boxes) are **dropped**.",
        "2. **Pure-Image Filtering (Preventing Unlabeled Foreground Conflicts):**",
        "   - In **Bangladesh**, any multi-object image containing dropped `Paper` or `Plastic waste` is excluded so that unlabeled paper/cardboard or plastic items never appear as false background negatives alongside `Plastic Bottle` or `Plastic bag`.",
        "   - In **Mendeley**, any multi-object image containing dropped `PLASTIC` (`5`) is excluded so unlabeled plastic bottles/bags never appear in the background of `paper`, `cardboard`, `glass`, `metal`, or `organic` images.",
        "3. **Deduplication & Leak Prevention:**",
        "   - **Bangladesh:** Deduplicates the `152` Roboflow augmented/leaked stems (`65` of which leaked across `train`/`valid`/`test`).",
        "   - **MedBin:** Deduplicates the `5` cross-split leaked stems and removes the `19` contrived `Radioactive_objects` images.",
        "   - **RecyBat24:** Groups all images by the **`110` unique physical battery IDs** (`battery001`–`battery110`) before splitting so the same physical battery never leaks across `train`, `val`, and `test`, and caps at `12` viewpoints per physical battery (`1,320` images / `1,341` boxes) so repetitive gray-tabletop photos do not dominate training.",
        "4. **Geometry & Density Cleaning:**",
        "   - **Mendeley:** Removes full-image macro/texture boxes (`w > 0.90` and `h > 0.90`, eliminating `251` cardboard texture crops and `313` paper macro/clipart boxes) and excludes dense stock-catalog produce piles (`>15` `BIODEGRADABLE` boxes per image), reducing `BIODEGRADABLE` from `19,019` noisy micro-boxes to **`2,446` clean `organic` instances**.",
        "",
        "---",
        "",
        "## 2. Proposed 10-Class Detector Taxonomy & Instance Distribution",
        "",
        "| Class ID | Detector Class | Backend Disposal Category | Included Source Classes (Dataset : Class Name) | Raw Mapped Instances | Clean / Filtered Instances | Final Balanced Target Instances | Clean Images (w/ Class) |",
        "| :-: | :--- | :--- | :--- | --: | --: | --: | --: |",
    ]

    source_descriptions = {
        "paper": "Mendeley `PAPER` (`1,838` clean / `2,168` raw), PackWISE `paper-bag` (`150`), `paper-other` (`485`), `paper-cup/pot` (`23`)",
        "cardboard": "Mendeley `CARDBOARD` (`2,401` clean / `2,718` raw), MedBin `paperbox` (`372`), `covid_buffer_box` (`90`)",
        "glass": "Mendeley `GLASS` (`2,091` clean / `2,407` raw), MedBin `glass_bottle` (`112`)",
        "metal": "Mendeley `METAL` (`3,648` clean / `3,820` raw, capped to `2,200`), PackWISE `metal-can` (`104`), `metal-foil` (`73`), MedBin `Oxygen_cylinder` (`98`)",
        "plastic_bottle": "Bangladesh `Plastic Bottle` (`1,264` pure / `1,634` raw), MedBin `plastic_medical_bottle` (`310`), PackWISE `plastic-bottle` (`143`)",
        "plastic_bag": "PackWISE `plastic-foil/bag` (`2,229`), Bangladesh `Plastic bag` (`464` pure / `1,037` raw), MedBin `plastic_medical_bag` (`279`)",
        "sanitary_ppe_waste": "PackWISE `tissue` (`457`), MedBin `medical_gloves` (`336`), `Mask` (`242`), `N95` (`208`), `cotton_swab` (`210`), `gauze` (`122`), `iodine_swab` (`109`), `bandage` (`105`)",
        "organic": "Mendeley `BIODEGRADABLE` (`2,446` clean in images with <=15 bio boxes / `19,019` raw)",
        "battery_ewaste": "RecyBat24 `cylindric` (`479`), `pouch` (`1,742`), `prismatic` (`660`) [capped to `12` imgs/battery = `1,341`], MedBin `Electronic_thermometer` (`126`)",
        "hazardous_medical": "MedBin `drug_packaging` (`375`), `syringe` (`266`), `Bloody_objects` (`148`), `unguent` (`146`), `mercury_thermometer` (`115`), `blade` (`104`), `scalpel` (`99`), `needle` (`98`), PackWISE `blister` (`95`)",
    }

    tot_raw = 0
    tot_clean = 0
    tot_capped = 0

    for cid, cname, backend in target_classes:
        r_i = raw_inst[cname]
        c_i = clean_inst[cname]
        cap_i = capped_inst[cname]
        c_imgs = len(clean_imgs[cname])
        tot_raw += r_i
        tot_clean += c_i
        tot_capped += cap_i
        lines.append(
            f"| **`{cid}`** | **`{cname}`** | **{backend}** | {source_descriptions[cname]} | `{r_i:,}` | `{c_i:,}` | **`{cap_i:,}`** | `{c_imgs:,}` |"
        )

    lines.append(
        f"| — | **TOTAL** | **All 5 Categories Covered** | **33 Kept Source Subclasses (40 Dropped)** | **`{tot_raw:,}`** | **`{tot_clean:,}`** | **`{tot_capped:,}`** | — |"
    )

    min_cls = min(capped_inst.values())
    max_cls = max(capped_inst.values())

    lines.extend([
        "",
        f"> **Class Balance Achievement:** Across all 10 detector classes, the final balanced instance counts range from **`{min_cls:,}`** (`hazardous_medical`) to **`{max_cls:,}`** (`plastic_bag`) — a tight **`{max_cls / min_cls:.2f} : 1` max-to-min ratio** (down from `1,358 : 1` in the raw datasets).",
        "",
        "---",
        "",
        "## 3. Backend Disposal Category Coverage Summary",
        "",
        "| Backend Category | Mapped Detector Classes | Total Final Balanced Instances | Share of Dataset (%) |",
        "| :--- | :--- | --: | --: |",
        f"| **1. Recyclable** | `paper` (`{capped_inst['paper']:,}`), `cardboard` (`{capped_inst['cardboard']:,}`), `glass` (`{capped_inst['glass']:,}`), `metal` (`{capped_inst['metal']:,}`), `plastic_bottle` (`{capped_inst['plastic_bottle']:,}`) | `{capped_inst['paper'] + capped_inst['cardboard'] + capped_inst['glass'] + capped_inst['metal'] + capped_inst['plastic_bottle']:,}` | `{(capped_inst['paper'] + capped_inst['cardboard'] + capped_inst['glass'] + capped_inst['metal'] + capped_inst['plastic_bottle']) * 100.0 / tot_capped:.1f}%` |",
        f"| **2. General Waste** | `plastic_bag` (`{capped_inst['plastic_bag']:,}`), `sanitary_ppe_waste` (`{capped_inst['sanitary_ppe_waste']:,}`) | `{capped_inst['plastic_bag'] + capped_inst['sanitary_ppe_waste']:,}` | `{(capped_inst['plastic_bag'] + capped_inst['sanitary_ppe_waste']) * 100.0 / tot_capped:.1f}%` |",
        f"| **3. Organic** | `organic` (`{capped_inst['organic']:,}`) | `{capped_inst['organic']:,}` | `{capped_inst['organic'] * 100.0 / tot_capped:.1f}%` |",
        f"| **4. E-Waste** | `battery_ewaste` (`{capped_inst['battery_ewaste']:,}`) | `{capped_inst['battery_ewaste']:,}` | `{capped_inst['battery_ewaste'] * 100.0 / tot_capped:.1f}%` |",
        f"| **5. Hazardous** | `hazardous_medical` (`{capped_inst['hazardous_medical']:,}`) | `{capped_inst['hazardous_medical']:,}` | `{capped_inst['hazardous_medical'] * 100.0 / tot_capped:.1f}%` |",
        f"| **TOTAL** | **10 Classes** | **`{tot_capped:,}`** | **`100.0%`** |",
        "",
        "---",
        "",
        "## 4. Dataset-by-Dataset Contribution & Dropped Classes Breakdown",
        "",
        "### 4.1 Bangladesh Recyclable Waste Dataset (`data_canonical_75-15-10`)",
        f"- **Images Retained:** `{ds_stats['Bangladesh']['clean_imgs']:,}` pure deduplicated images (from `{ds_stats['Bangladesh']['raw_imgs']:,}` raw images).",
        "- **Class Mapping & Filtering:**",
        "  - `0: Paper` (`1,575` raw) $\\to$ **DROPPED** *(mixes cardboard boxes and paper; any Bangladesh image containing `Paper` is excluded)*.",
        f"  - `1: Plastic Bottle` (`1,634` raw) $\\to$ **`4: plastic_bottle`** (`{ds_stats['Bangladesh']['src_clean'][1]:,}` clean instances in pure deduplicated images).",
        f"  - `2: Plastic bag` (`1,037` raw) $\\to$ **`5: plastic_bag`** (`{ds_stats['Bangladesh']['src_clean'][2]:,}` clean instances in pure deduplicated images).",
        "  - `3: Plastic waste` (`1,830` raw) $\\to$ **DROPPED** *(catch-all class mixing cups, straws, and wrappers; images containing `Plastic waste` are excluded to prevent unlabeled plastic clutter)*.",
        "",
        "### 4.2 Mendeley Trash Detection Dataset (`CUSTOM_DATASET`)",
        f"- **Images Retained:** `{ds_stats['Mendeley']['clean_imgs']:,}` clean images (from `{ds_stats['Mendeley']['raw_imgs']:,}` unique images).",
        "- **Class Mapping & Filtering:**",
        f"  - `0: BIODEGRADABLE` (`19,019` raw) $\\to$ **`7: organic`** (`{ds_stats['Mendeley']['src_clean'][0]:,}` clean instances after removing images with `>15` bio boxes and full-image boxes).",
        f"  - `1: CARDBOARD` (`2,718` raw) $\\to$ **`1: cardboard`** (`{ds_stats['Mendeley']['src_clean'][1]:,}` clean instances after removing `251` full-image texture crops).",
        f"  - `2: GLASS` (`2,407` raw) $\\to$ **`2: glass`** (`{ds_stats['Mendeley']['src_clean'][2]:,}` clean instances).",
        f"  - `3: METAL` (`3,820` raw) $\\to$ **`3: metal`** (`{ds_stats['Mendeley']['src_clean'][3]:,}` clean instances, subsampled/capped to **`2,200`**).",
        f"  - `4: PAPER` (`2,168` raw) $\\to$ **`0: paper`** (`{ds_stats['Mendeley']['src_clean'][4]:,}` clean instances after removing `313` full-image macro/clipart boxes).",
        "  - `5: PLASTIC` (`3,950` raw) $\\to$ **DROPPED** *(broad class mixing bottles and bags; all images containing `PLASTIC` are excluded)*.",
        "",
        "### 4.3 PackWISE Packaging Waste Dataset (`data`)",
        f"- **Images Retained:** `{ds_stats['PackWISE']['clean_imgs']:,}` images containing kept target classes (from `{ds_stats['PackWISE']['raw_imgs']:,}` raw images). *(Note: `4096×4096` images will be pre-resized to `1280×1280` during dataset construction to prevent disk/I/O bottlenecks.)*",
        "- **Kept Classes (9):**",
        "  - `3: paper-bag` (`150`), `15: paper-other` (`485`), `22: paper-cup/pot` (`23`) $\\to$ **`0: paper`** (`658` instances).",
        "  - `9: metal-can` (`104`), `10: metal-foil` (`73`) $\\to$ **`3: metal`** (`177` instances).",
        "  - `18: plastic-bottle` (`143`) $\\to$ **`4: plastic_bottle`** (`143` instances).",
        "  - `1: plastic-foil/bag` (`2,229`) $\\to$ **`5: plastic_bag`** (`2,229` instances).",
        "  - `13: tissue` (`457`) $\\to$ **`6: sanitary_ppe_waste`** (`457` instances).",
        "  - `8: blister` (`95`) $\\to$ **`9: hazardous_medical`** (`95` instances).",
        "- **Dropped Classes (14):**",
        "  - `2: aluminium-coated-foil` (`391` — dropped per user instruction), `0: undefined` (`521` — unrecognizable edge slivers), `4: beverage-carton` (`179` — composite Tetra Pak), `5: plastic-other` (`458` — ambiguous plastic fragments), `6: plastic-lid` (`156`), `7: plastic-cup/pot` (`657` — optional rigid plastic, excluded to keep `plastic_bottle` strictly bottles), `11: net` (`32`), `12: metal-lid` (`162` — 51.2% tiny), `14: wood` (`32`), `16: foamed-plastic` (`226` — 50% tiny styrofoam crumbs), `17: textile` (`34`), `19: ribbon/tape` (`77`), `20: metal-other` (`15`), `21: tube` (`14`).",
        "",
        "### 4.4 MedBin Dataset 2 (`MedBin_dataset 2.v1i.yolov11`)",
        f"- **Images Retained:** `{ds_stats['MedBin']['clean_imgs']:,}` deduplicated images containing kept classes (from `{ds_stats['MedBin']['raw_imgs']:,}` raw images).",
        "- **Kept Classes (22):**",
        "  - `23: paperbox` (`372`), `11: covid_buffer_box` (`90`) $\\to$ **`1: cardboard`** (`462` instances).",
        "  - `15: glass_bottle` (`112`) $\\to$ **`2: glass`** (`112` instances).",
        "  - `4: Oxygen_cylinder` (`98`) $\\to$ **`3: metal`** (`98` instances).",
        "  - `26: plastic_medical_bottle` (`310`) $\\to$ **`4: plastic_bottle`** (`310` instances).",
        "  - `25: plastic_medical_bag` (`279`) $\\to$ **`5: plastic_bag`** (`279` instances).",
        "  - `2: Mask` (`242`), `3: N95` (`208`), `19: medical_gloves` (`336`), `9: cotton_swab` (`210`), `18: iodine_swab` (`109`), `14: gauze` (`122`), `6: bandage` (`105`) $\\to$ **`6: sanitary_ppe_waste`** (`1,331` deduplicated instances).",
        "  - `1: Electronic_thermometer` (`126`) $\\to$ **`8: battery_ewaste`** (`126` instances).",
        "  - `13: drug_packaging` (`375`), `31: syringe` (`266`), `0: Bloody_objects` (`148`), `36: unguent` (`146`), `21: mercury_thermometer` (`115`), `7: blade` (`104`), `29: scalpel` (`99`), `22: needle` (`98`) $\\to$ **`9: hazardous_medical`** (`1,349` deduplicated instances).",
        "- **Dropped Classes (15):**",
        "  - `5: Radioactive_objects` (`19` — contrived tin can with paper radiation symbol), `8: capsule` (`194` — 43% tiny), `24: pill` (`242` — 58% tiny), and specialized clinical/lab instruments (`10: covid_buffer`, `12: covid_test_case`, `16: harris_uni_core`, `17: harris_uni_core_cap`, `20: medical_infusion_bag`, `27: reagent_tube`, `28: reagent_tube_cap`, `30: single_channel_pipette`, `32: transferpettor_glass`, `33: transferpettor_plastic`, `34: tweezer_metal`, `35: tweezer_plastic`).",
        "",
        "### 4.5 RecyBat24 Battery Dataset (`recybat24`)",
        f"- **Images Retained:** `{ds_stats['RecyBat24']['clean_imgs']:,}` subsampled images (`12` viewpoints per physical battery across all `110` unique physical batteries, down from `{ds_stats['RecyBat24']['raw_imgs']:,}` raw images).",
        "- **Class Mapping & Leak-Free Split Strategy:**",
        f"  - `0: cylindric` (`479` raw), `1: pouch` (`1,742` raw), `2: prismatic` (`660` raw) $\\to$ **`8: battery_ewaste`** (`{ds_stats['RecyBat24']['capped_inst']:,}` balanced instances).",
        "  - **Grouped Split Assignment:** The `110` physical battery IDs (`battery001`–`battery110`) are partitioned at the **physical battery level** (`80%` train = `88` batteries, `10%` val = `11` batteries, `10%` test = `11` batteries) to eliminate the cross-split leakage present in the raw dataset.",
        "",
        "---",
        "",
        "## 5. Summary of Unified Dataset Volume & Split Plan",
        "",
        "| Metric | Count / Specification |",
        "| :--- | :--- |",
        "| **Total Curated Images** | **`~7,473` images** (`1,094` Bangladesh + `3,005` Mendeley (after metal cap) + `536` PackWISE + `1,508` MedBin + `1,320` RecyBat24) |",
        f"| **Total Curated Bounding Boxes** | **`{tot_capped:,}` instances** across `10` classes |",
        "| **Target Train / Val / Test Split** | **`80%` Train / `10%` Val / `10%` Test** (stratified by class and grouped by physical object/stem ID so zero augmented or multi-angle copies leak across splits) |",
        "| **Output Format** | Standard Ultralytics YOLOv11 directory (`images/{train,val,test}`, `labels/{train,val,test}`, `data.yaml`) |",
    ])

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MD} successfully.")


if __name__ == "__main__":
    main()
