#!/usr/bin/env python3
"""Ensure all datasets in 'RAW Datasets' are 100% YOLOv11 object-detection compatible.

Actions performed:
1. Bangladesh Recyclable Waste Dataset (canonical_75-15-10 & robustness_70-15-15):
   - Converts polygon label lines (>4 coordinates) to tight axis-aligned YOLO bounding boxes
     (class_id x_center y_center width height), resolving mixed bbox/polygon label files.
   - Standardizes data.yaml with `path: .` and relative split paths.
2. MedBin_dataset 2.v1i.yolov11:
   - Converts 62 polygon label lines across 50 files (48 mixed bbox/polygon files) to standard
     5-column YOLO bounding boxes.
   - Fixes `../train/images` relative paths in data.yaml and adds `path: .`.
3. Mendely (Trash Detection Dataset/CUSTOM_DATASET):
   - Removes 29 byte-identical orphan `*(1).jpg` images and 30 byte-identical orphan `*(1).txt`
     labels from `train` so images and labels match 1-to-1 (3,536 pairs).
   - Converts 1,166 polygon label lines in `train` to 5-column YOLO bounding boxes.
   - Removes stale Ultralytics `.cache` files (`train/labels.cache`, `valid/labels.cache`).
   - Fixes `../train/images` relative paths in data.yaml and adds `path: .`.
4. Packwise:
   - Converts COCO JSON annotations (`data/train.json`, `data/val.json`, `data/test.json`) to
     YOLOv11 `.txt` bounding-box label files for all 586 images (`train`: 410, `val`: 88, `test`: 88).
   - Generates `data/data.yaml`, `data/yolo.yaml`, and root `data.yaml`.
5. recybat24:
   - Converts COCO JSON bounding-box annotations (`train/annotations.json`, `val/annotations.json`)
     to YOLOv11 `.txt` label files for all 2,835 images (`train`: 1,425, `val`: 1,410), avoiding
     the 3x triplicated mask entries in `instance_segmentation_annotations.json`.
   - Generates `data.yaml`.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "RAW Datasets"


def clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def convert_polygons_in_labels_dir(lbl_dir: Path) -> tuple[int, int]:
    """Convert any polygon lines in lbl_dir/*.txt to 5-column YOLO bbox lines.

    Returns (files_modified, lines_converted).
    """
    files_modified = 0
    lines_converted = 0
    if not lbl_dir.is_dir():
        return 0, 0

    for lp in sorted(lbl_dir.glob("*.txt")):
        raw_text = lp.read_text(encoding="utf-8", errors="replace").strip()
        if not raw_text:
            continue
        new_lines = []
        changed = False
        for line in raw_text.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            cid = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            if len(coords) == 4:
                xc, yc, w, h = [clamp(v) for v in coords]
                new_lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            elif len(coords) >= 6 and len(coords) % 2 == 0:
                xs = [clamp(coords[i]) for i in range(0, len(coords), 2)]
                ys = [clamp(coords[i + 1]) for i in range(0, len(coords), 2)]
                xmin, xmax = min(xs), max(xs)
                ymin, ymax = min(ys), max(ys)
                xc = (xmin + xmax) / 2.0
                yc = (ymin + ymax) / 2.0
                w = xmax - xmin
                h = ymax - ymin
                new_lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                lines_converted += 1
                changed = True
            else:
                raise ValueError(f"Unexpected coordinate count ({len(coords)}) in {lp}: {line}")

        if changed:
            lp.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
            files_modified += 1

    return files_modified, lines_converted


def coco_to_yolo_split(coco_json: Path, labels_dir: Path) -> tuple[int, int]:
    """Convert COCO JSON bbox annotations to YOLOv11 .txt files in labels_dir."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads(coco_json.read_text(encoding="utf-8"))
    categories = sorted(data["categories"], key=lambda c: c["id"])
    cat_to_idx = {c["id"]: idx for idx, c in enumerate(categories)}
    imgs = {img["id"]: img for img in data["images"]}

    anns_by_img: dict[int, list[dict]] = {}
    for ann in data.get("annotations", []):
        if "bbox" not in ann or ann.get("iscrowd", 0) == 1:
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    files_written = 0
    boxes_written = 0
    for img_id, img in imgs.items():
        w, h = float(img["width"]), float(img["height"])
        out_lines = []
        for ann in anns_by_img.get(img_id, []):
            x, y, bw, bh = [float(v) for v in ann["bbox"]]
            xc = clamp((x + bw / 2.0) / w)
            yc = clamp((y + bh / 2.0) / h)
            nw = clamp(bw / w)
            nh = clamp(bh / h)
            cls_idx = cat_to_idx[ann["category_id"]]
            out_lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
            boxes_written += 1

        txt_path = labels_dir / (Path(img["file_name"]).stem + ".txt")
        txt_path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
        files_written += 1

    return files_written, boxes_written


def write_yolo_yaml(yaml_path: Path, train: str, val: str, test: str | None, names: list[str], extra_lines: list[str] | None = None) -> None:
    lines = [
        "path: .",
        f"train: {train}",
        f"val: {val}",
    ]
    if test is not None:
        lines.append(f"test: {test}")
    lines.append("")
    lines.append(f"nc: {len(names)}")
    lines.append("names:")
    for idx, name in enumerate(names):
        lines.append(f"  {idx}: {name}")
    if extra_lines:
        lines.append("")
        lines.extend(extra_lines)
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_bangladesh() -> None:
    bd_dir = RAW_DIR / "bangladesh_recyclable_waste_dataset"
    names = ["Paper", "Plastic Bottle", "Plastic bag", "Plastic waste"]
    for part, ver in [("data_canonical_75-15-10", 12), ("data_robustness_70-15-15", 10)]:
        part_dir = bd_dir / part
        total_files = 0
        total_lines = 0
        for sp in ("train", "valid", "test"):
            f_mod, l_conv = convert_polygons_in_labels_dir(part_dir / sp / "labels")
            total_files += f_mod
            total_lines += l_conv
        extra = [
            "roboflow:",
            "  license: CC BY 4.0",
            "  project: waste-detections-c0mkz",
            f"  url: https://universe.roboflow.com/waste-detections-by-fu/waste-detections-c0mkz/dataset/{ver}",
            f"  version: {ver}",
            "  workspace: waste-detections-by-fu",
        ]
        write_yolo_yaml(
            part_dir / "data.yaml",
            train="train/images",
            val="valid/images",
            test="test/images",
            names=names,
            extra_lines=extra,
        )
        print(f"[Bangladesh {part}] Converted {total_lines} polygon annotations across {total_files} files; updated data.yaml.")


def prepare_medbin() -> None:
    med_dir = RAW_DIR / "MedBin_dataset 2.v1i.yolov11"
    names = [
        "Bloody_objects", "Electronic_thermometer", "Mask", "N95", "Oxygen_cylinder",
        "Radioactive_objects", "bandage", "blade", "capsule", "cotton_swab",
        "covid_buffer", "covid_buffer_box", "covid_test_case", "drug_packaging", "gauze",
        "glass_bottle", "harris_uni_core", "harris_uni_core_cap", "iodine_swab", "medical_gloves",
        "medical_infusion_bag", "mercury_thermometer", "needle", "paperbox", "pill",
        "plastic_medical_bag", "plastic_medical_bottle", "reagent_tube", "reagent_tube_cap", "scalpel",
        "single_channel_pipette", "syringe", "transferpettor_glass", "transferpettor_plastic", "tweezer_metal",
        "tweezer_plastic", "unguent",
    ]
    total_files = 0
    total_lines = 0
    for sp in ("train", "valid", "test"):
        f_mod, l_conv = convert_polygons_in_labels_dir(med_dir / sp / "labels")
        total_files += f_mod
        total_lines += l_conv

    extra = [
        "roboflow:",
        "  workspace: medical-waste-e3vov",
        "  project: medbin_dataset-2",
        "  version: 1",
        "  license: CC BY 4.0",
        "  url: https://universe.roboflow.com/medical-waste-e3vov/medbin_dataset-2/dataset/1",
    ]
    write_yolo_yaml(
        med_dir / "data.yaml",
        train="train/images",
        val="valid/images",
        test="test/images",
        names=names,
        extra_lines=extra,
    )
    print(f"[MedBin] Converted {total_lines} polygon annotations across {total_files} files; updated data.yaml.")


def prepare_mendely() -> None:
    men_dir = RAW_DIR / "Mendely" / "Trash Detection Dataset" / "CUSTOM_DATASET"
    names = ["BIODEGRADABLE", "CARDBOARD", "GLASS", "METAL", "PAPER", "PLASTIC"]

    # 1. Remove byte-identical orphan (1) duplicate images/labels in train
    train_img_dir = men_dir / "train" / "images"
    train_lbl_dir = men_dir / "train" / "labels"
    removed_imgs = 0
    removed_lbls = 0

    for p in list(train_img_dir.glob("*(1).*")):
        base_name = p.name.replace("(1)", "")
        base_path = train_img_dir / base_name
        lbl_for_dup = train_lbl_dir / (p.stem + ".txt")
        if base_path.is_file() and not lbl_for_dup.exists() and p.read_bytes() == base_path.read_bytes():
            p.unlink()
            removed_imgs += 1

    for p in list(train_lbl_dir.glob("*(1).txt")):
        base_name = p.name.replace("(1)", "")
        base_path = train_lbl_dir / base_name
        img_for_dup = list(train_img_dir.glob(p.stem + ".*"))
        if base_path.is_file() and not img_for_dup and p.read_bytes() == base_path.read_bytes():
            p.unlink()
            removed_lbls += 1

    # 2. Remove stale .cache files
    removed_caches = 0
    for cache_file in men_dir.rglob("*.cache"):
        cache_file.unlink()
        removed_caches += 1

    # 3. Convert polygon lines to bounding boxes
    total_files = 0
    total_lines = 0
    for sp in ("train", "valid", "test"):
        f_mod, l_conv = convert_polygons_in_labels_dir(men_dir / sp / "labels")
        total_files += f_mod
        total_lines += l_conv

    write_yolo_yaml(
        men_dir / "data.yaml",
        train="train/images",
        val="valid/images",
        test="test/images",
        names=names,
    )
    print(
        f"[Mendely] Removed {removed_imgs} duplicate orphan images, {removed_lbls} duplicate orphan labels, "
        f"{removed_caches} stale caches; converted {total_lines} polygon annotations across {total_files} files; updated data.yaml."
    )


def prepare_packwise() -> None:
    pw_dir = RAW_DIR / "Packwise"
    data_dir = pw_dir / "data"
    train_json = json.loads((data_dir / "train.json").read_text(encoding="utf-8"))
    categories = sorted(train_json["categories"], key=lambda c: c["id"])
    names = [c["name"] for c in categories]

    total_files = 0
    total_boxes = 0
    for sp in ("train", "val", "test"):
        f_cnt, b_cnt = coco_to_yolo_split(data_dir / f"{sp}.json", data_dir / sp)
        total_files += f_cnt
        total_boxes += b_cnt

    write_yolo_yaml(data_dir / "data.yaml", train="train", val="val", test="test", names=names)
    write_yolo_yaml(data_dir / "yolo.yaml", train="train", val="val", test="test", names=names)
    write_yolo_yaml(pw_dir / "data.yaml", train="data/train", val="data/val", test="data/test", names=names)
    print(f"[Packwise] Generated {total_files} YOLO label files ({total_boxes} bounding boxes) and data.yaml.")


def prepare_recybat24() -> None:
    rb_dir = RAW_DIR / "recybat24"
    train_json = json.loads((rb_dir / "train" / "annotations.json").read_text(encoding="utf-8"))
    categories = sorted(train_json["categories"], key=lambda c: c["id"])
    names = [c["name"] for c in categories]

    total_files = 0
    total_boxes = 0
    for sp in ("train", "val"):
        f_cnt, b_cnt = coco_to_yolo_split(rb_dir / sp / "annotations.json", rb_dir / sp)
        total_files += f_cnt
        total_boxes += b_cnt

    write_yolo_yaml(rb_dir / "data.yaml", train="train", val="val", test=None, names=names)
    print(f"[recybat24] Generated {total_files} YOLO label files ({total_boxes} bounding boxes) and data.yaml.")


def main() -> None:
    prepare_bangladesh()
    prepare_medbin()
    prepare_mendely()
    prepare_packwise()
    prepare_recybat24()


if __name__ == "__main__":
    main()
