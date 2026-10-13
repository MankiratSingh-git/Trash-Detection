from pathlib import Path
import json
import os


# function to convert COCO annotations to YOLO format
def coco_to_yolo(coco_json, labels_dir):
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    data = json.load(open(coco_json, encoding="utf-8"))
    cats = {c["id"]: i for i, c in enumerate(sorted(data["categories"], key=lambda x: x["id"]))}
    imgs = {img["id"]: img for img in data["images"]}
    anns_by_img = {}
    for ann in data["annotations"]:
        if "bbox" not in ann or ann.get("iscrowd", 0) == 1:
            continue
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    for img_id, img in imgs.items():
        w, h = img["width"], img["height"]
        out = []
        for ann in anns_by_img.get(img_id, []):
            x, y, bw, bh = ann["bbox"]
            xc, yc = (x + bw / 2) / w, (y + bh / 2) / h
            out.append(f'{cats[ann["category_id"]]} {xc:.6f} {yc:.6f} {bw/w:.6f} {bh/h:.6f}')
        (labels_dir / (Path(img["file_name"]).stem + ".txt")).write_text(
            "\n".join(out) + ("\n" if out else ""), encoding="utf-8"
        )


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[1]
    base_dir = root_dir / "RAW Datasets" / "Packwise" / "data"

    # Create YOLO labels for train, val, and test sets
    coco_to_yolo(base_dir / "train.json", base_dir / "train")
    coco_to_yolo(base_dir / "val.json", base_dir / "val")
    coco_to_yolo(base_dir / "test.json", base_dir / "test")

    print("generated YOLO text files in the folders train, val, and test.")

    # Load class names from train.json
    with open(base_dir / "train.json", "r", encoding="utf-8") as f:
        train_data = json.load(f)

    categories = sorted(train_data["categories"], key=lambda x: x["id"])
    lines = [
        "path: .",
        "train: train",
        "val: val",
        "test: test",
        "",
        f"nc: {len(categories)}",
        "names:",
    ]
    for i, cat in enumerate(categories):
        lines.append(f"  {i}: {cat['name']}")

    yaml_text = "\n".join(lines) + "\n"
    (base_dir / "yolo.yaml").write_text(yaml_text, encoding="utf-8")
    (base_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")

    print("generated yaml files at:", base_dir / "data.yaml")
