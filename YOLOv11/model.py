#!/usr/bin/env python3
"""Sortify YOLOv11 Training, Evaluation & Disposal-Category Inference Script.

Trains an Ultralytics YOLO11 object detector on the curated 10-class Sortify
dataset in `Dataset/` (`Dataset/data.yaml`), evaluates `best.pt` on the held-out
`test` split, exports per-class test metrics, and maps detections to Sortify's
5 application-level disposal categories:
  1. Recyclable
  2. General waste
  3. Organic
  4. E-waste
  5. Hazardous
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "Dataset"
DATASET_YAML = DATASET_DIR / "data.yaml"
RUNTIME_DATASET_YAML = Path(__file__).resolve().parent / "dataset.runtime.yaml"
RUNS_DIR = Path(__file__).resolve().parent / "runs"
SUMMARIES_RESULTS_DIR = ROOT / "Summaries" / "Results"

# Canonical 10 detector classes -> 5 Sortify backend disposal categories
SORTIFY_CLASSES: Dict[int, Dict[str, str]] = {
    0: {"name": "paper", "category": "Recyclable", "guidance": "Clean dry paper bin"},
    1: {"name": "cardboard", "category": "Recyclable", "guidance": "Flatten and place in cardboard/paper recycling"},
    2: {"name": "glass", "category": "Recyclable", "guidance": "Rinse glass bottle/jar and place in glass recycling"},
    3: {"name": "metal", "category": "Recyclable", "guidance": "Empty metal can/foil and place in metal recycling"},
    4: {"name": "plastic_bottle", "category": "Recyclable", "guidance": "Empty/crush rigid plastic bottle and cap for recycling"},
    5: {"name": "plastic_bag", "category": "General waste", "guidance": "General waste bin (or dedicated soft-plastic film drop-off)"},
    6: {"name": "sanitary_ppe_waste", "category": "General waste", "guidance": "Seal masks, gloves, tissues, and dressings in general waste"},
    7: {"name": "organic", "category": "Organic", "guidance": "Compost or biodegradable organic waste bin"},
    8: {"name": "battery_ewaste", "category": "E-waste", "guidance": "Never bin; take to battery / e-waste collection point"},
    9: {"name": "hazardous_medical", "category": "Hazardous", "guidance": "Sharps / hazardous clinical or pharmaceutical disposal container"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train, evaluate, or run inference with the Sortify 10-class YOLO11 waste detector."
    )
    parser.add_argument(
        "--model",
        default="yolo11s.pt",
        help="YOLO11 checkpoint or architecture YAML (default: yolo11s.pt)",
    )
    parser.add_argument(
        "--data",
        default=None,
        help="Path to dataset data.yaml (defaults to Dataset/data.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square input image resolution (default: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size; use 16 for standard GPUs or -1 for Ultralytics AutoBatch (default: 16)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Compute device, e.g. '0' for GPU 0, 'cpu', or omit for automatic selection",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Dataloader worker threads (default: 4)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=25,
        help="Early-stopping patience in epochs without validation mAP improvement (default: 25)",
    )
    parser.add_argument(
        "--cache",
         default=False,
        action="store_true",
        help="Cache images in RAM/disk for faster epoch iteration",
    )
    parser.add_argument(
        "--name",
        default="sortify_yolo11s_10cls",
        help="Run folder name under YOLOv11/runs (default: sortify_yolo11s_10cls)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the last checkpoint in the named run folder",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip evaluating best.pt on the held-out test split after training",
    )
    parser.add_argument(
        "--eval-only",
        default=None,
        metavar="WEIGHTS",
        help="Skip training and only evaluate the specified checkpoint (e.g. YOLOv11/runs/sortify_yolo11s_10cls/weights/best.pt) on the test split",
    )
    parser.add_argument(
        "--predict",
        default=None,
        metavar="SOURCE",
        help="Run inference with Sortify disposal-category mapping on an image file or folder using --eval-only or best.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for --predict inference (default: 0.25)",
    )
    return parser.parse_args()


def validate_inputs(data_yaml: Path) -> Path:
    dataset_dir = data_yaml.parent
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML was not found: {data_yaml}")
    for split in ("train", "val", "test"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"Missing YOLO {split!r} image or label directory under {dataset_dir}"
            )
    return dataset_dir


def create_runtime_dataset_yaml(data_yaml: Path, dataset_dir: Path) -> Path:
    """Resolve `path: .` to the absolute dataset directory path for Ultralytics."""
    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    path_line_found = False
    resolved_lines: List[str] = []
    for line in lines:
        if line.strip().startswith("path:"):
            resolved_lines.append(f"path: {dataset_dir.as_posix()}")
            path_line_found = True
        else:
            resolved_lines.append(line)
    if not path_line_found:
        resolved_lines.insert(0, f"path: {dataset_dir.as_posix()}")
    RUNTIME_DATASET_YAML.write_text("\n".join(resolved_lines) + "\n", encoding="utf-8")
    return RUNTIME_DATASET_YAML


def export_backend_mapping(run_dir: Path) -> Path:
    """Save the 10-class -> 5-category backend lookup JSON inside the model run directory."""
    out_path = run_dir / "sortify_class_mapping.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(SORTIFY_CLASSES, indent=2), encoding="utf-8")
    return out_path


def write_test_evaluation_markdown(metrics, run_dir: Path, weights_path: Path) -> Path:
    """Write a clean Markdown table of overall and per-class test set metrics."""
    box = metrics.box
    mp = float(box.mp)
    mr = float(box.mr)
    map50 = float(box.map50)
    map50_95 = float(box.map)

    lines = [
        "# Sortify YOLOv11 — Held-Out Test Split Evaluation Report",
        "",
        f"- **Evaluated Checkpoint:** `{weights_path}`",
        f"- **Overall Precision (`P`):** `{mp:.4f}`",
        f"- **Overall Recall (`R`):** `{mr:.4f}`",
        f"- **Overall `mAP@50`:** **`{map50:.4f}`**",
        f"- **Overall `mAP@50-95`:** **`{map50_95:.4f}`**",
        "",
        "## Per-Class Test Metrics",
        "",
        "| Class ID | Detector Class | Backend Disposal Category | Precision | Recall | mAP@50 | mAP@50-95 |",
        "| :-: | :--- | :--- | --: | --: | --: | --: |",
    ]

    ap_class_index = list(getattr(box, "ap_class_index", []))
    for i, cid in enumerate(ap_class_index):
        cid_int = int(cid)
        cinfo = SORTIFY_CLASSES.get(cid_int, {"name": f"class_{cid_int}", "category": "Unknown"})
        p_i, r_i, ap50_i, ap_i = box.class_result(i)
        lines.append(
            f"| **`{cid_int}`** | **`{cinfo['name']}`** | **{cinfo['category']}** | "
            f"`{float(p_i):.4f}` | `{float(r_i):.4f}` | **`{float(ap50_i):.4f}`** | `{float(ap_i):.4f}` |"
        )

    md_content = "\n".join(lines) + "\n"
    run_md = run_dir / "test_evaluation_summary.md"
    run_md.write_text(md_content, encoding="utf-8")

    SUMMARIES_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_md = SUMMARIES_RESULTS_DIR / "yolo11_test_evaluation.md"
    summary_md.write_text(md_content, encoding="utf-8")
    return summary_md


def run_test_evaluation(
    YOLO_cls,
    weights_path: Path,
    runtime_data_yaml: Path,
    args: argparse.Namespace,
    run_dir: Path,
) -> None:
    print("\nEvaluating checkpoint on the held-out test split (Dataset/images/test)...")
    val_batch = 16 if args.batch == -1 else args.batch
    eval_model = YOLO_cls(str(weights_path))
    metrics = eval_model.val(
        data=str(runtime_data_yaml),
        split="test",
        imgsz=args.imgsz,
        batch=val_batch,
        device=args.device,
        workers=args.workers,
        project=str(RUNS_DIR),
        name=f"{run_dir.name}_test",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    summary_md = write_test_evaluation_markdown(metrics, run_dir, weights_path)
    print(f"Saved per-class test evaluation Markdown report to: {summary_md}")


def run_prediction(YOLO_cls, weights_path: Path, source: str, args: argparse.Namespace) -> None:
    print(f"\nRunning Sortify inference on {source!r} using {weights_path}...")
    pred_model = YOLO_cls(str(weights_path))
    results = pred_model.predict(
        source=source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(RUNS_DIR),
        name=f"{args.name}_predict",
        exist_ok=True,
        save=True,
    )
    for res in results:
        img_name = Path(res.path).name
        print(f"\nImage: {img_name}")
        if res.boxes is None or len(res.boxes) == 0:
            print("  No waste objects detected.")
            continue
        for cls_tensor, conf_tensor in zip(res.boxes.cls, res.boxes.conf):
            cid = int(cls_tensor.item())
            conf = float(conf_tensor.item())
            info = SORTIFY_CLASSES.get(cid, {"name": f"class_{cid}", "category": "Unknown", "guidance": ""})
            print(
                f"  -> Detected: {info['name']} ({conf * 100:.1f}%) | "
                f"Sortify Category: [{info['category']}] | {info['guidance']}"
            )


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data).resolve() if args.data else DATASET_YAML
    dataset_dir = validate_inputs(data_yaml)

    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError:
        print(
            "Ultralytics is not installed. Install it with:\n\n"
            "    python -m pip install -U ultralytics\n",
            file=sys.stderr,
        )
        raise SystemExit(1)

    runtime_data_yaml = create_runtime_dataset_yaml(data_yaml, dataset_dir)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    # Mode 1: Evaluation-only or Prediction-only on existing weights
    if args.eval_only:
        weights_path = Path(args.eval_only).resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
        run_dir = weights_path.parents[1] if weights_path.parent.name == "weights" else RUNS_DIR / args.name
        if not args.skip_test:
            run_test_evaluation(YOLO, weights_path, runtime_data_yaml, args, run_dir)
        if args.predict:
            run_prediction(YOLO, weights_path, args.predict, args)
        return

    # Mode 2: Full Training + Validation + Held-out Test Evaluation
    print("=" * 76)
    print("Sortify 10-Class YOLO11 Object Detector Training")
    print(f"Dataset:    {dataset_dir} ({runtime_data_yaml})")
    print(f"Base Model: {args.model}")
    print(f"Run Output: {RUNS_DIR / args.name}")
    print(f"Schedule:   {args.epochs} epochs | imgsz={args.imgsz} | batch={args.batch} | patience={args.patience}")
    print("=" * 76)

    model = YOLO(args.model)
    model.train(
        data=str(runtime_data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=args.resume,
        resume=args.resume,
        pretrained=True,
        optimizer="auto",
        cos_lr=True,
        amp=True,
        seed=42,
        deterministic=True,
        cache=args.cache,
        # Multi-domain waste augmentation settings (bridges street, bin, conveyor & tabletop domains)
        mosaic=1.0,
        close_mosaic=15,
        mixup=0.10,
        degrees=15.0,
        translate=0.10,
        scale=0.50,
        fliplr=0.50,
        flipud=0.10,
        hsv_h=0.015,
        hsv_s=0.70,
        hsv_v=0.40,
        plots=True,
        save=True,
        save_period=10,
        verbose=True,
    )

    run_dir = (
        Path(model.trainer.save_dir)
        if hasattr(model, "trainer") and hasattr(model.trainer, "save_dir")
        else (RUNS_DIR / args.name)
    )
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        print(f"Training finished, but best weights were not found at {best_weights}", file=sys.stderr)
        raise SystemExit(2)

    mapping_file = export_backend_mapping(run_dir)
    print("\n" + "=" * 76)
    print(f"Best checkpoint:          {best_weights}")
    print(f"Backend category mapping: {mapping_file}")
    print(f"Training artifacts:       {run_dir}")

    if not args.skip_test:
        run_test_evaluation(YOLO, best_weights, runtime_data_yaml, args, run_dir)

    if args.predict:
        run_prediction(YOLO, best_weights, args.predict, args)

    print("\nDone.")
    print(f"Best model ready at: {best_weights}")


if __name__ == "__main__":
    main()
