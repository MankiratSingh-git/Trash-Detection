#!/usr/bin/env python3
#
# Train an Ultralytics YOLO11 detector on the adjacent Dataset-Augmented directory.
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "Dataset-Augmented"
DATASET_YAML = DATASET_DIR / "data.yaml"
RUNTIME_DATASET_YAML = Path(__file__).resolve().parent / "dataset.runtime.yaml"
RUNS_DIR = Path(__file__).resolve().parent / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLO11 trash detector with live terminal progress."
    )
    parser.add_argument("--model", default="yolo11s.pt", help="YOLO11 checkpoint or model YAML (default: yolo11s.pt)")
    parser.add_argument("--data", default=None, help="Dataset data.yaml; defaults to Dataset-Augmented/data.yaml")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs (default: 100)")
    parser.add_argument("--imgsz", type=int, default=640, help="Square input size (default: 640)")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size; -1 lets Ultralytics choose automatically (default: -1)")
    parser.add_argument("--device", default=None, help="CUDA device, 'cpu', or omit for automatic selection")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers (default: 4)")
    parser.add_argument("--patience", type=int, default=30, help="Early-stopping patience in epochs (default: 30)")
    parser.add_argument("--name", default="trash_yolo11_augmented", help="Run folder name under Model/runs (default: trash_yolo11_augmented)")
    parser.add_argument("--resume", action="store_true", help="Resume the named run from its last checkpoint")
    parser.add_argument("--skip-test", action="store_true", help="Do not evaluate best.pt on Dataset-Augmented/images/test after training")
    return parser.parse_args()


def validate_inputs(data_yaml: Path) -> Path:
    dataset_dir = data_yaml.parent
    if not data_yaml.is_file():
        raise FileNotFoundError(f"Dataset YAML was not found: {data_yaml}")
    for split in ("train", "val", "test"):
        image_dir = dataset_dir / "images" / split
        label_dir = dataset_dir / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing YOLO {split!r} image or label directory under {dataset_dir}")
    return dataset_dir


def create_runtime_dataset_yaml(data_yaml: Path, dataset_dir: Path) -> Path:
    """Resolve `path: .` to the real dataset directory for Ultralytics."""
    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    path_line_found = False
    resolved_lines = []
    for line in lines:
        if line.strip().startswith("path:"):
            resolved_lines.append(f"path: {dataset_dir.as_posix()}")
            path_line_found = True
        else:
            resolved_lines.append(line)
    if not path_line_found:
        raise ValueError(f"Dataset YAML has no 'path:' field: {data_yaml}")
    RUNTIME_DATASET_YAML.write_text("\n".join(resolved_lines) + "\n", encoding="utf-8")
    return RUNTIME_DATASET_YAML


def main() -> None:
    args = parse_args()
    data_yaml = Path(args.data).resolve() if args.data else DATASET_YAML
    dataset_dir = validate_inputs(data_yaml)

    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError:
        print("Ultralytics is not installed. Install it with:\n\n    python -m pip install -U ultralytics\n", file=sys.stderr)
        raise SystemExit(1)

    runtime_data_yaml = create_runtime_dataset_yaml(data_yaml, dataset_dir)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 72)
    print("YOLO11 trash-detector training")
    print(f"Dataset: {dataset_dir}")
    print(f"Model:   {args.model}")
    print(f"Output:  {RUNS_DIR / args.name}")
    print(f"Epochs:  {args.epochs} | Image size: {args.imgsz} | Batch: {args.batch}")
    print("Live batch/epoch progress and validation metrics follow below.")
    print("=" * 72)

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
        seed=42,
        deterministic=True,
        cache=False,
        plots=True,
        save=True,
        save_period=10,
        verbose=True,
    )

    # Use the actual trainer save_dir if available (handles auto-incremented names like run-2)
    run_dir = Path(model.trainer.save_dir) if hasattr(model, "trainer") and hasattr(model.trainer, "save_dir") else (RUNS_DIR / args.name)
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        print(f"Training finished, but best weights were not found at {best_weights}", file=sys.stderr)
        raise SystemExit(2)

    print("\n" + "=" * 72)
    print(f"Best checkpoint: {best_weights}")
    print(f"Training artifacts: {run_dir}")

    if not args.skip_test:
        print("\nEvaluating best.pt on the held-out test split...")
        val_batch = 16 if args.batch == -1 else args.batch
        YOLO(str(best_weights)).val(
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

    print("\nDone.")
    print(f"Best model: {best_weights}")


if __name__ == "__main__":
    main()
