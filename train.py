#!/usr/bin/env python3


from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the ambulance/police detector")
    ap.add_argument("--data", default="dataset/data.yaml", help="dataset yaml")
    ap.add_argument("--weights", default="yolov8n.pt", help="base model to fine-tune")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="auto", help="auto | cpu | 0 | mps")
    ap.add_argument("--name", default="smart_traffic_model")
    ap.add_argument("--patience", type=int, default=25, help="early stopping patience")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    data_path = Path(args.data)
    if not data_path.exists() and Path("data.yaml").exists():
        data_path = Path("data.yaml")
    if not data_path.exists():
        print(f"ERROR: dataset yaml not found ({args.data}).")
        print("       Run: python prepare_dataset.py split")
        return 1

    raw = data_path.read_text()
    for line in raw.splitlines():
        if line.strip().startswith("path:"):
            root = Path(line.split(":", 1)[1].strip().strip("'\""))
            train_dir = root / "images" / "train"
            if not train_dir.exists():
                print(f"ERROR: no training images at {train_dir}")
                print("       Run: python prepare_dataset.py split")
                return 1
            n_imgs = len(list(train_dir.glob('*.*')))
            print(f"[train] dataset {root}  ({n_imgs} training images)")

    device = pick_device(args.device)
    if device == "cpu":
        print("[train] WARNING: no GPU detected. 100 epochs on CPU can take HOURS.")
        print("[train]          Use notebooks/train_colab.ipynb (free T4 GPU) instead.")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        patience=args.patience,
        name=args.name,
        resume=args.resume,
        # night footage: be aggressive with lighting augmentations
        hsv_h=0.02, hsv_s=0.7, hsv_v=0.5,
        degrees=0.0, translate=0.1, scale=0.5, fliplr=0.5,
        mosaic=1.0, mixup=0.0,
        plots=True,
        val=True,
    )

    best = Path("runs/detect") / args.name / "weights" / "best.pt"
    if best.exists():
        dest = Path("models") / "best.pt"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dest)
        print(f"\n[train] best weights -> {best}")
        print(f"[train] copied to    -> {dest}")
        print("[train] main.py now picks this model up automatically.")
    else:
        print("[train] could not find best.pt — check the run folder")

    try:
        metrics = results.results_dict
        print("\n[train] validation")
        for k, v in metrics.items():
            print(f"   {k:<28} {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
