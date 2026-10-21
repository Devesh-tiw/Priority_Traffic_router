#!/usr/bin/env python3
"""
prepare_dataset.py — everything between "raw videos" and "train.py".

The repo had data.yaml but no dataset, which is why the custom model never
existed. This script makes the dataset path obvious and repeatable.

Three sub-commands
------------------
1. frames      pull stills out of the traffic videos so you have something
               to label (or to pre-label).

       python prepare_dataset.py frames --video Traffic_east_west.mp4 \
              --out dataset/raw/images --every 25

2. prelabel    semi-automatic annotation. Runs the COCO model over a folder of
               images and writes YOLO .txt labels for car / motorcycle / bus /
               truck (using the class ids from data.yaml). You then only have
               to DRAW ambulances and police cars, which is 10x faster than
               labelling every vehicle by hand.

       python prepare_dataset.py prelabel --images dataset/raw/images

3. split       validate + split into images/train, images/val, labels/train,
               labels/val and print a class-balance report.

       python prepare_dataset.py split --raw dataset/raw --out dataset \
              --val 0.2

After that:  python train.py
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# COCO ids that map onto our custom class names
COCO_TO_NAME = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


# --------------------------------------------------------------------------- #
def load_names(yaml_path: Path) -> dict[int, str]:
    data = yaml.safe_load(yaml_path.read_text())
    names = data.get("names", {})
    if isinstance(names, list):                       # allow list form
        return {i: n for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def name_to_id(names: dict[int, str]) -> dict[str, int]:
    return {v.lower(): k for k, v in names.items()}


# --------------------------------------------------------------------------- #
def cmd_frames(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = [Path(v) for v in args.video] if args.video else sorted(
        p for p in Path(".").glob("*.mp4")
    )
    if not videos:
        print("no videos found")
        return 1

    total = 0
    for vid in videos:
        cap = cv2.VideoCapture(str(vid))
        if not cap.isOpened():
            print(f"skip (cannot open): {vid}")
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, int(round(fps / args.per_second))) if args.per_second else args.every
        idx = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                name = f"{vid.stem}_{idx:06d}.jpg"
                cv2.imwrite(str(out / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
                saved += 1
            idx += 1
        cap.release()
        total += saved
        print(f"{vid.name}: {saved} frames -> {out}")
    print(f"total {total} images")
    return 0


# --------------------------------------------------------------------------- #
def cmd_prelabel(args) -> int:
    from ultralytics import YOLO

    images_dir = Path(args.images)
    labels_dir = Path(args.labels) if args.labels else images_dir.parent / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    names = load_names(Path(args.data))
    n2i = name_to_id(names)
    wanted = {cid: n2i[name] for cid, name in COCO_TO_NAME.items() if name in n2i}
    print(f"pre-labelling classes: {wanted}")

    model = YOLO(args.weights)
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    written = 0
    for i, img_path in enumerate(files, 1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        res = model.predict(img, conf=args.conf, imgsz=640, verbose=False)[0]
        lines = []
        for box in res.boxes:
            cid = int(box.cls)
            if cid not in wanted:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            lines.append(f"{wanted[cid]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (labels_dir / f"{img_path.stem}.txt").write_text("\n".join(lines))
        written += 1
        if i % 25 == 0:
            print(f"  {i}/{len(files)}")
    print(f"pre-labelled {written} images -> {labels_dir}")
    print("NEXT: open the images in labelImg / Roboflow and ADD the ambulance")
    print("      and police boxes by hand (the model cannot guess those).")
    return 0


# --------------------------------------------------------------------------- #
def cmd_split(args) -> int:
    raw = Path(args.raw)
    out = Path(args.out)
    names = load_names(Path(args.data))

    images = sorted(
        p for p in raw.rglob("*") if p.suffix.lower() in IMG_EXTS
    )
    if not images:
        print(f"no images under {raw}")
        return 1

    pairs: list[tuple[Path, Path]] = []
    missing = 0
    for img in images:
        lbl = raw / "labels" / f"{img.stem}.txt"
        if not lbl.exists():
            # also support a flat layout where .txt sits next to the image
            alt = img.with_suffix(".txt")
            lbl = alt if alt.exists() else lbl
        if lbl.exists():
            pairs.append((img, lbl))
        else:
            missing += 1

    print(f"images: {len(images)}  with labels: {len(pairs)}  unlabelled: {missing}")
    if not pairs:
        print("nothing to split")
        return 1

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * args.val)) if len(pairs) > 4 else 0
    val = pairs[:n_val]
    train = pairs[n_val:]

    for split, items in (("train", train), ("val", val)):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, lbl in items:
            shutil.copy2(img, out / "images" / split / img.name)
            shutil.copy2(lbl, out / "labels" / split / f"{img.stem}.txt")

    # class balance report
    counts = Counter()
    for _, lbl in pairs:
        for line in lbl.read_text().splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
    print("\nclass balance")
    for cid, name in sorted(names.items()):
        print(f"  {cid} {name:<12} {counts.get(cid, 0):>6} boxes")
    if counts.get(0, 0) == 0:
        print("\nWARNING: zero 'ambulance' boxes -> the model will never learn it.")

    # rewrite data.yaml with the real path
    yaml_path = out / "data.yaml"
    yaml_path.write_text(yaml.safe_dump({
        "path": str(out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": {k: v for k, v in sorted(names.items())},
    }, sort_keys=False))
    print(f"\nwrote {yaml_path}")
    print(f"train {len(train)} / val {len(val)}")
    print("NEXT: python train.py")
    return 0


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data.yaml", help="class definition file")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("frames", help="extract stills from videos")
    f.add_argument("--video", action="append", help="video file (repeatable)")
    f.add_argument("--out", default="dataset/raw/images")
    f.add_argument("--per-second", type=float, default=1.0,
                   help="frames to save per second of video")
    f.add_argument("--every", type=int, default=25, help="fallback: 1 frame every N")
    f.set_defaults(func=cmd_frames)

    p = sub.add_parser("prelabel", help="auto-label cars with COCO weights")
    p.add_argument("--images", default="dataset/raw/images")
    p.add_argument("--labels", default=None)
    p.add_argument("--weights", default="yolov8n.pt")
    p.add_argument("--conf", type=float, default=0.4)
    p.set_defaults(func=cmd_prelabel)

    s = sub.add_parser("split", help="validate + split into train/val")
    s.add_argument("--raw", default="dataset/raw")
    s.add_argument("--out", default="dataset")
    s.add_argument("--val", type=float, default=0.2)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_split)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
