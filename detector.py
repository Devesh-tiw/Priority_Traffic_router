"""
detector.py — the "eyes".

Wraps Ultralytics YOLOv8 and turns raw detections into something the
decision logic can reason about:

    * how many vehicles are in this lane
    * is there an ambulance / police / fire vehicle (a "priority" vehicle)
    * a list of boxes ready to be drawn

Two modes, chosen automatically:

    CUSTOM   models/best.pt exists  -> class names come from YOUR dataset
             (data.yaml: ambulance, police, car, motorcycle), so
             'ambulance' is a real class.
    COCO     only yolov8n.pt        -> COCO has NO ambulance/police class,
             so priority vehicles can only come from --simulate or the
             dashboard's "Trigger ambulance" button. Counting still works
             (car / motorcycle / bus / truck).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import config as cfg


@dataclass
class LaneDetection:
    """Result of one inference pass on one lane."""

    count: int = 0                                  # vehicles (incl. priority)
    priority_count: int = 0                         # ambulances / police / fire
    priority_labels: List[str] = field(default_factory=list)
    boxes: List[Tuple[Tuple[int, int, int, int], str, float, bool]] = field(
        default_factory=list
    )
    raw_names: List[str] = field(default_factory=list)


class Detector:
    """Thin, well-behaved wrapper around the YOLO model."""

    def __init__(
        self,
        weights: Optional[Path | str] = None,
        conf: float = cfg.CONF_THRESHOLD,
        iou: float = cfg.IOU_THRESHOLD,
        imgsz: int = cfg.IMG_SIZE,
        device: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        from ultralytics import YOLO  # lazy: keeps `unit tests` torch-free

        self.weights = Path(weights) if weights else None
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = device

        if self.weights is None or not self.weights.exists():
            raise FileNotFoundError(f"Model weights not found: {self.weights}")

        self.model = YOLO(str(self.weights))
        self.names: Dict[int, str] = {
            int(k): str(v) for k, v in (self.model.names or {}).items()
        }

        self.priority_ids = {
            i for i, n in self.names.items() if self._is_priority_name(n)
        }
        self.vehicle_ids = {
            i for i, n in self.names.items() if self._is_vehicle_name(n)
        }

        # A custom 4-class traffic model: every class is traffic, so anything
        # not matched by keyword is still counted rather than silently dropped.
        self.is_custom = self._looks_custom()
        self.count_everything = self.is_custom and len(self.vehicle_ids) < len(self.names)

        if verbose:
            print(f"[detector] weights      : {self.weights}")
            print(f"[detector] mode         : {'CUSTOM' if self.is_custom else 'COCO (pretrained)'}")
            pri = {i: self.names[i] for i in sorted(self.priority_ids)}
            print(f"[detector] classes      : {len(self.names)} -> {list(self.names.values())[:8]}"
                  + (" ..." if len(self.names) > 8 else ""))
            print(f"[detector] priority     : {pri or 'none'}")
            if not self.priority_ids:
                print("[detector] NOTE: no ambulance/police class in this model.")
                print("[detector]       Use --simulate NORTH or the dashboard button to")
                print("[detector]       demo emergency preemption, or train models/best.pt.")

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _is_priority_name(name: str) -> bool:
        n = name.lower()
        if any(k in n for k in cfg.NON_PRIORITY_KEYWORDS):
            return False
        return any(k in n for k in cfg.PRIORITY_KEYWORDS)

    @staticmethod
    def _is_vehicle_name(name: str) -> bool:
        n = name.lower()
        return any(k in n for k in cfg.VEHICLE_KEYWORDS)

    def _looks_custom(self) -> bool:
        """True when the loaded weights are not the stock 80-class COCO model."""
        return len(self.names) != 80

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> LaneDetection:
        """Run one inference pass and classify what was found."""
        out = LaneDetection()
        if frame is None or frame.size == 0:
            return out

        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return out

        boxes = results[0].boxes
        if boxes is None:
            return out

        for box in boxes:
            cls_id = int(box.cls)
            name = self.names.get(cls_id, str(cls_id))
            score = float(box.conf)
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())

            is_priority = cls_id in self.priority_ids
            is_vehicle = (
                is_priority
                or cls_id in self.vehicle_ids
                or self.count_everything
            )
            if not is_vehicle:
                continue

            out.count += 1
            out.raw_names.append(name)
            if is_priority:
                out.priority_count += 1
                out.priority_labels.append(name)
            out.boxes.append(((x1, y1, x2, y2), name, score, is_priority))

        return out
