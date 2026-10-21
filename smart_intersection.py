"""
smart_intersection.py — the engine that ties everything together.

    Lane               one camera / video feed
    EmergencySimulator injects a fake ambulance (for demos, and for the
                       pretrained-COCO mode which has no ambulance class)
    GridRenderer       draws tiles, traffic lights, HUD and the 2x2 grid
    SmartIntersectionAI the object that runs the whole junction

Run it through main.py (CLI) or web/dashboard.py (browser).
"""

from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

import config as cfg
from controller import Decision, SignalController
from detector import Detector, LaneDetection


# --------------------------------------------------------------------------- #
# Lane
# --------------------------------------------------------------------------- #
class Lane:
    """One approach to the junction (one camera or one video file)."""

    def __init__(
        self,
        name: str,
        source: str | int,
        offset: float = 0.0,
        tile_size: Tuple[int, int] = cfg.TILE_SIZE,
        infer_size: Tuple[int, int] = cfg.INFER_SIZE,
    ) -> None:
        self.name = name
        self.source = source
        self.offset = float(offset)
        self.tile_size = tile_size
        self.infer_size = infer_size

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.detection = LaneDetection()
        self.count = 0
        self.priority_count = 0
        self.priority_labels: List[str] = []
        self.simulated = False
        self.light = "RED"
        self.is_camera = isinstance(source, int) or str(source).isdigit()
        self.loops = 0
        self._infer_cache: Optional[np.ndarray] = None

    # -- lifecycle ------------------------------------------------------ #
    def open(self) -> bool:
        src = int(self.source) if self.is_camera else str(self.source)
        if self.is_camera:
            self.cap = cv2.VideoCapture(int(src))
        else:
            self.cap = cv2.VideoCapture(str(src))

        if not self.cap or not self.cap.isOpened():
            raise RuntimeError(f"[{self.name}] cannot open source: {self.source}")

        if not self.is_camera and self.offset > 0:
            total = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            if total > 0:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * self.offset))
        return True

    def read(self) -> bool:
        """Read the next frame. Video files loop forever, so demos never stall."""
        if self.cap is None:
            return False
        ok, frame = self.cap.read()
        if not ok:
            if self.is_camera:
                return False
            # rewind and try once more (file loop)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int((self.cap.get(7) or 0) * self.offset))
            ok, frame = self.cap.read()
            self.loops += 1
            if not ok:
                return False
        self.frame = frame
        self._infer_cache = None
        return True

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    # -- helpers -------------------------------------------------------- #
    def infer_frame(self) -> np.ndarray:
        """Downscaled frame fed to YOLO. Cached per source frame (resizing a
        1440p frame is not free, so we do it once and reuse it for display)."""
        assert self.frame is not None
        if self._infer_cache is None:
            self._infer_cache = cv2.resize(
                self.frame, self.infer_size, interpolation=cv2.INTER_AREA
            )
        return self._infer_cache

    def tile_frame(self) -> np.ndarray:
        """Small frame for the dashboard tile, derived from the cached one."""
        return cv2.resize(self.infer_frame(), self.tile_size, interpolation=cv2.INTER_AREA)

    def reset_detection(self) -> None:
        self.detection = LaneDetection()
        self.count = 0
        self.priority_count = 0
        self.priority_labels = []


# --------------------------------------------------------------------------- #
# Emergency simulation
# --------------------------------------------------------------------------- #
class EmergencySimulator:
    """
    Pretends an ambulance is present on a lane.

    Needed because the shipped demo clips contain no ambulance, and because
    the pretrained COCO model has no ambulance class at all. With a trained
    models/best.pt the real detections take over and this is only used for
    the dashboard's manual button.
    """

    def __init__(self) -> None:
        self._manual: Dict[str, float] = {}     # lane -> absolute expiry time
        self._schedule: List[Tuple[str, float, float]] = []   # relative seconds
        self._t0: Optional[float] = None        # engine start, set on first use

    # -- manual --------------------------------------------------------- #
    def trigger(self, lane: str, seconds: float = 6.0, now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        self._manual[lane] = max(self._manual.get(lane, 0.0), now + seconds)

    def clear(self) -> None:
        self._manual.clear()

    # -- scripted (demo video / repeatable screenshots) ----------------- #
    def schedule(self, lane: str, start: float, duration: float = 6.0) -> None:
        self._schedule.append((lane, start, start + duration))

    def start(self, now: float) -> None:
        """Remember the engine's start time so scheduled demo events are relative."""
        if self._t0 is None:
            self._t0 = now

    def active(self, lane: str, now: float) -> bool:
        if self._manual.get(lane, 0.0) > now:
            return True
        if not self._schedule or self._t0 is None:
            return False
        rel = now - self._t0
        return any(l == lane and s <= rel < e for l, s, e in self._schedule)

    def active_lanes(self, now: float, lanes: Sequence[str]) -> List[str]:
        return [l for l in lanes if self.active(l, now)]


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
class GridRenderer:
    """All drawing lives here so the engine stays readable."""

    def __init__(self, tile_size: Tuple[int, int] = cfg.TILE_SIZE) -> None:
        self.tw, self.th = tile_size
        self.gap = 4
        self.header_h = cfg.HEADER_HEIGHT
        self.blink_on = True          # updated once per frame by the engine

    # -- small helpers --------------------------------------------------- #
    @staticmethod
    def _panel(img: np.ndarray, p1, p2, color, alpha: float = 0.55) -> None:
        overlay = img.copy()
        cv2.rectangle(overlay, p1, p2, color, -1)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def _text(self, img, text, org, scale=0.45, color=cfg.COLOR_TEXT, thick=1) -> None:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, thick, cv2.LINE_AA)

    def _traffic_light(self, img: np.ndarray, state: str) -> None:
        """3-lamp signal in the top-right corner of the tile."""
        w, h, r = 34, 94, 11
        x2 = self.tw - 12
        x1 = x2 - w
        y1 = 40
        self._panel(img, (x1, y1), (x2, y1 + h), (18, 18, 18), 0.78)
        cv2.rectangle(img, (x1, y1), (x2, y1 + h), (70, 70, 70), 1)
        lamps = [("RED", cfg.COLOR_RED), ("AMBER", cfg.COLOR_AMBER), ("GREEN", cfg.COLOR_GREEN)]
        for i, (name, col) in enumerate(lamps):
            cy = y1 + 16 + i * 31
            on = (name == state) or (state == "GREEN" and name == "GREEN")
            colour = col if on else (45, 45, 45)
            cv2.circle(img, (x1 + w // 2, cy), r, colour, -1, cv2.LINE_AA)
            if on:
                glow = img.copy()
                cv2.circle(glow, (x1 + w // 2, cy), r + 6, colour, -1, cv2.LINE_AA)
                cv2.addWeighted(glow, 0.22, img, 0.78, 0, img)
            cv2.circle(img, (x1 + w // 2, cy), r, (85, 85, 85), 1, cv2.LINE_AA)

    # -- one tile -------------------------------------------------------- #
    def draw_tile(
        self,
        lane: Lane,
        state: str,
        now: float,
        reason: str = "",
        emergency: bool = False,
    ) -> np.ndarray:
        tile = lane.tile_frame().copy()
        if state != "GREEN":
            # dim lanes that are held at red so the green lane pops
            tile = cv2.convertScaleAbs(tile, alpha=0.74, beta=-8)
        sx = self.tw / lane.infer_size[0]
        sy = self.th / lane.infer_size[1]

        # bounding boxes (coords come from the inference resolution)
        for (x1, y1, x2, y2), name, score, is_priority in lane.detection.boxes:
            p1 = (int(x1 * sx), int(y1 * sy))
            p2 = (int(x2 * sx), int(y2 * sy))
            colour = cfg.COLOR_PRIORITY if is_priority else cfg.COLOR_VEHICLE
            thick = 3 if is_priority else 1
            cv2.rectangle(tile, p1, p2, colour, thick, cv2.LINE_AA)
            # in a busy night scene every tiny car would print a label and the
            # tile becomes unreadable: label only what matters
            box_h = p2[1] - p1[1]
            if not (is_priority or score >= 0.55 or box_h >= 46):
                continue
            label = f"{name.upper() if is_priority else name} {score:.2f}"
            scale = 0.42 if is_priority else 0.36
            (tw_, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            ly = max(p1[1] - 3, th_ + 4)
            cv2.rectangle(tile, (p1[0], ly - th_ - 3), (p1[0] + tw_ + 5, ly + 2), colour, -1)
            cv2.putText(
                tile, label, (p1[0] + 2, ly - 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 15, 15), 1, cv2.LINE_AA,
            )

        # top strip: lane name + counts
        self._panel(tile, (0, 0), (self.tw, 30), (0, 0, 0), 0.55)
        accent = cfg.COLOR_GREEN if state == "GREEN" else cfg.COLOR_RED
        cv2.rectangle(tile, (0, 0), (6, 30), accent, -1)
        self._text(tile, lane.name, (14, 21), 0.62, cfg.COLOR_TEXT, 1)
        tag = f"{lane.count} veh"
        if lane.priority_count:
            tag += f"   {lane.priority_count} PRIORITY"
        self._text(tile, tag, (118, 21), 0.48,
                   cfg.COLOR_PRIORITY if lane.priority_count else cfg.COLOR_DIM, 1)

        self._traffic_light(tile, state)

        # bottom banner: GREEN / RED + reason
        self._panel(tile, (0, self.th - 34), (self.tw, self.th), (0, 0, 0), 0.62)
        cv2.rectangle(tile, (0, self.th - 34), (5, self.th), accent, -1)
        self._text(tile, state, (14, self.th - 12), 0.58, accent, 1)
        sub = "VEHICLES MOVING" if state == "GREEN" else "STOP"
        self._text(tile, sub, (88, self.th - 12), 0.38, cfg.COLOR_DIM, 1)

        # emergency overlay
        if emergency:
            if self.blink_on:
                self._panel(tile, (6, 40), (self.tw - 8, 96), (25, 8, 22), 0.82)
                cv2.rectangle(tile, (6, 40), (self.tw - 8, 96), cfg.COLOR_PRIORITY, 2, cv2.LINE_AA)
                self._text(tile, "EMERGENCY", (16, 68), 0.78, cfg.COLOR_PRIORITY, 2)
                sub = ("SIMULATED AMBULANCE" if lane.simulated and not lane.detection.priority_count
                       else "PRIORITY VEHICLE DETECTED")
                self._text(tile, sub, (16, 88), 0.4, cfg.COLOR_TEXT, 1)
            cv2.rectangle(tile, (0, 0), (self.tw - 1, self.th - 1), cfg.COLOR_PRIORITY, 3)

        return tile

    # -- header + grid --------------------------------------------------- #
    def draw_header(
        self,
        green: str,
        reason: str,
        total: int,
        fps: float,
        emergency: bool,
        mode: str,
        now: float,
    ) -> np.ndarray:
        w = self.tw * 2 + self.gap
        bar = np.zeros((self.header_h, w, 3), np.uint8)
        bar[:] = cfg.COLOR_PANEL

        mid = self.header_h // 2 + 6
        cv2.circle(bar, (20, self.header_h // 2), 7,
                   cfg.COLOR_PRIORITY if emergency else cfg.COLOR_GREEN, -1, cv2.LINE_AA)

        # right-hand meta block is placed first so the middle text knows its budget
        meta = f"{total} veh | {fps:4.1f} fps | {mode}"
        (mw, _), _ = cv2.getTextSize(meta, cv2.FONT_HERSHEY_DUPLEX, 0.4, 1)
        meta_x = w - mw - 14
        self._text(bar, meta, (meta_x, mid), 0.4, cfg.COLOR_DIM, 1)

        def width(text, scale):
            return cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 1)[0][0]

        x = 36
        title = "SMART INTERSECTION AI"
        self._text(bar, title, (x, mid), 0.52, cfg.COLOR_TEXT, 1)
        x += width(title, 0.52) + 26

        status = f"GREEN: {green}"
        self._text(bar, status, (x, mid), 0.5,
                   cfg.COLOR_GREEN if not emergency else cfg.COLOR_PRIORITY, 1)
        x += width(status, 0.5) + 26

        detail = reason
        if emergency and not reason.upper().startswith("EMERGENCY"):
            detail = f"OVERRIDE - {reason}"
        budget = meta_x - x - 24
        scale = 0.4
        while detail and width(detail, scale) > budget:
            detail = detail[:-1]
        if detail and detail != reason and not detail.endswith("."):
            detail = detail[:-2] + ".." if len(detail) > 3 else detail
        self._text(bar, detail, (x, mid), scale, cfg.COLOR_DIM, 1)
        return bar

    def compose(self, header: np.ndarray, tiles: Sequence[np.ndarray]) -> np.ndarray:
        if len(tiles) != 4:
            raise ValueError("expected exactly 4 tiles")
        row1 = np.hstack((tiles[0], np.zeros((self.th, self.gap, 3), np.uint8), tiles[1]))
        row2 = np.hstack((tiles[2], np.zeros((self.th, self.gap, 3), np.uint8), tiles[3]))
        spacer = np.zeros((self.gap, row1.shape[1], 3), np.uint8)
        return np.vstack((header, row1, spacer, row2))


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class SmartIntersectionAI:
    """The junction: lanes + detector + controller + renderer."""

    def __init__(
        self,
        sources: Optional[Sequence[str | int]] = None,
        offsets: Optional[Sequence[float]] = None,
        weights: Optional[str | Path] = None,
        conf: float = cfg.CONF_THRESHOLD,
        imgsz: int = cfg.IMG_SIZE,
        detect_every: int = cfg.DETECT_EVERY_N_FRAMES,
        min_green: float = cfg.MIN_GREEN_SECONDS,
        emergency_hold: float = cfg.EMERGENCY_HOLD_SECONDS,
        switch_margin: int = cfg.SWITCH_MARGIN,
        device: Optional[str] = None,
        log_csv: Optional[str | Path] = None,
        clock: Callable[[], float] = time.monotonic,
        verbose: bool = True,
    ) -> None:
        # ---- weights: prefer a trained custom model when present ---- #
        if weights is None:
            weights = cfg.CUSTOM_WEIGHTS if cfg.CUSTOM_WEIGHTS.exists() else cfg.DEFAULT_WEIGHTS
        self.weights = Path(weights)

        sources = tuple(sources) if sources else cfg.DEFAULT_SOURCES
        if len(sources) != 4:
            raise ValueError("need exactly 4 sources (N, S, E, W)")
        offsets = tuple(offsets) if offsets else (cfg.DEFAULT_OFFSETS if sources == cfg.DEFAULT_SOURCES
                                                  else (0.0, 0.0, 0.0, 0.0))

        self.lanes = [
            Lane(name, src, off)
            for name, src, off in zip(cfg.LANE_NAMES, sources, offsets)
        ]
        for lane in self.lanes:
            lane.open()

        self.detector = Detector(weights=self.weights, conf=conf, imgsz=imgsz,
                                 device=device, verbose=verbose)
        self.controller = SignalController(
            cfg.LANE_NAMES,
            min_green_s=min_green,
            emergency_hold_s=emergency_hold,
            switch_margin=switch_margin,
        )
        self.renderer = GridRenderer()
        self.simulator = EmergencySimulator()
        self.detect_every = max(1, int(detect_every))
        self.clock = clock
        self.frame_index = 0
        self.fps = 0.0
        self._last_t = None
        self.decision = Decision(cfg.LANE_NAMES[0], "startup")
        self.grid: Optional[np.ndarray] = None
        self.mode = "CUSTOM model" if self.detector.is_custom else "COCO + simulate"

        self._csv_fh = None
        self._csv_writer = None
        self._csv_path = Path(log_csv) if log_csv else None
        self._open_log()

    # ------------------------------------------------------------------ #
    def _open_log(self) -> None:
        if not self._csv_path:
            return
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_fh = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_fh)
        self._csv_writer.writerow(
            ["frame", "time_s", "lane", "vehicles", "priority", "light", "green_lane", "reason"]
        )

    def _log(self, now: float) -> None:
        if not self._csv_writer:
            return
        for lane in self.lanes:
            self._csv_writer.writerow([
                self.frame_index, f"{now:.2f}", lane.name, lane.count, lane.priority_count,
                lane.light, self.decision.green, self.decision.reason,
            ])
        self._csv_fh.flush()

    # ------------------------------------------------------------------ #
    def step(self) -> np.ndarray:
        """Read a frame from every lane, decide, render. Returns the 2x2 grid."""
        if self._last_t is not None:
            dt = max(1e-3, self.clock() - self._last_t)
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
        self._last_t = self.clock()

        now = self.clock()
        self.simulator.start(now)

        # ---- 1. LOOK + FIND ---------------------------------------- #
        for lane in self.lanes:
            if not lane.read():
                lane.reset_detection()
                continue
            if self.frame_index % self.detect_every == 0:
                lane.detection = self.detector.detect(lane.infer_frame())
                lane.count = lane.detection.count
                lane.priority_count = lane.detection.priority_count
                lane.priority_labels = lane.detection.priority_labels

            lane.simulated = self.simulator.active(lane.name, now)
            if lane.simulated and lane.priority_count == 0:
                # simulation counts as one priority vehicle
                lane.count = max(lane.count, 1)
                lane.priority_count = 1
                lane.priority_labels = ["ambulance (sim)"]

        # ---- 2. DECIDE --------------------------------------------- #
        counts = {l.name: l.count for l in self.lanes}
        priority = {l.name: l.priority_count for l in self.lanes}
        self.decision = self.controller.decide(counts, priority, now)

        # ---- 3. SHOW ------------------------------------------------ #
        self.renderer.blink_on = self.frame_index % 6 < 3    # ~2.5 blinks/s
        tiles = []
        for lane in self.lanes:
            state = "GREEN" if lane.name == self.decision.green else "RED"
            lane.light = state
            tiles.append(
                self.renderer.draw_tile(
                    lane, state, now,
                    reason=self.decision.reason,
                    emergency=bool(lane.priority_count) and state == "GREEN",
                )
            )

        header = self.renderer.draw_header(
            self.decision.green, self.decision.reason,
            total=sum(counts.values()), fps=self.fps,
            emergency=self.decision.emergency, mode=self.mode, now=now,
        )
        self.grid = self.renderer.compose(header, tiles)
        self._log(now)
        self.frame_index += 1
        return self.grid

    # ------------------------------------------------------------------ #
    def run(
        self,
        show: bool = True,
        save_path: Optional[str | Path] = None,
        max_frames: Optional[int] = None,
        on_frame: Optional[Callable[[np.ndarray, int], bool]] = None,
        quiet: bool = False,
    ) -> None:
        writer = None
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            h, w = self.grid.shape[:2] if self.grid is not None else (
                cfg.TILE_SIZE[1] * 2 + cfg.HEADER_HEIGHT + 4, cfg.TILE_SIZE[0] * 2 + 4)
            # Prefer H.264 (plays in browsers); fall back to mp4v, which every
            # OpenCV build can write. Re-encode afterwards if your viewers are
            # browser-based:  ffmpeg -i demo.mp4 -c:v libx264 -pix_fmt yuv420p out.mp4
            writer = None
            for fourcc_id in ("avc1", "mp4v"):
                w_try = cv2.VideoWriter(str(save_path), cv2.VideoWriter_fourcc(*fourcc_id),
                                        cfg.FPS_TARGET, (w, h))
                if w_try.isOpened():
                    writer = w_try
                    if fourcc_id != "avc1":
                        print("[main] note: H.264 encoder unavailable, wrote mp4v "
                              "(browsers may not play it; re-encode with ffmpeg)")
                    break
                w_try.release()
            if writer is None:
                raise RuntimeError(f"could not open a video writer for {save_path}")

        try:
            while max_frames is None or self.frame_index < max_frames:
                grid = self.step()
                if writer:
                    writer.write(grid)
                if on_frame and on_frame(grid, self.frame_index) is False:
                    break
                if show:
                    cv2.imshow("Smart Intersection AI  --  press Q to quit", grid)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    # hotkeys 1-4: simulate an ambulance on that lane
                    if key in (ord("1"), ord("2"), ord("3"), ord("4")):
                        lane = cfg.LANE_NAMES[key - ord("1")]
                        self.simulator.trigger(lane, 6.0, now=self.clock())
                        print(f"[main] simulated ambulance -> {lane}")
        except KeyboardInterrupt:
            if not quiet:
                print("\n[main] interrupted by user")
        finally:
            self.release()
            if writer:
                writer.release()
                if not quiet:
                    print(f"[main] demo video saved -> {save_path}")

    # ------------------------------------------------------------------ #
    def release(self) -> None:
        for lane in self.lanes:
            lane.release()
        cv2.destroyAllWindows()
        if self._csv_fh:
            self._csv_fh.close()
            self._csv_fh = None
            if self._csv_path:
                print(f"[main] decision log -> {self._csv_path}")
