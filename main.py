#!/usr/bin/env python3
"""
main.py — start here.

Examples
--------
python main.py                              # the shipped demo clips, live window
python main.py --save demo/demo.mp4         # render a demo video (no window)
python main.py --simulate NORTH --simulate EAST:20:8 --headless --save demo/x.mp4
python main.py --source 0 --source 0 --source 1 --source 1   # 2 webcams x2 lanes
python main.py --source a.mp4,b.mp4,c.mp4,d.mp4               # four separate clips
python main.py --weights models/best.pt                       # trained ambulance model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import config as cfg


def parse_simulate(values) -> list[tuple[str, float, float]]:
    """
    Accepts  NORTH        -> ambulance on NORTH at t=2s for 6s
             NORTH:20:8   -> lane : start second : duration
    """
    out = []
    for raw in values or []:
        parts = raw.split(":")
        lane = parts[0].upper()
        if lane not in cfg.LANE_NAMES:
            raise SystemExit(f"--simulate: unknown lane '{lane}'. Use one of {cfg.LANE_NAMES}")
        start = float(parts[1]) if len(parts) > 1 else 2.0
        dur = float(parts[2]) if len(parts) > 2 else 6.0
        out.append((lane, start, dur))
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Smart Intersection AI - priority traffic router",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--weights", default=None,
                   help="model weights (default: models/best.pt if it exists, else yolov8n.pt)")
    p.add_argument("--source", action="append", default=None,
                   help="lane source, repeat 4 times: video path or camera index. "
                        "Comma separated also works. Default: shipped demo clips.")
    p.add_argument("--offset", default=None,
                   help="comma separated start fractions, e.g. 0,0.4,0.15,0.6")
    p.add_argument("--conf", type=float, default=cfg.CONF_THRESHOLD, help="confidence threshold")
    p.add_argument("--imgsz", type=int, default=cfg.IMG_SIZE, help="inference resolution")
    p.add_argument("--detect-every", type=int, default=cfg.DETECT_EVERY_N_FRAMES,
                   help="run YOLO every Nth frame (raise it on slow CPUs)")
    p.add_argument("--min-green", type=float, default=cfg.MIN_GREEN_SECONDS,
                   help="minimum green time per lane in seconds")
    p.add_argument("--emergency-hold", type=float, default=cfg.EMERGENCY_HOLD_SECONDS,
                   help="seconds the emergency lane stays green after the vehicle leaves")
    p.add_argument("--switch-margin", type=int, default=cfg.SWITCH_MARGIN,
                   help="a lane must lead by this many vehicles to take the green")
    p.add_argument("--simulate", action="append", default=[],
                   help="demo ambulance: LANE or LANE:START:DURATION (repeatable)")
    p.add_argument("--save", default=None, help="write the 2x2 dashboard to this .mp4")
    p.add_argument("--frames", type=int, default=None, help="stop after N frames")
    p.add_argument("--log", default=None, help="write a per-frame decision log to this .csv")
    p.add_argument("--headless", action="store_true",
                   help="do not open a window (use with --save)")
    p.add_argument("--device", default=None, help="cpu, cuda, 0, mps ...")
    return p


def resolve_sources(values) -> tuple[tuple, tuple]:
    if not values:
        return cfg.DEFAULT_SOURCES, cfg.DEFAULT_OFFSETS
    flat: list[str] = []
    for v in values:
        flat.extend([x.strip() for x in v.split(",") if x.strip()])
    if len(flat) != 4:
        raise SystemExit(f"--source needs exactly 4 entries (got {len(flat)}): N,S,E,W")
    sources = tuple(int(s) if s.isdigit() else s for s in flat)
    offsets = (0.0, 0.0, 0.0, 0.0) if all(isinstance(s, int) for s in sources) else cfg.DEFAULT_OFFSETS
    return sources, offsets


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    sources, offsets = resolve_sources(args.source)
    if args.offset:
        offsets = tuple(float(x) for x in args.offset.split(","))
        if len(offsets) != 4:
            raise SystemExit("--offset needs 4 comma separated numbers")

    # imports after arg parsing so --help is instant
    from smart_intersection import SmartIntersectionAI

    engine = SmartIntersectionAI(
        sources=sources,
        offsets=offsets,
        weights=args.weights,
        conf=args.conf,
        imgsz=args.imgsz,
        detect_every=args.detect_every,
        min_green=args.min_green,
        emergency_hold=args.emergency_hold,
        switch_margin=args.switch_margin,
        device=args.device,
        log_csv=args.log,
    )
    for lane, start, dur in parse_simulate(args.simulate):
        engine.simulator.schedule(lane, start, dur)

    if not args.headless and not args.save:
        print("\nKeys:  Q = quit    1/2/3/4 = simulate an ambulance on N/S/E/W\n")

    engine.run(show=not args.headless, save_path=args.save, max_frames=args.frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
