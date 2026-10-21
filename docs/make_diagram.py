#!/usr/bin/env python3
"""make_diagram.py — draws docs/arch.png (system architecture) and docs/decision_flow.png."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent
C_INPUT = "#e8f0fe"
C_PROC = "#1f6feb"
C_DECIDE = "#ff3cdc"
C_OUT = "#e6f7e6"
EDGE = "#33415c"


def box(ax, x, y, w, h, text, fc="#ffffff", ec=EDGE, tc="#111111", fs=9, bold=False, r=0.02):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.006,rounding_size={r}",
                       linewidth=1.3, facecolor=fc, edgecolor=ec)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p1, p2, color=EDGE, style="-|>", lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=11,
                                 color=color, linewidth=lw, linestyle=ls,
                                 shrinkA=2, shrinkB=2))


def architecture():
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=200)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.975, "Smart Intersection AI — system architecture",
            ha="center", fontsize=13, fontweight="bold", color="#0d1b2a")

    # stage labels
    for x, label in [(0.075, "1. LOOK"), (0.30, "2. FIND"), (0.545, "3. DECIDE"), (0.80, "4. SHOW")]:
        ax.text(x, 0.885, label, ha="center", fontsize=9, color=C_PROC, fontweight="bold")

    # inputs
    for i, lane in enumerate(["NORTH", "SOUTH", "EAST", "WEST"]):
        box(ax, 0.015, 0.70 - i * 0.155, 0.12, 0.115,
            f"camera / video\n{lane}", fc=C_INPUT, fs=8)

    # detector
    box(ax, 0.215, 0.55, 0.17, 0.24,
        "detector.py\nYOLOv8\n\ncounts +\npriority list", fc=C_PROC, tc="#ffffff",
        fs=8.5, bold=True)

    # controller
    box(ax, 0.45, 0.55, 0.20, 0.24,
        "controller.py\n\n emergency ?\n hold ?\n density + hysteresis",
        fc=C_DECIDE, tc="#ffffff", fs=8.5, bold=True)

    # outputs
    box(ax, 0.71, 0.72, 0.26, 0.12, "2x2 dashboard (OpenCV)\nboxes · lights · HUD", fc=C_OUT, fs=8.5)
    box(ax, 0.71, 0.55, 0.26, 0.12, "browser dashboard\nFlask + MJPEG", fc=C_OUT, fs=8.5)
    box(ax, 0.71, 0.38, 0.26, 0.12, "decision log\nCSV per frame", fc=C_OUT, fs=8.5)

    # arrows in
    for i in range(4):
        arrow(ax, (0.135, 0.7575 - i * 0.155), (0.215, 0.67 - i * 0.01))
    arrow(ax, (0.385, 0.67), (0.45, 0.67), lw=1.8)
    for y in (0.78, 0.61, 0.44):
        arrow(ax, (0.65, 0.67), (0.71, y), lw=1.4)

    # config + simulator
    box(ax, 0.215, 0.33, 0.17, 0.11, "config.py\nthresholds · timings", fc="#f4f4f6", fs=8)
    box(ax, 0.45, 0.33, 0.20, 0.11, "EmergencySimulator\n--simulate / hotkeys 1-4",
        fc="#fff0fb", fs=8)
    arrow(ax, (0.30, 0.44), (0.30, 0.55), ls=":", lw=1.1)
    arrow(ax, (0.55, 0.44), (0.55, 0.55), ls=":", lw=1.1)

    # training row
    ax.add_patch(FancyBboxPatch((0.015, 0.05), 0.97, 0.20,
                                boxstyle="round,pad=0.006,rounding_size=0.02",
                                linewidth=1.1, facecolor="#fbfbfd", edgecolor="#c8ced8",
                                linestyle="--"))
    ax.text(0.03, 0.205, "training path (offline, needs GPU time)",
            fontsize=8.5, color="#5a6474", fontweight="bold")
    for i, (x, label) in enumerate([
        (0.03, "prepare_dataset.py\nframes"), (0.225, "prelabel\n(COCO auto-labels cars)"),
        (0.42, "human: draw\nambulance + police"), (0.615, "prepare_dataset.py\nsplit + data.yaml"),
        (0.80, "train.py / Colab\n-> models/best.pt"),
    ]):
        box(ax, x, 0.085, 0.165, 0.10, label, fc="#ffffff", fs=7.5)
        if i < 4:
            arrow(ax, (x + 0.165, 0.135), (x + 0.195, 0.135), lw=1.0)
    ax.annotate("", xy=(0.215, 0.615), xytext=(0.045, 0.20),
                arrowprops=dict(arrowstyle="-|>", color=C_DECIDE, lw=1.2, ls="--",
                                connectionstyle="arc3,rad=-0.18", mutation_scale=11))
    ax.text(0.012, 0.30, "models/best.pt\nauto-detected", fontsize=7.2, color=C_DECIDE,
            rotation=90, va="center", ha="left")

    fig.tight_layout()
    fig.savefig(OUT / "arch.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / "arch.png")


def decision_flow():
    fig, ax = plt.subplots(figsize=(9.5, 3.4), dpi=200)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.95, "Decision logic (controller.decide)", ha="center",
            fontsize=12, fontweight="bold", color="#0d1b2a")

    box(ax, 0.02, 0.55, 0.17, 0.22, "priority vehicle\nin any lane?", fc="#fff0fb", fs=8.5)
    box(ax, 0.28, 0.62, 0.19, 0.17, "that lane GREEN\nall others RED", fc=C_DECIDE,
        tc="#ffffff", fs=8.5, bold=True)
    box(ax, 0.28, 0.30, 0.19, 0.17, "hold it green\n(convoy clears)", fc="#ffe9c7", fs=8.5)
    box(ax, 0.55, 0.55, 0.19, 0.22, "busiest lane\nleads by margin?", fc="#e8f0fe", fs=8.5)
    box(ax, 0.80, 0.62, 0.18, 0.17, "busiest lane\nGREEN", fc=C_PROC, tc="#ffffff", fs=8.5, bold=True)
    box(ax, 0.80, 0.30, 0.18, 0.17, "keep current\ngreen", fc="#e6f7e6", fs=8.5)

    arrow(ax, (0.19, 0.72), (0.28, 0.70))
    ax.text(0.235, 0.775, "yes", fontsize=7.5, color="#c2185b")
    arrow(ax, (0.19, 0.60), (0.28, 0.40), ls="--")
    ax.text(0.20, 0.47, "no\n(until hold ends)", fontsize=7.5, color="#5a6474")
    arrow(ax, (0.375, 0.62), (0.375, 0.47), lw=1.1, color="#d38b00")
    ax.text(0.39, 0.545, "after 6 s", fontsize=7.5, color="#d38b00")
    arrow(ax, (0.47, 0.70), (0.55, 0.70))
    arrow(ax, (0.74, 0.70), (0.80, 0.70))
    ax.text(0.775, 0.745, "yes", fontsize=7.5, color="#1a7f37")
    arrow(ax, (0.645, 0.55), (0.845, 0.47))
    ax.text(0.70, 0.495, "no", fontsize=7.5, color="#1a7f37")
    ax.text(0.02, 0.13,
            "minimum green (3 s) and switch margin (2 vehicles) are checked before any switch —\n"
            "this is what stops the light flickering when detection counts wobble by one vehicle.",
            fontsize=7.8, color="#5a6474", linespacing=1.5)

    fig.tight_layout()
    fig.savefig(OUT / "decision_flow.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / "decision_flow.png")


if __name__ == "__main__":
    architecture()
    decision_flow()
