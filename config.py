"""
config.py — single place for every knob in the project.

Change values here instead of hunting through the code.

Author: Devesh Tiwari (B.Tech AI, Amrita Vishwa Vidyapeetham, Amaravati)
"""

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent

MODELS_DIR = BASE_DIR / "models"
DEFAULT_WEIGHTS = BASE_DIR / "yolov8n.pt"          # COCO pretrained (fallback)
CUSTOM_WEIGHTS = MODELS_DIR / "best.pt"           # produced by train.py

VERTICAL_VIDEO = BASE_DIR / "Traffic_north_south.mp4"
HORIZONTAL_VIDEO = BASE_DIR / "Traffic_east_west.mp4"

# The repo ships 2 clips, so each one feeds two lanes (offset so they
# do not look like an obvious copy of each other).
DEFAULT_SOURCES = (VERTICAL_VIDEO, VERTICAL_VIDEO, HORIZONTAL_VIDEO, HORIZONTAL_VIDEO)
DEFAULT_OFFSETS = (0.00, 0.42, 0.15, 0.63)        # fraction of clip to start at

LANE_NAMES = ("NORTH", "SOUTH", "EAST", "WEST")

# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.45
IMG_SIZE = 640
DETECT_EVERY_N_FRAMES = 2       # run the network every Nth frame (CPU speed-up)

# COCO class ids (used when running the pretrained yolov8n.pt fallback)
COCO_VEHICLE_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# A class name containing any of these words is treated as an emergency
# vehicle. Works for custom datasets whose names are e.g. 'ambulance',
# 'Ambulance-108', 'police_car', 'fire_truck'.
PRIORITY_KEYWORDS = (
    "ambulance", "police", "patrol", "fire truck", "firetruck", "fire engine",
    "emergency", "priority", "siren", "rescue",
)
# Anything matching these is explicitly NOT an emergency vehicle even if a
# keyword above appears inside its name (e.g. COCO's "fire hydrant").
NON_PRIORITY_KEYWORDS = ("hydrant", "station", "extinguisher", "alarm", "exit")

# A class name containing any of these is counted as traffic
# (anything not matching and not priority is still counted if the model is
# a custom 4-class model — see detector.py).
VEHICLE_KEYWORDS = ("car", "motor", "bike", "bus", "truck", "auto", "van",
                    "taxi", "rickshaw", "vehicle")

# --------------------------------------------------------------------------- #
# Signal logic
# --------------------------------------------------------------------------- #
MIN_GREEN_SECONDS = 3.0         # a lane keeps green at least this long
SWITCH_MARGIN = 2               # a lane must beat the current one by this many
                                # vehicles before it takes over (stops flicker)
EMERGENCY_HOLD_SECONDS = 6.0    # green is held this long after the last sighting
EMERGENCY_MIN_SECONDS = 0.0     # emergency green never waits for min_green

# --------------------------------------------------------------------------- #
# Visuals
# --------------------------------------------------------------------------- #
TILE_SIZE = (480, 270)          # width, height of one lane tile (display)
INFER_SIZE = (640, 360)         # width, height fed to YOLO
HEADER_HEIGHT = 46
FPS_TARGET = 25                 # output video fps
EMERGENCY_BLINK_HZ = 2.5

COLOR_GREEN = (60, 220, 60)
COLOR_RED = (40, 40, 230)
COLOR_AMBER = (0, 165, 255)
COLOR_PRIORITY = (255, 60, 220)     # magenta boxes for ambulances / police
COLOR_VEHICLE = (235, 205, 40)      # cyan-ish boxes for normal traffic
COLOR_TEXT = (245, 245, 245)
COLOR_PANEL = (28, 28, 32)
COLOR_DIM = (150, 150, 155)


def add_styles() -> None:
    """Reserved for future theme overrides (kept for API symmetry)."""
    return None
