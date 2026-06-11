#!/usr/bin/env python3

import os
from pathlib import Path

# ---------- Paths ----------
CAMERA_SERIALS_FILE = os.path.expanduser("~/Desktop/SPARC-Project/camera_serials.txt")

# ---------- Hand landmarks & processing ----------
WRIST_ID = 0
NUM_LANDMARKS = 21
WRIST_SEP_PX = 30
KEYPOINTS_MOV = [0, 1, 2, 3, 4]  # wrist + thumb ids
HANDS = ("L", "R")

# Anchor reset if both-hands reference is lost for this many processed frames
ANCHOR_MISS_RESET = 300

# ---------- Colors (BGR) ----------
COLOR_LEFT  = (0, 255, 0)
COLOR_RIGHT = (0, 0, 255)
COLOR_TEXT  = (255, 255, 255)

# ---------- Chart & CSV cadences ----------
PLOT_UPDATE_INTERVAL = 7        # frames
CUM_CSV_INTERVAL_SEC = 10.0     # seconds

# ---------- Depth scale ----------
FALLBACK_DEPTH_SCALE = 0.0010000000474974513  # meters per unit

# ---------- Misc ----------
DEFAULT_CELL_W = 640
DEFAULT_CELL_H = 360

# ---------- Audio device whitelist ----------
VALID_MIC_IDS = [
    'hw:2,0',   # card index, device index
    'hw:3,0',

    # RODE Wireless GO II Receiver (single USB device)
    # NOTE: receiver provides 2-channel input (TX1 + TX2) on the SAME device
    # Using CARD-based address keeps it stable even if card index changes.
    'hw:CARD=RX,DEV=0',
]

# (Optional) channel map per device for arecord (-c)
# If a device is not listed here, audio_worker defaults to 1 channel.
MIC_CHANNELS = {
    'hw:CARD=RX,DEV=0': 2,   # RODE GO II RX → 2 channels (TX1 + TX2)
}

# ================== Event trigger tunables (NEW) =======================================
# Throttle logs/overlays to once per this many seconds (per slot)
SPEED_TRIGGER_CADENCE_WINDOW_S = 30.0  # adjust as you like

# Default path to the expected reference CSV (can be overridden per trigger)
# SPEED_TRIGGER_REFCSV_PATH = "~/Desktop/SPARC-Project/right_wrist_speed_bounds.csv"
SPEED_TRIGGER_REFCSV_PATH = "/home/robotics/Desktop/SPARC-Project/right_wrist_speed_bounds.csv"

# How long the "High/Low Speed" label stays visible on the PreviewGrid
SPEED_TRIGGER_OVERLAY_TTL_S = 6.0  # 5–7 seconds as requested

# ========================================================================================

# ─── Object interaction knobs ───────────────────────────────
OBJ_REF_FRAME         = 100
OBJ_P_START           = 30
OBJ_Q_END             = 60
OBJ_DEPTH_UNITS       = 0.0010000000474974513
# IMPORTANT: keep "yellow" (singular) so tracking logic can split to yellow_1/_2 internally
OBJ_COLORS = ["red", "green", "gray", "yellow", "gold"]  # ✅ changed from explicit _1/_2

# ─── Object trigger thresholds ──────────────────────────────
OBJECT_TRIGGER_WINDOW_SEC = 120.0
OBJECT_TRIGGER_MIN_PCT = 90.0
