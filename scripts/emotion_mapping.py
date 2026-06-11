#!/usr/bin/env python3
from typing import Tuple
import numpy as np

MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
MOUTH_UP     = 13
MOUTH_DOWN   = 14
BROW_LEFT_UP = 105
BROW_RIGHT_UP= 334
NOSE_BRIDGE  = 6

def _safe_norm(a, b):
    ax, ay = a
    bx, by = b
    return float(np.hypot(ax - bx, ay - by))

def normalize_feature(v, lo, hi):
    if hi <= lo:
        return 0.0
    v = min(max(v, lo), hi)
    return (v - lo) / (hi - lo)

def map_to_valence_arousal(landmarks_px: np.ndarray) -> Tuple[float, float]:
    mouth_left  = landmarks_px[MOUTH_LEFT]
    mouth_right = landmarks_px[MOUTH_RIGHT]
    mouth_up    = landmarks_px[MOUTH_UP]
    mouth_down  = landmarks_px[MOUTH_DOWN]
    nose        = landmarks_px[NOSE_BRIDGE]
    brow_l      = landmarks_px[BROW_LEFT_UP]
    brow_r      = landmarks_px[BROW_RIGHT_UP]

    mouth_width = _safe_norm(mouth_left, mouth_right) + 1e-6
    mouth_open  = _safe_norm(mouth_up, mouth_down) / mouth_width
    brow_raise  = ((_safe_norm(brow_l, nose) + _safe_norm(brow_r, nose)) / 2.0) / mouth_width

    mo_norm = normalize_feature(mouth_open,  0.05, 0.6)
    br_norm = normalize_feature(brow_raise,  0.9,  2.0)

    val = max(-1.0, min(1.0, (mo_norm * 1.25) - 0.25))
    aro = max(0.0, min(1.0, 0.6 * br_norm + 0.4 * mo_norm))
    return (float(val), float(aro))
