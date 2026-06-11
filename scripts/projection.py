#!/usr/bin/env python3
from typing import Optional, Tuple
import numpy as np

def project_xy_to_xyz(px: float, py: float, depth_img: np.ndarray,
                      fx: float, fy: float, cx: float, cy: float, depth_scale_m: float) -> Optional[Tuple[float,float,float]]:
    h, w = depth_img.shape[:2]
    x = int(round(px)); y = int(round(py))
    if not (0 <= x < w and 0 <= y < h):
        return None
    raw = int(depth_img[y, x])
    if raw <= 0:
        return None
    z_m = raw * depth_scale_m
    z_mm = z_m * 1000.0
    X_mm = (x - cx) * z_mm / fx
    Y_mm = (y - cy) * z_mm / fy
    return (round(X_mm, 2), round(Y_mm, 2), round(z_mm, 2))
