#!/usr/bin/env python3
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import numpy as np

# ───────────────────────── Core RealSense Frame ─────────────────────────
@dataclass
class FramePacket:
    cam_label: str
    frame_id: int
    t_ns: int
    rs_ts_ms: float
    color: np.ndarray            # HxWx3 BGR
    depth: np.ndarray            # HxW uint16 (aligned to color)
    fx: float; fy: float; cx: float; cy: float
    depth_scale_m: float         # meters per unit


# ───────────────────────── Object State Containers ─────────────────────────
@dataclass
class ObjectStates:
    # for current frame, per object:
    untouched: Dict[str, bool]
    checking:  Dict[str, bool]
    frame_idx: int


@dataclass
class ObjectOutputs:
    # append-only buffers for CSV & intervals
    xy_row: Dict[str, str]
    z_row: Dict[str, str]
    # running intervals (closed spans)
    untouched_out: Dict[str, List[List[int]]]
    checking_out: Dict[str, List[List[int]]]


# ───────────────────────── New: Preview Overlay Exchange ─────────────────────────
@dataclass
class ObjectOverlayPacket:
    """
    Packet forwarded to PreviewGrid.update_objects()
    from object_worker or ObjectInteraction.ingest_frame().
    Contains optional full-frame binary masks and crop info.
    """
    cam_label: str
    frame_idx: int
    untouched: Dict[str, bool]
    checking: Dict[str, bool]
    masks: Optional[Dict[str, np.ndarray]] = None     # {obj_name: HxW uint8 mask}
    crop_box: Optional[Tuple[int, int, int, int]] = None  # (x_min, y_min, x_max, y_max)
