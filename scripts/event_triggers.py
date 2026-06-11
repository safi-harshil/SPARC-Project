#!/usr/bin/env python3
"""
event_triggers.py

Central place for trigger checks, flagging, and logging while capture/processing runs.
(RightWristSpeedTrigger uses 3 segment-local averages; ObjectUntouchedTrigger updated per spec.)
"""

from __future__ import annotations
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import random
import json

# from ros_publisher_node import ROS2PublisherNode

from tunables import (
    SPEED_TRIGGER_CADENCE_WINDOW_S,
    SPEED_TRIGGER_REFCSV_PATH,
    SPEED_TRIGGER_OVERLAY_TTL_S,  # still used by speed trigger
    OBJECT_TRIGGER_WINDOW_SEC,     # ← window length used for object trigger
)

from logger_utils import (
    DebouncedLogger,
    get_speed_trigger_logger,
    get_object_trigger_logger,  # logs to <cam>/logs/object_trigger
)

# --------------------------- Utilities ---------------------------

def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


# (speed trigger still uses preview overlays via annotate_event)
def _notify_preview_grid(preview_grid, cam_label: str, text: str, kind: str = "speed",
                         ttl_s: float = SPEED_TRIGGER_OVERLAY_TTL_S) -> None:
    if preview_grid is None:
        return
    try:
        if hasattr(preview_grid, "annotate_event"):
            preview_grid.annotate_event(cam_label, kind, text, ttl_s)
    except Exception:
        pass


# --------------------------- Reference CSV ---------------------------

@dataclass
class RefRow:
    time_s: float
    lo: float
    hi: float


class ReferenceBounds:
    def __init__(self, csv_path: Path):
        self.rows: List[RefRow] = []
        self._load(csv_path)

    def _load(self, csv_path: Path) -> None:
        if not csv_path or not Path(csv_path).exists():
            raise FileNotFoundError(f"[event_triggers] Reference CSV not found: {csv_path}")
        with open(csv_path, "r", newline="") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                t = float(r["time_s"])
                lo = float(r["lower_bound"])
                hi = float(r["upper_bound"])
                self.rows.append(RefRow(t, lo, hi))
        self.rows.sort(key=lambda r: r.time_s)
        if not self.rows:
            raise ValueError("[event_triggers] Reference CSV is empty.")

    def get_bounds(self, t: float) -> Tuple[float, float]:
        if t <= self.rows[0].time_s:
            return self.rows[0].lo, self.rows[0].hi
        if t >= self.rows[-1].time_s:
            return self.rows[-1].lo, self.rows[-1].hi
        last = self.rows[0]
        for row in self.rows[1:]:
            if row.time_s > t:
                return last.lo, last.hi
            last = row
        return self.rows[-1].lo, self.rows[-1].hi


# --------------------------- Trigger Base ---------------------------

class BaseTrigger:
    def __init__(self, name: str):
        self.name = name

    def reset(self):
        pass


# ---------------------- Right Wrist Speed Trigger (UPDATED) -------------------

class RightWristSpeedTrigger(BaseTrigger):
    """
    Right wrist speed trigger with 3 equal time segments over the total task duration.

    - total_duration_s (seconds) is optionally passed at construction.
    - T_total is divided into 3 equal segments:
        [0, T_total/3), [T_total/3, 2*T_total/3), [2*T_total/3, T_total]
      Anything beyond T_total is treated as segment 2.
    - For each segment, we compute a *segment-local* average speed:

          segment_movement = cumulative_movement - segment_base_movement
          segment_elapsed  = elapsed_time_s - segment_start_time_s
          avg_speed        = segment_movement / segment_elapsed

      where segment_base_movement and segment_start_time_s are reset when we
      enter a new segment. This ensures movement is not carried forward from any
      previous segment into the next one.

    - The reference CSV lookup still uses the *global* elapsed time:

          lo, hi = ref.get_bounds(elapsed_time_s)

      so we compare the segment-local average speed against bounds defined as a
      function of global elapsed time.

    - Additionally, we compute a global average speed:
          global_avg_speed = cumulative_movement / elapsed_time_s

      and log both speeds.
    """

    def __init__(self,
                 movement_cam_dir: Path,
                 cam_label: str,
                 reference_csv: Optional[Path] = None,
                 cadence_window_s: float = SPEED_TRIGGER_CADENCE_WINDOW_S,
                 overlay_ttl_s: float = SPEED_TRIGGER_OVERLAY_TTL_S,
                 logger: Optional[DebouncedLogger] = None,
                 total_duration_s: Optional[float] = None):
        super().__init__("right_wrist_speed")
        self.cam_label = cam_label
        self.cadence_window_s = max(0.1, float(cadence_window_s))
        self.overlay_ttl_s = float(overlay_ttl_s)
        self.movement_cam_dir = Path(movement_cam_dir)

        ref_path = Path(reference_csv) if reference_csv else Path(SPEED_TRIGGER_REFCSV_PATH)
        self.ref = ReferenceBounds(ref_path)

        self.logger: DebouncedLogger = logger if logger is not None else get_speed_trigger_logger(self.movement_cam_dir)
        self._last_bad_slot_logged: Optional[int] = None

        # 3-segment support
        self.total_duration_s: Optional[float] = (
            float(total_duration_s) if total_duration_s is not None and total_duration_s > 0 else None
        )
        self.segment_len_s: Optional[float] = (
            self.total_duration_s / 3.0 if self.total_duration_s is not None else None
        )

        # segment-local state
        self._cur_segment_index: Optional[int] = None
        self._segment_start_time_s: float = 0.0
        self._segment_base_movement: float = 0.0

    def reset(self):
        self._last_bad_slot_logged = None
        # reset segment state as well
        self._cur_segment_index = None
        self._segment_start_time_s = 0.0
        self._segment_base_movement = 0.0

    def _slot_index(self, elapsed_s: float) -> int:
        return int(math.floor(elapsed_s / self.cadence_window_s))

    def _segment_index(self, elapsed_s: float) -> int:
        """
        Map global elapsed_s to segment index:
          0 -> [0, T/3)
          1 -> [T/3, 2T/3)
          2 -> [2T/3, T] and anything beyond T
        """
        if self.segment_len_s is None or elapsed_s < 0.0:
            return 0  # fall back to segment 0 if no total_duration_s
        if self.total_duration_s is not None and elapsed_s >= self.total_duration_s:
            return 2
        idx = int(elapsed_s // self.segment_len_s)
        if idx < 0:
            idx = 0
        if idx > 2:
            idx = 2
        return idx

    def _judge(self, value: float, lo: float, hi: float) -> Tuple[bool, str]:
        if value < lo:
            return False, "low"
        if value > hi:
            return False, "high"
        return True, "good"

    def _format_line(self, *, ts_s: float, frame_idx: int, slot: int,
                     status: str,
                     seg_value: float,
                     global_value: float,
                     lo: float, hi: float,
                     segment_index: int,
                     segment_elapsed: float) -> str:
        return (
            f"{_iso_now()} ts={ts_s:.3f} frame={frame_idx} slot={slot} "
            f"seg={segment_index} seg_elapsed={segment_elapsed:.3f} "
            f"status={status.upper()} seg_speed={seg_value:.6f} "
            f"global_speed={global_value:.6f} lo={lo:.6f} hi={hi:.6f} "
            f"cam={self.cam_label}"
        )

    def update(self,
               *,
               elapsed_time_s: float,
               frame_idx: int,
               cumulative_movement: float,
               preview_grid=None) -> dict:
        if elapsed_time_s <= 0:
            return {
                "good": True,
                "reason": "good",
                "value": 0.0,              # segment-local (degenerate)
                "global_value": 0.0,
                "lower": float("nan"),
                "upper": float("nan"),
                "slot": self._slot_index(0.0),
                "elapsed_s": 0.0,
                "segment_index": 0,
                "segment_elapsed_s": 0.0,
                "segment_movement": 0.0,
            }

        # Global average speed (always defined if elapsed_time_s > 0)
        global_avg_speed = float(cumulative_movement) / float(elapsed_time_s)

        # --- segment-local averaging ---
        if self.segment_len_s is not None:
            seg_idx = self._segment_index(elapsed_time_s)

            # first time we are called, or segment change
            if self._cur_segment_index is None or seg_idx != self._cur_segment_index:
                self._cur_segment_index = seg_idx
                self._segment_start_time_s = float(elapsed_time_s)
                self._segment_base_movement = float(cumulative_movement)

            segment_elapsed = float(elapsed_time_s - self._segment_start_time_s)
            if segment_elapsed <= 0.0:
                segment_elapsed = 0.0
                segment_movement = 0.0
                seg_avg_speed = 0.0
            else:
                segment_movement = float(cumulative_movement - self._segment_base_movement)
                seg_avg_speed = segment_movement / segment_elapsed
        else:
            # Fallback: original global-average behavior if no total_duration_s
            seg_idx = 0
            segment_elapsed = float(elapsed_time_s)
            segment_movement = float(cumulative_movement)
            seg_avg_speed = (
                segment_movement / segment_elapsed if segment_elapsed > 0.0 else 0.0
            )

        # Reference CSV lookup still uses *global* elapsed_time_s
        lo, hi = self.ref.get_bounds(elapsed_time_s)
        # Judge based on *segment-local* speed
        good, reason = self._judge(seg_avg_speed, lo, hi)
        slot = self._slot_index(elapsed_time_s)

        if not good and self._last_bad_slot_logged != slot:
            line = self._format_line(
                ts_s=elapsed_time_s,
                frame_idx=frame_idx,
                slot=slot,
                status=("LOW" if reason == "low" else "HIGH"),
                seg_value=seg_avg_speed,
                global_value=global_avg_speed,
                lo=lo,
                hi=hi,
                segment_index=seg_idx,
                segment_elapsed=segment_elapsed,
            )
            try:
                self.logger.info(line)
                self.logger.periodic_flush()
            except Exception:
                pass
            self._last_bad_slot_logged = slot
            _notify_preview_grid(
                preview_grid,
                self.cam_label,
                "Low Speed" if reason == "low" else "High Speed",
                kind="speed",
            )

        # node = ROS2PublisherNode.get_instance()
        # node.handspeed_piece_data = reason

        return {
            "good": good,
            "reason": reason,
            "value": seg_avg_speed,          # segment-local speed
            "global_value": global_avg_speed,
            "lower": lo,
            "upper": hi,
            "slot": slot,
            "elapsed_s": elapsed_time_s,
            "segment_index": seg_idx,
            "segment_elapsed_s": segment_elapsed,
            "segment_movement": segment_movement,
        }


# ---------------------- Object Untouched Trigger (UPDATED) -------------------

class ObjectUntouchedTrigger(BaseTrigger):
    """
    At every frame (each update call), evaluate the last W = OBJECT_TRIGGER_WINDOW_SEC seconds
    (based on provided fps). Count objects untouched ≥ 90% of that window:

        < 2  -> "Bad:Less"
        > 4  -> "Bad:More"
        else -> "Good"

    Log format (only line, no extras):
        "[start_frame: end_frame) = <Good|Bad:Less|Bad:More>"

    Logging starts only after a full window of frames is available.
    """

    def __init__(self,
                 cam_dir: Path,
                 cam_label: Optional[str] = None,                 # ← optional for compatibility
                 cadence_window_s: float = OBJECT_TRIGGER_WINDOW_SEC,
                 logger: Optional[DebouncedLogger] = None):
        super().__init__("objects_untouched")
        self.cam_label = cam_label or "(unknown)"
        self.cam_dir = Path(cam_dir)
        self.W = float(max(0.5, cadence_window_s))
        self.tol_thresh = 0.9 * self.W
        self.logger: DebouncedLogger = logger if logger is not None else get_object_trigger_logger(self.cam_dir)
        self._last_slot_logged: Optional[int] = None  # kept for compatibility with prior versions

    def reset(self):
        self._last_slot_logged = None

    @staticmethod
    def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
        # inclusive frame intervals [a0,a1], [b0,b1]
        lo = max(a0, b0)
        hi = min(a1, b1)
        return max(0, hi - lo + 1)

    @staticmethod
    def _contains_frame(spans: List[List[int]], f: int) -> bool:
        for s, e in spans:
            if s <= f <= e:
                return True
        return False

    def update(self,
               *,
               elapsed_time_s: float,
               frame_idx: int,
               fps: float,
               untouched_out: Dict[str, List[List[int]]],
               preview_grid=None,
               cam_label: Optional[str] = None) -> dict:
        # Adopt runtime camera label if provided (keeps logs accurate)
        if cam_label:
            self.cam_label = cam_label

        # minimal robustness: guard fps
        efps = float(fps)
        if not (efps > 0.0 and math.isfinite(efps)):
            efps = 30.0  # safe default

        # --- sliding window in frames based on fps ---
        frames_per_window = max(1, int(round(self.W * efps)))
        end_excl = int(frame_idx) + 1           # half-open window end
        start_f  = end_excl - frames_per_window # inclusive start for full window
        if start_f < 0:
            # not enough history yet → do not log
            # build a minimal return payload and exit
            return {
                "qualified_count": 0,
                "per_object_sec": {},
                "per_object_pct": {},
                "per_object_state": {},
                "slot": int(math.floor(elapsed_time_s / self.W)),
                "window_start_s": 0.0,
                "window_end_s": end_excl / efps,
            }

        start_incl = start_f
        end_incl   = end_excl - 1
        win_len_f  = frames_per_window
        thresh_f   = int(math.ceil(0.9 * win_len_f))  # ≥ 90%

        # Evaluate untouched coverage in this frame window
        per_obj_sec: Dict[str, float] = {}
        per_obj_pct: Dict[str, float] = {}
        per_obj_state: Dict[str, str] = {}
        count_qualified = 0
        untouched_objects = []

        for obj, spans in untouched_out.items():
            total_f = 0
            for s, e in spans:
                total_f += self._overlap_len(s, e, start_incl, end_incl)
            if total_f >= thresh_f:
                count_qualified += 1
                untouched_objects.append(obj)

            # metrics for return payload (seconds & percentage of W)
            sec = total_f / efps
            per_obj_sec[obj] = sec
            per_obj_pct[obj] = (sec / self.W) * 100.0 if self.W > 0 else 0.0
            per_obj_state[obj] = "untouched" if self._contains_frame(spans, end_incl) else "other"

        object_list = {"red": True, "green": True, "gray": True, "yellow_1": True, "yellow_2": True, "gold": True}
        object_list_ros = {"red": True, "green": True, "grey": True, "yellow": True, "small yellow": True, "brown": True}
        for obj in object_list.keys():
            if obj not in untouched_objects:
                object_list[obj] = False

        for obj in object_list_ros.keys():
            if obj not in untouched_objects:
                object_list_ros[obj] = False

        # --- Log exactly one simple line per frame (no extras) ---
        line = f"[{start_incl}: {end_excl}] = {object_list}"
        try:
            self.logger.info(line)
            self.logger.periodic_flush()
        except Exception:
            pass

        # label1 = json.dumps(object_list_ros)   # "{object} : {True|False}"
        # node = ROS2PublisherNode.get_instance()
        # node.untouched_piece_data = label1

        # Return payload (kept for compatibility)
        return {
            "qualified_count": count_qualified,
            "per_object_sec": per_obj_sec,
            "per_object_pct": per_obj_pct,
            "per_object_state": per_obj_state,
            "slot": int(math.floor(elapsed_time_s / self.W)),
            "window_start_s": start_incl / efps,
            "window_end_s": end_excl / efps,
        }


# --------------------------- Manager (future) ---------------------------

class EventManager:
    def __init__(self):
        self.triggers = []

    def add(self, trig: BaseTrigger):
        self.triggers.append(trig)

    def reset_all(self):
        for t in self.triggers:
            t.reset()
