#!/usr/bin/env python3
# movement_processor.py — cumulative calc + preview + event logging

import time
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import deque  # (kept as-is; safe even if unused)

import cv2
import numpy as np
import mediapipe as mp

from control_flags import pause_event, stop_event
from tunables import (
    NUM_LANDMARKS, HANDS, KEYPOINTS_MOV, ANCHOR_MISS_RESET,
    PLOT_UPDATE_INTERVAL, CUM_CSV_INTERVAL_SEC
)
# centralized movement logger (per-cam)
from logger_utils import get_movement_logger, DebouncedLogger
from types_shared import FramePacket
from hand_utils import (
    get_mediapipe_detections,
    collapse_overlap_mediapipe,
    collapse_overlap_raw,
    label_by_proximity,
    draw_annotations,
)
from projection import project_xy_to_xyz
from preview_grid import PreviewGrid

# NEW: import the centralized trigger
from event_triggers import RightWristSpeedTrigger

# ---------- CSV streaming (landmarks) ----------
class CSVStream:
    def __init__(self, csv_path: Path, flush_every: int = 30):
        self.csv_path = Path(csv_path)
        self.flush_every = max(1, int(flush_every))
        self._fh = None
        self._lock = threading.Lock()
        self._count = 0
        self._header_written = False

    def _ensure_open(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self._fh is None:
            self._fh = open(self.csv_path, "a", buffering=1)

    def write_header_if_needed(self):
        with self._lock:
            if not self._header_written:
                cols = self.build_columns()
                self._ensure_open()
                self._fh.write(",".join(cols) + "\n")
                self._header_written = True

    @staticmethod
    def build_columns():
        cols = ["frame_id", "t_ns"]
        for lid in range(NUM_LANDMARKS):
            cols += [f"x_{lid}_L_px", f"y_{lid}_L_px"]
        for lid in range(NUM_LANDMARKS):
            cols += [f"x_{lid}_R_px", f"y_{lid}_R_px"]
        for h in HANDS:
            for lid in KEYPOINTS_MOV:
                cols += [f"{h}_{lid}_X_mm", f"{h}_{lid}_Y_mm", f"{h}_{lid}_Z_mm"]
        for h in HANDS:
            for lid in KEYPOINTS_MOV:
                cols += [f"{h}_{lid}_move_mm"]
        return cols

    def append_row(self, row_vals: List[str]):
        with self._lock:
            self._ensure_open()
            self._fh.write(",".join(map(str, row_vals)) + "\n")
            self._count += 1
            if self._count % self.flush_every == 0:
                self._fh.flush()

    def close(self):
        with self._lock:
            if self._fh:
                try:
                    self._fh.flush()
                    self._fh.close()
                finally:
                    self._fh = None

# ---------- Extra CSVs ----------
class CumMovementWideCSV:
    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self._lock = threading.Lock()

    def write_snapshot(self, cam_label: str, bins: Dict[int, Dict[str, float]]):
        with self._lock:
            pmax = max(bins.keys()) if bins else 0
            cols = ["cam"]
            for tag in ("L0", "L4", "R0", "R4"):
                cols += [f"{tag}_p{p}" for p in range(1, pmax + 1)]
            row = [cam_label]
            for tag in ("L0", "L4", "R0", "R4"):
                for p in range(1, pmax + 1):
                    v = bins.get(p, {}).get(tag, 0.0)
                    row.append(f"{v:.6f}")
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w") as f:
                f.write(",".join(cols) + "\n")
                f.write(",".join(map(str, row)) + "\n")

class MovementStatsFinalCSV:
    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)

    def write_final(self, cam_label: str, elapsed_sec: float,
                    L0_tot: float, L4_tot: float, R0_tot: float, R4_tot: float):
        L0_avg = (L0_tot / elapsed_sec) if elapsed_sec > 0 else 0.0
        L4_avg = (L4_tot / elapsed_sec) if elapsed_sec > 0 else 0.0
        R0_avg = (R0_tot / elapsed_sec) if elapsed_sec > 0 else 0.0
        R4_avg = (R4_tot / elapsed_sec) if elapsed_sec > 0 else 0.0
        cols = ["cam","elapsed_sec",
                "L0_total","L0_avg_speed","L4_total","L4_avg_speed",
                "R0_total","R0_avg_speed","R4_total","R4_avg_speed"]
        row = [cam_label, f"{elapsed_sec:.6f}",
               f"{L0_tot:.6f}", f"{L0_avg:.6f}", f"{L4_tot:.6f}", f"{L4_avg:.6f}",
               f"{R0_tot:.6f}", f"{R0_avg:.6f}", f"{R4_tot:.6f}", f"{R4_avg:.6f}"]
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w") as f:
            f.write(",".join(cols) + "\n")
            f.write(",".join(map(str, row)) + "\n")

# ---------- Worker ----------
def processor_worker(
    cam_label: str,
    q: "queue.Queue[FramePacket]",
    out_dir: Path,
    force_flip: str,
    stride: int,
    csv_flush_every: int,
    log_flush_sec: int,
    preview: Optional[PreviewGrid],
    viz_save_every: int = 0,
    total_duration_sec: float = 0.0,   # 👈 NEW: T_total passed from realtime_capture
    *,
    event_checker_enabled: bool = True,
):
    cam_dir = out_dir / cam_label
    (cam_dir / "CSV").mkdir(parents=True, exist_ok=True)
    (cam_dir / "logs").mkdir(parents=True, exist_ok=True)
    mp_dir = (cam_dir / "color_mp")
    mp_dir.mkdir(parents=True, exist_ok=True)

    logger = get_movement_logger(cam_dir, flush_sec=log_flush_sec)
    csv_stream = CSVStream(cam_dir / "CSV" / "hand_landmark_rt.csv", flush_every=csv_flush_every)
    csv_stream.write_header_if_needed()

    cum_csv = CumMovementWideCSV(cam_dir / "CSV" / "cumulative_movement_rt.csv")
    stats_final_csv = MovementStatsFinalCSV(cam_dir / "CSV" / "movement_stats_rt.csv")

    last_xyz: Dict[str, Optional[Tuple[float, float, float]]] = {
        f"{h}_{k}": None for h in HANDS for k in KEYPOINTS_MOV
    }

    prev_two_left_wrist: Optional[Tuple[float, float]] = None
    prev_two_right_wrist: Optional[Tuple[float, float]] = None
    have_anchor = False
    consecutive_no_two = 0

    cum = {"L_0": 0.0, "L_4": 0.0, "R_0": 0.0, "R_4": 0.0}

    first_t_ns: Optional[int] = None
    last_t_ns: Optional[int] = None

    bins: Dict[int, Dict[str, float]] = {}
    last_bin_written = 0
    plot_tick = 0

    # NEW: instantiate centralized right-wrist speed trigger (logs to cam_dir/logs)
    # Pass total_duration_sec so it can build the 3 equal time segments.
    speed_trigger = RightWristSpeedTrigger(
        movement_cam_dir=cam_dir,
        cam_label=cam_label,
        total_duration_s=float(total_duration_sec) if total_duration_sec > 0 else None,
    )

    try:
        with mp.solutions.hands.Hands(
            static_image_mode=False,
            model_complexity=1,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands_model:

            processed = 0
            while True:
                if stop_event.is_set():
                    break

                try:
                    pkt: FramePacket = q.get(timeout=0.1)
                except queue.Empty:
                    logger.periodic_flush()
                    if getattr(q, "closed", False) or stop_event.is_set():
                        break
                    continue

                while pause_event.is_set() and not stop_event.is_set():
                    time.sleep(0.05)
                if stop_event.is_set():
                    break

                processed += 1
                if stride > 1 and (processed - 1) % stride != 0:
                    continue

                if first_t_ns is None:
                    first_t_ns = pkt.t_ns
                last_t_ns = pkt.t_ns

                img_bgr = pkt.color
                if img_bgr is None or img_bgr.size == 0:
                    logger.warn("Empty color frame in movement processor; skipping.")
                    logger.periodic_flush()
                    continue

                h, w = img_bgr.shape[:2]
                frame_id = pkt.frame_id

                # --- detect hands ---
                results = hands_model.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
                dets = get_mediapipe_detections(results, w, h)
                dets = collapse_overlap_mediapipe(dets, frame_id, logger)

                L_lab = None
                R_lab = None
                raw_pts = []
                for d in dets[:2]:
                    raw_pts.append(d["pts"])
                    if d["label"] == "L":
                        L_lab = d["pts"]
                    elif d["label"] == "R":
                        R_lab = d["pts"]

                if force_flip == "flip":
                    L_lab, R_lab = R_lab, L_lab

                if have_anchor:
                    raw_pts = collapse_overlap_raw(raw_pts, prev_two_left_wrist, prev_two_right_wrist, logger, frame_id)
                    labeled = label_by_proximity(raw_pts, prev_two_left_wrist, prev_two_right_wrist)
                    L = labeled["L"]
                    R = labeled["R"]
                    if L is not None and R is not None:
                        prev_two_left_wrist = tuple(L[0])
                        prev_two_right_wrist = tuple(R[0])
                        consecutive_no_two = 0
                    else:
                        consecutive_no_two += 1
                        if consecutive_no_two > ANCHOR_MISS_RESET:
                            logger.warn(f"Lost two-hand reference for {consecutive_no_two} frames → resetting anchor.")
                            have_anchor = False
                            prev_two_left_wrist = prev_two_right_wrist = None
                            consecutive_no_two = 0
                else:
                    L, R = L_lab, R_lab
                    if L is not None and R is not None:
                        prev_two_left_wrist = tuple(L[0])
                        prev_two_right_wrist = tuple(R[0])
                        have_anchor = True
                        consecutive_no_two = 0
                        logger.info(f"Anchor set at frame {frame_id} (two-hand reference acquired).")

                # --- build CSV row + compute per-frame movement in mm ---
                row: List[float] = [frame_id, pkt.t_ns]

                # raw 2D landmarks (px)
                for lid in range(NUM_LANDMARKS):
                    if L is not None:
                        row += [float(L[lid, 0]), float(L[lid, 1])]
                    else:
                        row += ["", ""]
                for lid in range(NUM_LANDMARKS):
                    if R is not None:
                        row += [float(R[lid, 0]), float(R[lid, 1])]
                    else:
                        row += ["", ""]

                moves: Dict[str, float] = {}
                for hand_label, pts in (("L", L), ("R", R)):
                    for lid in KEYPOINTS_MOV:
                        tag = f"{hand_label}_{lid}"
                        if pts is None:
                            row += ["", "", ""]
                            moves[tag] = 0.0
                            continue
                        xyz = project_xy_to_xyz(
                            pts[lid, 0], pts[lid, 1], pkt.depth,
                            pkt.fx, pkt.fy, pkt.cx, pkt.cy, pkt.depth_scale_m,
                        )
                        if xyz is None:
                            row += ["", "", ""]
                            moves[tag] = 0.0
                        else:
                            row += [xyz[0], xyz[1], xyz[2]]
                            prev = last_xyz[tag]
                            if prev is None:
                                moves[tag] = 0.0
                            else:
                                dx, dy, dz = xyz[0]-prev[0], xyz[1]-prev[1], xyz[2]-prev[2]
                                moves[tag] = round(float(np.sqrt(dx*dx + dy*dy + dz*dz)), 4)
                            last_xyz[tag] = xyz

                for hand_label in HANDS:
                    for lid in KEYPOINTS_MOV:
                        row += [moves[f"{hand_label}_{lid}"]]

                csv_stream.append_row([str(v) for v in row])

                # --- update cumulatives for the plot + p-style bins ---
                for tag in ("L_0", "L_4", "R_0", "R_4"):
                    cum[tag] += float(moves.get(tag, 0.0))

                annotated = draw_annotations(img_bgr.copy(), L, R, frame_id)

                if viz_save_every > 0 and (frame_id % viz_save_every == 0):
                    try:
                        cv2.imwrite(str(mp_dir / f"frame_{frame_id:06d}.png"), annotated)
                    except Exception:
                        pass

                if preview is not None:
                    preview.update_hand_frame(annotated)

                plot_tick += 1
                if (plot_tick % PLOT_UPDATE_INTERVAL) == 0 and first_t_ns is not None and preview is not None:
                    elapsed_sec = max(1e-6, (pkt.t_ns - first_t_ns) / 1e9)
                    r0_cum = cum["R_0"]
                    r0_avg_speed = r0_cum / elapsed_sec
                    preview.push_r0_cumulative(r0_cum, r0_avg_speed)

                # p-style CSV bins every CUM_CSV_INTERVAL_SEC
                if first_t_ns is not None:
                    total_elapsed = (pkt.t_ns - first_t_ns) / 1e9
                    current_p = int(total_elapsed // CUM_CSV_INTERVAL_SEC)
                    if current_p > last_bin_written:
                        for p_idx in range(last_bin_written + 1, current_p + 1):
                            bins[p_idx] = {
                                "L0": cum["L_0"],
                                "L4": cum["L_4"],
                                "R0": cum["R_0"],
                                "R4": cum["R_4"],
                            }
                        last_bin_written = current_p
                        if last_bin_written > 0:
                            cum_csv.write_snapshot(cam_label, bins)

                    # NEW: centralized right-wrist speed trigger (average speed vs reference CSV)
                    if event_checker_enabled:
                        try:
                            speed_trigger.update(
                                elapsed_time_s=total_elapsed,
                                frame_idx=frame_id,
                                cumulative_movement=cum["R_0"],
                                preview_grid=preview,
                            )
                        except Exception:
                            # never let trigger failures stop the movement worker
                            pass

                logger.periodic_flush()

    except Exception as e:
        try:
            logger.error(f"[FATAL] Movement worker crashed: {e}")
        except Exception:
            pass
    finally:
        csv_stream.close()
        if bins:
            try:
                cum_csv.write_snapshot(cam_label, bins)
            except Exception:
                pass
        if first_t_ns is not None and last_t_ns is not None:
            try:
                elapsed_sec = max(1e-6, (last_t_ns - first_t_ns) / 1e9)
                stats_final_csv.write_final(
                    cam_label, elapsed_sec, cum["L_0"], cum["L_4"], cum["R_0"], cum["R_4"]
                )
            except Exception:
                pass

