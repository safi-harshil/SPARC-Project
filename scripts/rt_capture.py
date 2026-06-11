#!/usr/bin/env python3
"""
Real-time unified pipeline: capture (all cams), optional per-cam processing (movement & emotion), optional audio,
streaming CSV append, unified preview grid, and debounced logging.

Controls:
  • SPACE → toggle Pause/Resume (applies to capture + processing + audio)
  • g     → reopen the unified preview grid (if closed)
  • q     → close ONLY the preview grid (pipeline continues)
  • ESC or Ctrl+C → Stop gracefully

Notes
- Captures from ALL mapped/connected RealSense cameras.
- Use --process-mov-cams to select which cam labels to PROCESS for movement (others still capture & save if --save-every > 0).
- Use --process-emo-cams to select which cam labels to PROCESS for emotion (valence & arousal).
- Hand labeling: MediaPipe handedness with optional global flip baseline (--force-flip flip|same),
  then stabilized by proximity to a two-hand anchor (same logic as your batch script).
- 3D projection uses COLOR intrinsics with depth aligned to color.
- Movement is computed frame-to-frame for keypoints [0..4] per hand (wrist + thumb).
- Emotion is derived via MediaPipe Face Mesh geometry into valence [-1,1] and arousal [0,1].
"""

import os, sys, json, time, threading, queue, signal, argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from collections import deque

# 3rd-party
import numpy as np
import cv2
import pandas as pd
import pyrealsense2 as rs
import mediapipe as mp

# ---------- Constants ----------
CAMERA_SERIALS_FILE = os.path.expanduser("~/Desktop/SPARC-Project/camera_serials.txt")

WRIST_ID = 0
NUM_LANDMARKS = 21
WRIST_SEP_PX = 30
KEYPOINTS_MOV = [0, 1, 2, 3, 4]
HANDS = ("L", "R")

# anchor reset if we haven't seen BOTH hands for this many consecutive processed frames
ANCHOR_MISS_RESET = 300

COLOR_LEFT  = (0, 255, 0)
COLOR_RIGHT = (0, 0, 255)
COLOR_TEXT  = (255, 255, 255)

# Chart refresh cadence (frames) — affects charts only, not CSV cadence
PLOT_UPDATE_INTERVAL = 7  # ~5–10 frames as requested

# Cumulative CSV update cadence (seconds)
CUM_CSV_INTERVAL_SEC = 10.0

# Depth scale defaults
FALLBACK_DEPTH_SCALE = 0.0010000000474974513  # meters per unit (your constant)
_CAM2_DEPTH_SCALE_LOCK = threading.Lock()
_CAM2_DEPTH_SCALE: Optional[float] = None  # learned during run from cam2

# Make OpenCV more deterministic with threads
try:
    cv2.setNumThreads(1)
except Exception:
    pass

mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

# ---------- Cooperative control ----------
pause_event = threading.Event()  # set => paused
stop_event  = threading.Event()  # set => stop ASAP

def _sig_stop(signum, frame):
    print("\n[⛔] SIGINT → stopping…", flush=True)
    stop_event.set()

signal.signal(signal.SIGINT, _sig_stop)

# ---------- Debounced logger ----------
class DebouncedLogger:
    def __init__(self, log_path: Path, flush_interval_sec: int = 5):
        self.log_path = Path(log_path)
        self._lines: List[str] = []
        self._last_msg: Optional[str] = None
        self._repeat: int = 0
        self._lock = threading.Lock()
        self._flush_interval = max(1, int(flush_interval_sec))
        self._last_flush = time.monotonic()

    def _flush_repeat(self):
        if self._last_msg is not None:
            if self._repeat > 1:
                self._lines.append(f"[WARN x{self._repeat}] {self._last_msg}")
            else:
                self._lines.append(f"[WARN] {self._last_msg}")
        self._last_msg = None
        self._repeat = 0

    def warn(self, msg: str):
        with self._lock:
            if msg == self._last_msg:
                self._repeat += 1
            else:
                self._flush_repeat()
                self._last_msg = msg
                self._repeat = 1

    def info(self, msg: str):
        with self._lock:
            self._flush_repeat()
            self._lines.append(f"[INFO] {msg}")

    def periodic_flush(self, force: bool = False):
        now = time.monotonic()
        with self._lock:
            if force or (now - self._last_flush) >= self._flush_interval:
                self._flush_repeat()
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a") as f:
                    if self._lines:
                        f.write("\n".join(self._lines) + "\n")
                        self._lines.clear()
                self._last_flush = now

# ---------- Helpers ----------
def load_serial_map() -> Dict[str, str]:
    """
    camera_serials.txt format:
      cam1: <serial>
      cam2: <serial>
      cam3: <serial>
    We'll invert to serial->label.
    """
    serial_to_label: Dict[str, str] = {}
    if not os.path.isfile(CAMERA_SERIALS_FILE):
        print(f"[WARN] camera_serials file missing: {CAMERA_SERIALS_FILE}")
        return serial_to_label
    with open(CAMERA_SERIALS_FILE, "r") as f:
        for line in f:
            if ":" in line:
                label, serial = line.strip().split(":")
                serial_to_label[serial.strip()] = label.strip()
    return serial_to_label

def write_camera_info(profile: rs.pipeline_profile, serial: str, label: str,
                      out_txt: Path, out_json: Path) -> Tuple[dict, float]:
    device = profile.get_device()
    sensors = device.query_sensors()

    intr_json = {}
    with open(out_txt, "w") as f:
        f.write(f"Camera Label: {label}\n")
        f.write(f"Serial Number: {serial}\n")
        f.write(f"Firmware Version: {device.get_info(rs.camera_info.firmware_version)}\n")
        f.write(f"USB Port ID: {device.get_info(rs.camera_info.physical_port)}\n")
        f.write(f"Product Line: {device.get_info(rs.camera_info.product_line)}\n\n")
        for sensor in sensors:
            f.write(f"[Sensor: {sensor.get_info(rs.camera_info.name)}]\n")
            for opt in sensor.get_supported_options():
                try:
                    val = sensor.get_option(opt)
                    f.write(f"  {opt.name}: {val}\n")
                except Exception:
                    continue
            f.write("\n")
        f.write("[Active Streams]\n")

    for s in profile.get_streams():
        try:
            vs = s.as_video_stream_profile()
            intr = vs.get_intrinsics()
            stream_key = f"{s.stream_type().name.lower()}_{s.format().name.lower()}"
            intr_json[stream_key] = {
                "width": vs.width(),
                "height": vs.height(),
                "fps": vs.fps(),
                "fx": intr.fx, "fy": intr.fy, "cx": intr.ppx, "cy": intr.ppy,
                "model": intr.model.name, "coeffs": list(intr.coeffs),
            }
        except Exception as e:
            print(f"[WARN] Skipping stream info: {e}")

    # depth scale (prefer device, fallback to our constant)
    depth_scale = None
    try:
        depth_sensor = device.first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())
    except Exception:
        depth_scale = None
    if depth_scale is None or depth_scale <= 0:
        depth_scale = FALLBACK_DEPTH_SCALE

    intr_json["alignment"] = {"depth_to_color": True, "alignment_target": "color", "use_intrinsics": "color"}
    intr_json["depth_scale_m"] = depth_scale

    with open(out_json, "w") as jf:
        json.dump(intr_json, jf, indent=2)

    return intr_json, depth_scale

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

# ---------- Hand helpers ----------
def mp_landmarks_to_pixels(landmarks, w, h):
    pts = np.zeros((NUM_LANDMARKS, 2), dtype=float)
    for i, lm in enumerate(landmarks):
        pts[i] = [lm.x * w, lm.y * h]
    return pts

def euclid(a, b):
    return float(np.hypot(a[0]-b[0], a[1]-b[1]))

def euclid2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx*dx + dy*dy

def get_mediapipe_detections(results, w, h):
    """
    Returns: [{'pts': (21,2), 'label': 'L'/'R', 'score': float}, ...]
    """
    out = []
    if not results.multi_hand_landmarks:
        return out
    for hand_landmarks, handed in zip(results.multi_hand_landmarks, results.multi_handedness):
        pts = mp_landmarks_to_pixels(hand_landmarks.landmark, w, h)
        label_str = handed.classification[0].label
        score = float(handed.classification[0].score)
        label = 'L' if label_str.lower().startswith('l') else 'R'
        out.append({'pts': pts, 'label': label, 'score': score})
    return out

def collapse_overlap_mediapipe(dets, frame_id, logger: DebouncedLogger):
    """If both detected but wrists too close, keep higher score."""
    if len(dets) == 2:
        w0 = dets[0]['pts'][WRIST_ID]; w1 = dets[1]['pts'][WRIST_ID]
        d = float(np.hypot(w0[0]-w1[0], w0[1]-w1[1]))
        if d < WRIST_SEP_PX:
            keep_idx = 0 if dets[0]['score'] >= dets[1]['score'] else 1
            logger.warn(f"Frame {frame_id}: wrist distance {d:.1f}px < {WRIST_SEP_PX}px → collapse 2→1 (keep {keep_idx}).")
            return [dets[keep_idx]]
    return dets

def collapse_overlap_raw(raw_pts, prev_L, prev_R, logger: DebouncedLogger, frame_id):
    """If exactly two raw detections and wrists are very close, keep the one closer to previous anchors."""
    if len(raw_pts) == 2:
        w0 = raw_pts[0][WRIST_ID]; w1 = raw_pts[1][WRIST_ID]
        d = euclid(w0, w1)
        if d < WRIST_SEP_PX:
            if prev_L is not None and prev_R is not None:
                d0 = min(euclid2(w0, prev_L), euclid2(w0, prev_R))
                d1 = min(euclid2(w1, prev_L), euclid2(w1, prev_R))
                keep_idx = 0 if d0 <= d1 else 1
            else:
                keep_idx = 0
            logger.warn(f"Frame {frame_id}: raw wrists {d:.1f}px < {WRIST_SEP_PX}px → collapse 2→1 (keep {keep_idx}).")
            return [raw_pts[keep_idx]]
    return raw_pts

def label_by_proximity(current_pts_list, prev_left_wrist, prev_right_wrist):
    """Assign L/R by nearest wrist to the anchors (prev_left_wrist/prev_right_wrist)."""
    L = None; R = None
    if len(current_pts_list) == 2:
        w0 = current_pts_list[0][WRIST_ID]
        w1 = current_pts_list[1][WRIST_ID]
        d0 = euclid2(w0, prev_left_wrist)
        d1 = euclid2(w1, prev_left_wrist)
        if d0 <= d1:
            L, R = current_pts_list[0], current_pts_list[1]
        else:
            L, R = current_pts_list[1], current_pts_list[0]
    elif len(current_pts_list) == 1:
        w = current_pts_list[0][WRIST_ID]
        dL = euclid2(w, prev_left_wrist)
        dR = euclid2(w, prev_right_wrist)
        if dL <= dR:
            L, R = current_pts_list[0], None
        else:
            L, R = None, current_pts_list[0]
    else:
        L, R = None, None
    return {'L': L, 'R': R}

def draw_annotations(img, pts_L, pts_R, frame_id):
    if pts_L is not None:
        for (x, y) in pts_L:
            cv2.circle(img, (int(x), int(y)), 2, COLOR_LEFT, -1)
        wx, wy = pts_L[WRIST_ID]
        cv2.putText(img, f"L (frame {frame_id})", (int(wx)+5, int(wy)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LEFT, 1, cv2.LINE_AA)
    if pts_R is not None:
        for (x, y) in pts_R:
            cv2.circle(img, (int(x), int(y)), 2, COLOR_RIGHT, -1)
        wx, wy = pts_R[WRIST_ID]
        cv2.putText(img, f"R (frame {frame_id})", (int(wx)+5, int(wy)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_RIGHT, 1, cv2.LINE_AA)
    cv2.putText(img, f"id:{frame_id}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)
    return img

# ---------- 3D projection ----------
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
class EmotionCSVStream:
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
                self._ensure_open()
                self._fh.write("frame_id,t_ns,valence,arousal\n")
                self._header_written = True

    def append_row(self, frame_id: int, t_ns: int, val: float, aro: float):
        with self._lock:
            self._ensure_open()
            self._fh.write(f"{frame_id},{t_ns},{val:.4f},{aro:.4f}\n")
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

class CumMovementWideCSV:
    """
    Cumulative movement CSV (wide format) “p-style”, updated every 10s:
    cam,L0_p1..L0_pN,L4_p1..L4_pN,R0_p1..R0_pN,R4_p1..R4_pN
    We rewrite the file in place on each update to extend columns.
    """
    def __init__(self, csv_path: Path):
        self.csv_path = Path(csv_path)
        self._lock = threading.Lock()

    def write_snapshot(self, cam_label: str, bins: Dict[int, Dict[str, float]]):
        """
        bins: {p_idx -> {'L0':val,'L4':val,'R0':val,'R4':val}}, p_idx starts at 1
        """
        with self._lock:
            pmax = max(bins.keys()) if bins else 0
            # Build column order similar to your reference: group by landmark, then by period
            cols = ["cam"]
            for tag in ("L0", "L4", "R0", "R4"):
                cols += [f"{tag}_p{p}" for p in range(1, pmax+1)]
            # Build single row
            row = [cam_label]
            for tag in ("L0", "L4", "R0", "R4"):
                for p in range(1, pmax+1):
                    v = bins.get(p, {}).get(tag, 0.0)
                    row.append(f"{v:.6f}")
            # Rewrite file
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w") as f:
                f.write(",".join(cols) + "\n")
                f.write(",".join(map(str, row)) + "\n")

class MovementStatsFinalCSV:
    """Write a one-row summary CSV at end of run for totals & avg speeds."""
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

# ---------- Emotion (valence/arousal) heuristics ----------
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

# ---------- Unified preview grid ----------
class PreviewGrid:
    """
    Single OpenCV window that renders a 2x2 grid:
      TL: movement cam
      TR: emotion cam
      BL: cumulative movement (R_0) with live avg speed at tip
      BR: valence & arousal lines
    Press 'q' to close (pipeline continues). Press 'g' (terminal) to reopen.
    """
    def __init__(self, title="Unified Preview", history_len=600, target_fps=30):
        self.title = title
        self.history_len = int(max(120, history_len))
        self.target_dt = 1.0 / float(max(5, target_fps))
        self.lock = threading.Lock()

        self.hand_frame = None
        self.emo_frame = None
        self.r0_cum_hist = deque(maxlen=self.history_len)
        self.r0_last_avg_speed = 0.0
        self.va_hist = deque(maxlen=self.history_len)  # list of (valence, arousal)

        self._run_flag = threading.Event()
        self._is_open = threading.Event()
        self._thread = None

        self.cell_h = 360
        self.cell_w = 640

    def start(self):
        if self._thread is not None:
            return
        self._run_flag.set()
        self._is_open.set()
        self._thread = threading.Thread(target=self._loop, name="preview-grid", daemon=True)
        self._thread.start()

    def reopen_window(self):
        self._is_open.set()

    def close_window(self):
        self._is_open.clear()
        try:
            cv2.destroyWindow(self.title)
        except Exception:
            pass

    def stop(self):
        self._run_flag.clear()
        self._is_open.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        try:
            cv2.destroyWindow(self.title)
        except Exception:
            pass
        self._thread = None

    def update_hand_frame(self, bgr_img: np.ndarray):
        with self.lock:
            self.hand_frame = bgr_img.copy()
            self._maybe_set_cell_size(bgr_img)

    def update_emo_frame(self, bgr_img: np.ndarray):
        with self.lock:
            self.emo_frame = bgr_img.copy()
            self._maybe_set_cell_size(bgr_img)

    def push_r0_cumulative(self, cum_val: float, avg_speed: float):
        with self.lock:
            self.r0_cum_hist.append(float(cum_val))
            self.r0_last_avg_speed = float(avg_speed)

    def push_valence_arousal(self, val: float, aro: float):
        with self.lock:
            self.va_hist.append((float(val), float(aro)))

    def _maybe_set_cell_size(self, img):
        h, w = img.shape[:2]
        maxw = 640
        scale = min(1.0, maxw / float(w)) if w > 0 else 1.0
        self.cell_w = int(w * scale) if w > 0 else self.cell_w
        self.cell_h = int(h * scale) if h > 0 else self.cell_h

    def _render_plot_line(self, values, ymin, ymax, label, avg_speed=None, with_axis=True):
        W, H = self.cell_w, self.cell_h
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        x0, y0 = 40, 20
        x1, y1 = W-20, H-40
        width  = max(1, x1 - x0)
        height = max(1, y1 - y0)

        if with_axis:
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (80,80,80), 1)
            cv2.putText(canvas, label, (x0+4, y0+16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{ymax:.2f}", (x1+2, y0+6),  cv2.FONT_HERSHEY_PLAIN, 1, (180,180,180), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{(ymax+ymin)/2:.2f}", (x1+2, (y0+y1)//2+4), cv2.FONT_HERSHEY_PLAIN, 1, (180,180,180), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{ymin:.2f}", (x1+2, y1),    cv2.FONT_HERSHEY_PLAIN, 1, (180,180,180), 1, cv2.LINE_AA)

        if not values or len(values) < 2:
            if avg_speed is not None:
                cv2.putText(canvas, f"avg: {avg_speed:.3f}", (x0+6, y0+34), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            return canvas

        def vy(v):
            if ymax == ymin: return y1
            r = (v - ymin) / (ymax - ymin)
            r = max(0.0, min(1.0, r))
            return int(round(y1 - r * height))

        n = len(values)
        pts = []
        for i, v in enumerate(values):
            x = x0 + int(round(i * (width-1) / (n-1)))
            y = vy(v)
            pts.append((x, y))
        for i in range(1, len(pts)):
            cv2.line(canvas, pts[i-1], pts[i], (0, 255, 255), 2)

        tip = pts[-1]
        cv2.circle(canvas, tip, 3, (255, 255, 255), -1)
        if avg_speed is not None:
            tx = min(tip[0] + 6, x1 - 80)
            ty = max(y0 + 24, tip[1] - 6)
            cv2.putText(canvas, f"avg: {avg_speed:.3f}", (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

        return canvas

    def _render_va(self, tuples_va):
        W, H = self.cell_w, self.cell_h
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        x0, y0 = 40, 20
        x1, y1 = W-20, H-40
        width  = max(1, x1 - x0)

        cv2.rectangle(canvas, (x0, y0), (x1, y1), (80,80,80), 1)
        cv2.putText(canvas, "Valence (G)  Arousal (M)", (x0+4, y0+16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "1.0", (x1+2, y0+6), cv2.FONT_HERSHEY_PLAIN, 1, (180,180,180), 1, cv2.LINE_AA)
        cv2.putText(canvas, "0.0", (x1+2, y1),    cv2.FONT_HERSHEY_PLAIN, 1, (180,180,180), 1, cv2.LINE_AA)
        cv2.putText(canvas, "+1",  (x0-28, y0+6), cv2.FONT_HERSHEY_PLAIN, 1, (120,255,120), 1, cv2.LINE_AA)
        cv2.putText(canvas, " 0",  (x0-24, (y0+y1)//2+4), cv2.FONT_HERSHEY_PLAIN, 1, (200,200,200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "-1",  (x0-28, y1), cv2.FONT_HERSHEY_PLAIN, 1, (255,120,120), 1, cv2.LINE_AA)

        if not tuples_va or len(tuples_va) < 2:
            return canvas

        def vy_val(v):  # [-1,1]
            r = (v + 1.0) / 2.0; r = max(0.0, min(1.0, r))
            return int(round(y1 - r * (y1 - y0)))
        def vy_aro(a):  # [0,1]
            r = max(0.0, min(1.0, a))
            return int(round(y1 - r * (y1 - y0)))

        prev_v = None; prev_a = None
        n = len(tuples_va)
        for i, (v, a) in enumerate(tuples_va):
            x = x0 + int(round(i * (width-1) / (n-1)))
            yv = vy_val(v); ya = vy_aro(a)
            if prev_v is not None:
                cv2.line(canvas, prev_v, (x, yv), (0, 255, 0), 2)
            if prev_a is not None:
                cv2.line(canvas, prev_a, (x, ya), (0, 180, 255), 2)
            prev_v = (x, yv); prev_a = (x, ya)
        return canvas

    def _compose_grid(self, hand_img, emo_img, mov_plot, va_plot):
        def fit(img):
            if img is None:
                return np.zeros((self.cell_h, self.cell_w, 3), dtype=np.uint8)
            if img.shape[1] != self.cell_w or img.shape[0] != self.cell_h:
                return cv2.resize(img, (self.cell_w, self.cell_h))
            return img

        tl = fit(hand_img)
        tr = fit(emo_img)
        bl = fit(mov_plot)
        br = fit(va_plot)

        H = self.cell_h * 2
        W = self.cell_w * 2
        grid = np.zeros((H, W, 3), dtype=np.uint8)
        grid[0:self.cell_h, 0:self.cell_w] = tl
        grid[0:self.cell_h, self.cell_w:W] = tr
        grid[self.cell_h:H, 0:self.cell_w] = bl
        grid[self.cell_h:H, self.cell_w:W] = br

        cv2.putText(grid, "Movement Camera", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(grid, "Emotion Camera", (self.cell_w+10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(grid, "Cumulative R_0 movement", (10, self.cell_h+22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(grid, "Valence & Arousal", (self.cell_w+10, self.cell_h+22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        return grid

    def _loop(self):
        while self._run_flag.is_set():
            if self._is_open.is_set():
                with self.lock:
                    hand = None if self.hand_frame is None else self.hand_frame.copy()
                    emo  = None if self.emo_frame is None else self.emo_frame.copy()
                    r0_hist = list(self.r0_cum_hist)
                    va_hist = list(self.va_hist)
                    avg_spd = self.r0_last_avg_speed

                ymax = max(1.0, max(r0_hist) if r0_hist else 1.0)
                mov_plot = self._render_plot_line(
                    r0_hist, ymin=0.0, ymax=ymax, label="R_0 cumulative (mm)",
                    avg_speed=avg_spd, with_axis=True
                )
                va_plot  = self._render_va(va_hist)
                grid = self._compose_grid(hand, emo, mov_plot, va_plot)

                cv2.imshow(self.title, grid)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.close_window()
            time.sleep(self.target_dt)

# Global preview instance
PREVIEW: Optional[PreviewGrid] = None

# ---------- Keyboard controls ----------
def toggle_pause(reason="keyboard"):
    if not pause_event.is_set():
        pause_event.set()
        print(f"[⏸] Pause ({reason})", flush=True)
    else:
        pause_event.clear()
        print(f"[▶] Resume ({reason})", flush=True)

def reopen_preview():
    global PREVIEW
    if PREVIEW is not None:
        PREVIEW.reopen_window()
        print("[🪟] Preview reopened.", flush=True)

def start_keyboard_listener():
    """Terminal listener; works even with --viz-live off. Skips if stdin not a TTY."""
    if not sys.stdin.isatty():
        return None
    import termios, tty, select
    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)

    def _run():
        try:
            tty.setcbreak(fd)
            while not stop_event.is_set():
                r, _, _ = select.select([fd], [], [], 0.05)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                if ch == ' ':
                    toggle_pause("space")
                elif ch == '\x1b':  # ESC
                    print("[⛔] ESC pressed → stopping.", flush=True)
                    stop_event.set()
                    break
                elif ch in ('g', 'G'):
                    reopen_preview()
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
            except Exception:
                pass

    t = threading.Thread(target=_run, daemon=True, name="kbd-listener")
    t.start()
    return t

# ---------- Movement Processor thread ----------
def processor_worker(cam_label: str,
                     q: "queue.Queue[FramePacket]",
                     out_dir: Path,
                     force_flip: str,
                     stride: int,
                     viz_live: str,
                     viz_save_every: int,
                     csv_flush_every: int,
                     log_flush_sec: int):
    # per-cam dirs & logger
    cam_dir = out_dir / cam_label
    (cam_dir / "CSV").mkdir(parents=True, exist_ok=True)
    (cam_dir / "logs").mkdir(parents=True, exist_ok=True)
    (cam_dir / "color_mp").mkdir(parents=True, exist_ok=True)
    logger = DebouncedLogger(cam_dir / "logs" / "hand_detection_rt.log", flush_interval_sec=log_flush_sec)
    csv_stream = CSVStream(cam_dir / "CSV" / "hand_landmark_rt.csv", flush_every=csv_flush_every)
    csv_stream.write_header_if_needed()

    # Extra CSVs
    cum_csv = CumMovementWideCSV(cam_dir / "CSV" / "cumulative_movement_rt.csv")
    stats_final_csv = MovementStatsFinalCSV(cam_dir / "CSV" / "movement_stats_rt.csv")

    # state
    last_xyz: Dict[str, Optional[Tuple[float,float,float]]] = {f"{h}_{k}": None for h in HANDS for k in KEYPOINTS_MOV}
    prev_two_left_wrist: Optional[Tuple[float,float]] = None
    prev_two_right_wrist: Optional[Tuple[float,float]] = None
    have_anchor = False
    consecutive_no_two = 0

    # cumulative movement trackers for L{0,4} and R{0,4}
    cum = {"L_0": 0.0, "L_4": 0.0, "R_0": 0.0, "R_4": 0.0}

    # timing
    first_t_ns: Optional[int] = None
    last_t_ns: Optional[int] = None

    # bins (10s) for wide CSV: p_idx -> snapshot dict('L0','L4','R0','R4')
    bins: Dict[int, Dict[str, float]] = {}
    last_bin_written = 0  # highest p_idx written

    # chart cadence
    plot_tick = 0

    with mp_hands.Hands(
        static_image_mode=False,
        model_complexity=1,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands_model:

        processed = 0
        saved_counter = 0

        while not stop_event.is_set():
            try:
                pkt: FramePacket = q.get(timeout=0.1)
            except queue.Empty:
                logger.periodic_flush()
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
            h, w = img_bgr.shape[:2]
            frame_id = pkt.frame_id

            results = hands_model.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            dets = get_mediapipe_detections(results, w, h)
            dets = collapse_overlap_mediapipe(dets, frame_id, logger)

            L_lab = None; R_lab = None
            raw_pts = []
            for d in dets[:2]:
                raw_pts.append(d['pts'])
                if d['label'] == 'L':
                    L_lab = d['pts']
                elif d['label'] == 'R':
                    R_lab = d['pts']

            if force_flip == "flip":
                L_lab, R_lab = R_lab, L_lab

            if have_anchor:
                raw_pts = collapse_overlap_raw(raw_pts, prev_two_left_wrist, prev_two_right_wrist, logger, frame_id)
                labeled = label_by_proximity(raw_pts, prev_two_left_wrist, prev_two_right_wrist)
                L = labeled['L']; R = labeled['R']

                if L is not None and R is not None:
                    prev_two_left_wrist  = tuple(L[WRIST_ID])
                    prev_two_right_wrist = tuple(R[WRIST_ID])
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
                    prev_two_left_wrist  = tuple(L[WRIST_ID])
                    prev_two_right_wrist = tuple(R[WRIST_ID])
                    have_anchor = True
                    consecutive_no_two = 0
                    logger.info(f"Anchor set at frame {frame_id} (two-hand reference acquired).")

            # Build landmark row + movement
            row: List[float] = [frame_id, pkt.t_ns]

            for lid in range(NUM_LANDMARKS):
                if L is not None:
                    row += [float(L[lid,0]), float(L[lid,1])]
                else:
                    row += ["", ""]
            for lid in range(NUM_LANDMARKS):
                if R is not None:
                    row += [float(R[lid,0]), float(R[lid,1])]
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
                        pts[lid,0], pts[lid,1], pkt.depth,
                        pkt.fx, pkt.fy, pkt.cx, pkt.cy, pkt.depth_scale_m
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
                            dx = xyz[0] - prev[0]
                            dy = xyz[1] - prev[1]
                            dz = xyz[2] - prev[2]
                            moves[tag] = round(float(np.sqrt(dx*dx + dy*dy + dz*dz)), 4)
                        last_xyz[tag] = xyz

            for hand_label in HANDS:
                for lid in KEYPOINTS_MOV:
                    row += [moves[f"{hand_label}_{lid}"]]

            csv_stream.append_row([str(v) for v in row])

            # Update cumulatives for L{0,4} and R{0,4}
            for tag in ("L_0", "L_4", "R_0", "R_4"):
                cum[tag] += float(moves.get(tag, 0.0))

            # Annotated frame to preview (TL)
            annotated = draw_annotations(img_bgr.copy(), L, R, frame_id)
            if PREVIEW is not None:
                PREVIEW.update_hand_frame(annotated)

            # Charts refresh cadence only
            plot_tick += 1
            if (plot_tick % PLOT_UPDATE_INTERVAL) == 0 and first_t_ns is not None:
                elapsed_sec = max(1e-6, (pkt.t_ns - first_t_ns) / 1e9)
                r0_cum = cum["R_0"]
                r0_avg_speed = r0_cum / elapsed_sec  # mm/s
                if PREVIEW is not None:
                    PREVIEW.push_r0_cumulative(r0_cum, r0_avg_speed)

            # Cumulative CSV update every 10 seconds (p-style bins)
            if first_t_ns is not None:
                total_elapsed = (pkt.t_ns - first_t_ns) / 1e9
                current_p = int(total_elapsed // CUM_CSV_INTERVAL_SEC)  # 0,1,2,...
                if current_p > last_bin_written:
                    # fill all missing bins up to current_p with current snapshot
                    for p_idx in range(last_bin_written + 1, current_p + 1):
                        bins[p_idx] = {
                            "L0": cum["L_0"], "L4": cum["L_4"],
                            "R0": cum["R_0"], "R4": cum["R_4"]
                        }
                    last_bin_written = current_p
                    # write wide CSV (p starts at 1)
                    if last_bin_written > 0:
                        # shift indices to start at 1 for naming (already 1..p)
                        cum_csv.write_snapshot(cam_label, bins)

            logger.periodic_flush()

        # cleanup + final stats CSV (one line at end)
        csv_stream.close()
        # Final wide CSV rewrite (ensure latest snapshot persisted)
        if bins:
            cum_csv.write_snapshot(cam_label, bins)
        # Final stats row
        if first_t_ns is not None and last_t_ns is not None:
            elapsed_sec = max(1e-6, (last_t_ns - first_t_ns) / 1e9)
            stats_final_csv.write_final(cam_label, elapsed_sec,
                                        cum["L_0"], cum["L_4"], cum["R_0"], cum["R_4"])

# ---------- Emotion Processor thread ----------
def emotion_worker(cam_label: str,
                   q: "queue.Queue[FramePacket]",
                   out_dir: Path,
                   stride: int,
                   viz_live: str,
                   csv_flush_every: int,
                   log_flush_sec: int,
                   history_len: int):
    cam_dir = out_dir / cam_label
    (cam_dir / "CSV").mkdir(parents=True, exist_ok=True)
    (cam_dir / "logs").mkdir(parents=True, exist_ok=True)
    (cam_dir / "color_mp").mkdir(parents=True, exist_ok=True)

    logger = DebouncedLogger(cam_dir / "logs" / "emotion_rt.log", flush_interval_sec=log_flush_sec)
    csv_stream = EmotionCSVStream(cam_dir / "CSV" / "emotion_rt.csv", flush_every=csv_flush_every)
    csv_stream.write_header_if_needed()

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_model:

        processed = 0

        while not stop_event.is_set():
            try:
                pkt: FramePacket = q.get(timeout=0.1)
            except queue.Empty:
                logger.periodic_flush()
                continue

            while pause_event.is_set() and not stop_event.is_set():
                time.sleep(0.05)
            if stop_event.is_set():
                break

            processed += 1
            if stride > 1 and (processed - 1) % stride != 0:
                continue

            img_bgr = pkt.color
            h, w = img_bgr.shape[:2]
            frame_id = pkt.frame_id

            res = face_model.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            val, aro = 0.0, 0.0
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0]
                pts = np.array([[p.x * w, p.y * h] for p in lm.landmark], dtype=np.float32)
                try:
                    val, aro = map_to_valence_arousal(pts)
                except Exception as e:
                    logger.warn(f"frame {frame_id}: VA mapping error: {e}")
            else:
                logger.warn(f"frame {frame_id}: no face detected")

            csv_stream.append_row(frame_id, pkt.t_ns, val, aro)

            annotated = img_bgr.copy()
            cv2.putText(annotated, f"Valence: {val:+.2f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(annotated, f"Arousal: {aro:.2f}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

            if PREVIEW is not None:
                PREVIEW.update_emo_frame(annotated)
                PREVIEW.push_valence_arousal(val, aro)

            logger.periodic_flush()

        csv_stream.close()

# ---------- Capture thread ----------
def capture_worker(serial: str,
                   cam_label: str,
                   out_dir: Path,
                   duration_sec: float,
                   save_every: int,
                   filters_on: bool,
                   q_mov: Optional["queue.Queue[FramePacket]"],
                   q_emo: Optional["queue.Queue[FramePacket]"],
                   backpressure: str):
    cam_dir = out_dir / cam_label
    color_dir = cam_dir / "color"
    depth_dir = cam_dir / "depth"
    (cam_dir / "logs").mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    logger = DebouncedLogger(cam_dir / "logs" / "capture_rt.log", flush_interval_sec=5)

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        profile = pipeline.start(cfg)
    except Exception as e:
        print(f"[ERROR] Failed to start RealSense {cam_label} ({serial}): {e}")
        return

    align = rs.align(rs.stream.color)
    if filters_on:
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        hole = rs.hole_filling_filter()
    else:
        spatial = temporal = hole = None

    # Write camera info & intrinsics json (returns a scale; we may override below)
    info_txt  = cam_dir / f"camera_info_{serial}.txt"
    info_json = cam_dir / f"camera_intrinsics_{serial}.json"
    intr_json, depth_scale_m = write_camera_info(profile, serial, cam_label, info_txt, info_json)

    # Prefer cam2's scale for others if available; ensure fallback constant
    with _CAM2_DEPTH_SCALE_LOCK:
        if cam_label == "cam2":
            _CAM2_DEPTH_SCALE = depth_scale_m
        else:
            if (depth_scale_m is None) or (depth_scale_m <= 0):
                if _CAM2_DEPTH_SCALE is not None:
                    depth_scale_m = _CAM2_DEPTH_SCALE
                else:
                    depth_scale_m = FALLBACK_DEPTH_SCALE

    # Use color intrinsics
    color_key = next((k for k in intr_json.keys() if k.startswith("color_")), None)
    if color_key is None:
        fx = 604.7; fy = 604.9; cx = 313.8; cy = 252.7  # fallback intrinsics
    else:
        fx = float(intr_json[color_key]["fx"]); fy = float(intr_json[color_key]["fy"])
        cx = float(intr_json[color_key]["cx"]); cy = float(intr_json[color_key]["cy"])

    print(f"[INFO] 🎥 {cam_label} → depth_scale={depth_scale_m:.12f} m/unit | fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    frame_id = 0
    active_elapsed = 0.0
    last_tick = time.monotonic()

    def _publish(qtarget, pkt):
        if qtarget is None:
            return
        try:
            if backpressure == "block":
                qtarget.put(pkt, timeout=0.01)
            else:
                qtarget.put_nowait(pkt)
        except queue.Full:
            try:
                _ = qtarget.get_nowait()
            except Exception:
                pass
            try:
                qtarget.put_nowait(pkt)
            except Exception:
                pass

    while not stop_event.is_set() and active_elapsed < duration_sec:
        if pause_event.is_set():
            last_tick = time.monotonic()
            time.sleep(0.02)
            continue

        try:
            frames = pipeline.wait_for_frames()
        except Exception:
            continue

        aligned = align.process(frames)
        c = aligned.get_color_frame()
        d = aligned.get_depth_frame()
        if not c or not d:
            continue

        if filters_on:
            try:
                d = spatial.process(d)
                d = temporal.process(d)
                d = hole.process(d)
            except Exception:
                pass

        color_img = np.asanyarray(c.get_data())
        depth_img = np.asanyarray(d.get_data())

        # Save raw
        if save_every > 0 and (frame_id % save_every == 0):
            cv2.imwrite(str(color_dir / f"frame_{frame_id:06d}.png"), color_img)
            cv2.imwrite(str(depth_dir / f"frame_{frame_id:06d}.tiff"), depth_img, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

        # Publish to processing pipelines (if enabled for this cam)
        pkt = FramePacket(
            cam_label=cam_label,
            frame_id=frame_id,
            t_ns=time.monotonic_ns(),
            rs_ts_ms=c.get_timestamp() if hasattr(c, "get_timestamp") else 0.0,
            color=color_img,
            depth=depth_img,
            fx=fx, fy=fy, cx=cx, cy=cy,
            depth_scale_m=depth_scale_m
        )
        _publish(q_mov, pkt)
        _publish(q_emo, pkt)

        frame_id += 1

        now = time.monotonic()
        active_elapsed += (now - last_tick)
        last_tick = now

    try:
        pipeline.stop()
    except Exception:
        pass
    logger.info(f"Finished capture {cam_label}: frames={frame_id}, active_elapsed={active_elapsed:.2f}s")
    logger.periodic_flush(force=True)

# ---------- Audio (minimal) ----------
try:
    from mic_config import VALID_MIC_IDS as _VALID_MIC_IDS
except Exception:
    _VALID_MIC_IDS = []

def audio_worker(device_str: str, out_dir: Path, duration_sec: float, rate: int):
    if duration_sec <= 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    wav_tmp = out_dir / f"mic_{device_str.replace(':','').replace(',','')}_{ts}.part"
    wav_final = Path(str(wav_tmp).replace(".part", ".wav"))

    import subprocess, signal as pysignal
    cmd = ["arecord", "-D", device_str, "-f", "cd", "-c", "1", "-r", str(rate), "-t", "wav", str(wav_tmp)]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    active_elapsed = 0.0
    last = time.monotonic()
    paused_child = False

    try:
        while not stop_event.is_set() and active_elapsed < duration_sec:
            now = time.monotonic()
            if pause_event.is_set():
                if proc.poll() is None and not paused_child:
                    os.kill(proc.pid, pysignal.SIGSTOP); paused_child = True
                last = now
                time.sleep(0.05)
                continue
            else:
                if proc.poll() is None and paused_child:
                    os.kill(proc.pid, pysignal.SIGCONT); paused_child = False
                active_elapsed += (now - last)
                last = now
                time.sleep(0.02)

        if proc and proc.poll() is None:
            if paused_child:
                os.kill(proc.pid, pysignal.SIGCONT); time.sleep(0.05)
            os.kill(proc.pid, pysignal.SIGINT)
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.terminate()
        if wav_tmp.exists():
            wav_tmp.replace(wav_final)
            print(f"[✅] Audio saved: {wav_final}")
    except Exception as e:
        print(f"[ERROR] audio({device_str}): {e}")
        try:
            proc.kill()
        except Exception:
            pass

# ---------- Args ----------
def build_argparser():
    ap = argparse.ArgumentParser(description="Real-time unified capture+process+audio pipeline")
    ap.add_argument("--output-dir", required=True, help="Base output directory")
    ap.add_argument("--duration-sec", type=float, required=True, help="ACTIVE duration in seconds")
    ap.add_argument("--save-every", type=int, default=1, help="Save raw color/depth every Nth frame (0=off)")
    ap.add_argument("--filters", choices=["on","off"], default="off", help="Depth filters on/off")
    ap.add_argument("--viz-live", choices=["off","window"], default="off", help="Preview window (unified 2x2 grid)")
    ap.add_argument("--viz-save-every", type=int, default=3, help="Save annotated movement preview every N frames (0=off)")
    ap.add_argument("--force-flip", choices=["flip","same"], default="flip", help="Global handedness flip baseline")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame for movement")
    ap.add_argument("--backpressure", choices=["drop-latest","block"], default="drop-latest", help="Processor queue policy")
    ap.add_argument("--csv-flush", type=int, default=30, help="CSV flush interval (frames) for movement landmarks")
    ap.add_argument("--log-flush-sec", type=int, default=5, help="Logger flush interval")
    ap.add_argument("--health-interval-sec", type=int, default=5, help="(reserved) health log cadence")

    # —— Movement vs Emotion camera selection ——
    ap.add_argument("--process-mov-cams", nargs="*", help="Labels to process for hand movement.")
    ap.add_argument("--process-cams", nargs="*", help=argparse.SUPPRESS)  # deprecated alias
    ap.add_argument("--process-emo-cams", nargs="*", help="Labels to process for emotion (valence/arousal).")

    # —— Emotion tunables ——
    ap.add_argument("--emo-history", type=int, default=240, help="Frames kept in on-screen VA plot (per cam)")
    ap.add_argument("--emo-stride", type=int, default=1, help="Process every Nth frame for emotion")
    ap.add_argument("--emo-csv-flush", type=int, default=30, help="CSV flush interval (frames) for emotion")

    # —— Audio ——
    ap.add_argument("--audio-out", default=None, help="Audio output directory (default: <output-dir>/audio)")
    ap.add_argument("--audio-duration-sec", type=float, default=0, help="Active duration for audio (0=off)")
    ap.add_argument("--rate", type=int, choices=[44100,48000], default=44100, help="Audio sample rate")
    ap.add_argument("--rt-publish", choices=["on","off"], default="off", help="(reserved) RT audio publish (queue)")
    ap.add_argument("--audio-backpressure", choices=["drop-latest","block"], default="drop-latest", help="(reserved)")
    return ap

# ---------- Main ----------
def main():
    args = build_argparser().parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    _kbd = start_keyboard_listener()

    # Discover cams
    serial_to_label = load_serial_map()
    ctx = rs.context()
    connected = [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]
    active = [(s, serial_to_label.get(s, f"cam_{s[-4:]}")) for s in connected]
    if not active:
        print("[ERROR] No RealSense cameras found.")
        return
    print(f"[INFO] Connected cams: {', '.join([f'{lab}({s})' for s,lab in active])}")

    # Movement selection:
    if args.process_mov_cams is not None:
        process_set_mov = set([lab.strip() for lab in args.process_mov_cams if lab and lab.strip()])
    elif getattr(args, "process_cams", None) is not None:
        process_set_mov = set([lab.strip() for lab in args.process_cams if lab and lab.strip()])
        print("[WARN] --process-cams is deprecated. Use --process-mov-cams instead.")
    else:
        process_set_mov = set(lab for _, lab in active)  # default: all for movement

    # Emotion selection:
    if args.process_emo_cams is None:
        process_set_emo = set()  # default: none for emotion unless specified
    else:
        process_set_emo = set([lab.strip() for lab in args.process_emo_cams if lab and lab.strip()])

    print(f"[INFO] Movement cams: {sorted(process_set_mov) if process_set_mov else 'NONE'}")
    print(f"[INFO] Emotion cams:  {sorted(process_set_emo) if process_set_emo else 'NONE'}")

    # Start unified preview if requested
    global PREVIEW
    PREVIEW = None
    if args.viz_live == "window":
        PREVIEW = PreviewGrid(title="Unified Preview", history_len=max(240, args.emo_history), target_fps=30)
        PREVIEW.start()

    # Spawn per-cam queues & threads
    cap_threads = []
    proc_threads = []
    queues_mov: Dict[str, queue.Queue] = {}
    queues_emo: Dict[str, queue.Queue] = {}

    for serial, label in active:
        q_mov = None
        q_emo = None

        if label in process_set_mov:
            q_mov = queue.Queue(maxsize=8)
            queues_mov[label] = q_mov
            t_proc_mov = threading.Thread(
                target=processor_worker,
                args=(label, q_mov, out_dir, args.force_flip, max(1,args.stride),
                      args.viz_live, max(0,args.viz_save_every), max(1,args.csv_flush), max(1,args.log_flush_sec)),
                daemon=True, name=f"proc-mov-{label}"
            )
            t_proc_mov.start()
            proc_threads.append(t_proc_mov)

        if label in process_set_emo:
            q_emo = queue.Queue(maxsize=8)
            queues_emo[label] = q_emo
            t_proc_emo = threading.Thread(
                target=emotion_worker,
                args=(label, q_emo, out_dir, max(1,args.emo_stride),
                      args.viz_live, max(1,args.emo_csv_flush), max(1,args.log_flush_sec), max(10,args.emo_history)),
                daemon=True, name=f"proc-emo-{label}"
            )
            t_proc_emo.start()
            proc_threads.append(t_proc_emo)

        t_cap = threading.Thread(
            target=capture_worker,
            args=(serial, label, out_dir, float(args.duration_sec), max(0,args.save_every),
                  (args.filters=="on"), queues_mov.get(label, None), queues_emo.get(label, None), args.backpressure),
            daemon=True, name=f"cap-{label}"
        )
        t_cap.start()
        cap_threads.append(t_cap)

    # Audio
    audio_threads = []
    aud_dir = Path(args.audio_out) if args.audio_out else (out_dir / "audio")
    if args.audio_duration_sec > 0 and _VALID_MIC_IDS:
        for dev in _VALID_MIC_IDS:
            t = threading.Thread(target=audio_worker, args=(dev, aud_dir, float(args.audio_duration_sec), int(args.rate)),
                                 daemon=True, name=f"aud-{dev}")
            t.start()
            audio_threads.append(t)
        print(f"[INFO] 🎙 Audio enabled on {len(audio_threads)} mic(s) → {aud_dir}")
    elif args.audio_duration_sec > 0:
        print("[WARN] audio requested but mic_config.VALID_MIC_IDS is empty; skipping audio.")

    # Wait
    try:
        for t in cap_threads:
            t.join()
        stop_event.set()
        for t in proc_threads:
            t.join()
        for t in audio_threads:
            t.join()
    finally:
        if PREVIEW is not None:
            PREVIEW.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    print("[🏁] Done.")

if __name__ == "__main__":
    main()
