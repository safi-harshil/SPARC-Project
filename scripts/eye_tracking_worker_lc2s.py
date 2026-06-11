#!/usr/bin/env python3
"""
eye_tracking_worker_l2cs.py
---------------------------
Drop-in replacement for eye_tracking_worker.py.
Replaces iris-ratio heuristics with L2CS-Net ML gaze estimation.

Key changes from eye_tracking_worker.py:
  - Imports L2CSGazeEstimator
  - Loads model once after FaceMesh init
  - Removes gaze_x/gaze_y heuristic block
  - Calls gaze_model.estimate_with_vector(color_img, face_bbox)
  - Computes face bbox from MediaPipe landmarks for face-cropped inference
  - CSV now stores eye_yaw_deg / eye_pitch_deg instead of gaze_x / gaze_y
"""

import sys
import time
from pathlib import Path
import traceback

import cv2
import numpy as np
import mediapipe as mp

from logger_utils import get_eye_logger
from eye_tracking_ui import draw_hud
from control_flags import stop_event, pause_event

# ── Add scripts/ to path so gaze_models package is importable ────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from gaze_models.l2cs_gaze import L2CSGazeEstimator


# ── MediaPipe landmark indices ────────────────────────────────────────────────
LEFT_EYE_LEFT   = 33
LEFT_EYE_RIGHT  = 133
LEFT_EYE_TOP    = 159
LEFT_EYE_BOTTOM = 145

RIGHT_EYE_LEFT   = 362
RIGHT_EYE_RIGHT  = 263
RIGHT_EYE_TOP    = 386
RIGHT_EYE_BOTTOM = 374

LEFT_EYE_CENTER  = 468
RIGHT_EYE_CENTER = 473

LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# Face mesh landmarks used for head-pose PnP solve
FACE_MODEL_POINTS = np.array([
    ( 0.0,   0.0,   0.0),   # nose tip       → lm 1
    ( 0.0, -63.6, -12.5),   # chin           → lm 152
    (-43.3, 32.7, -26.0),   # left eye outer → lm 33
    ( 43.3, 32.7, -26.0),   # right eye outer→ lm 263
    (-28.9,-28.9, -24.1),   # left mouth     → lm 61
    ( 28.9,-28.9, -24.1),   # right mouth    → lm 291
], dtype=np.float64)

dist_coeffs    = np.zeros((4, 1), dtype=np.float64)
camera_matrix  = None

# ── Path to L2CS checkpoint ───────────────────────────────────────────────────
MODEL_PATH = "models/L2CSNet_gaze360.pkl"


# ── Helpers ───────────────────────────────────────────────────────────────────

def pixel_to_3d(u, v, depth, fx, fy, cx, cy):
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return X, Y, Z


def get_iris_center(face_landmarks, iris_indices, w, h):
    points = [
        (int(face_landmarks.landmark[i].x * w),
         int(face_landmarks.landmark[i].y * h))
        for i in iris_indices
    ]
    center = np.mean(points, axis=0).astype(np.int32)
    return center, points


def landmark_to_pixel(face_landmarks, idx, w, h):
    lm = face_landmarks.landmark[idx]
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def get_face_bbox(face_landmarks, w, h, padding: float = 0.15):
    """
    Returns (x1, y1, x2, y2) bounding box from all face landmarks.
    padding: fractional padding added around the tight box.
    """
    xs = [lm.x * w for lm in face_landmarks.landmark]
    ys = [lm.y * h for lm in face_landmarks.landmark]

    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))

    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    return x1, y1, x2, y2


# ── Main worker ───────────────────────────────────────────────────────────────

def eye_tracking_worker(cam_label, q_eye, out_dir):

    global camera_matrix

    print(f"[INFO] Eye tracking (L2CS) started for {cam_label}")

    # ── Output dirs ───────────────────────────────────────────────────────────
    cam_dir = Path(out_dir) / cam_label
    cam_dir.mkdir(parents=True, exist_ok=True)

    logger = get_eye_logger(cam_dir, flush_sec=5)
    logger.info("Eye tracking (L2CS) started")

    eye_dir = Path(out_dir) / cam_label / "eye_tracking"
    eye_dir.mkdir(parents=True, exist_ok=True)

    csv_path = eye_dir / "eye_tracking.csv"
    csv_file = open(csv_path, "w", buffering=1)
    csv_file.write(
        "timestamp_ns,frame_id,"
        "left_iris_x_px,left_iris_y_px,"
        "right_iris_x_px,right_iris_y_px,"
        "X,Y,Z,"
        "smooth_X,smooth_Y,smooth_Z,"
        "head_horizontal,head_vertical,"
        "gaze_direction,dummy,"
        "final_yaw_deg,final_pitch_deg,"
        "head_yaw_deg,head_pitch_deg,"
        "eye_yaw_deg,eye_pitch_deg,"
        "gaze_vec_x,gaze_vec_y,gaze_vec_z\n"
    )
    csv_file.flush()

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # ── L2CS model ────────────────────────────────────────────────────────────
    # use_face_crop=True gives best accuracy (crops face before passing to L2CS)
    gaze_model = L2CSGazeEstimator(
        model_path=MODEL_PATH,
        arch="ResNet50",
        use_face_crop=True,
    )
    print("[INFO] L2CS loaded")

    # ── Smoothing state ───────────────────────────────────────────────────────
    prev_X = prev_Y = prev_Z = None
    alpha = 0.8
    flush_counter = 0

    # ─────────────────────────────────────────────────────────────────────────
    while not stop_event.is_set():

        if pause_event.is_set():
            time.sleep(0.02)
            continue

        try:
            pkt = q_eye.get(timeout=0.1)
        except Exception as e:
            print(f"[QUEUE ERROR] {e}")
            traceback.print_exc()
            continue

        if pkt is None:
            continue

        try:
            color_img  = pkt.color.copy()
            depth_img  = pkt.depth
            fx, fy, cx, cy = pkt.fx, pkt.fy, pkt.cx, pkt.cy
            depth_scale = pkt.depth_scale_m

            if camera_matrix is None:
                camera_matrix = np.array(
                    [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                    dtype=np.float64
                )

            h, w, _ = color_img.shape

            # ── MediaPipe face mesh ───────────────────────────────────────────
            rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                continue

            face_landmarks = results.multi_face_landmarks[0]

            # ── Head pose via PnP ─────────────────────────────────────────────
            image_points = np.array([
                [face_landmarks.landmark[1].x   * w, face_landmarks.landmark[1].y   * h],
                [face_landmarks.landmark[152].x  * w, face_landmarks.landmark[152].y * h],
                [face_landmarks.landmark[33].x   * w, face_landmarks.landmark[33].y  * h],
                [face_landmarks.landmark[263].x  * w, face_landmarks.landmark[263].y * h],
                [face_landmarks.landmark[61].x   * w, face_landmarks.landmark[61].y  * h],
                [face_landmarks.landmark[291].x  * w, face_landmarks.landmark[291].y * h],
            ], dtype=np.float64)

            success, rvec, tvec, _ = cv2.solvePnPRansac(
                FACE_MODEL_POINTS, image_points,
                camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                continue

            R, _ = cv2.Rodrigues(rvec)
            angles, *_ = cv2.RQDecomp3x3(R)
            head_pitch_deg = angles[0]
            head_yaw_deg   = angles[1]

            # Draw head-pose axes on frame
            nose_pt = tuple(image_points[0].astype(int))
            axes_3d = np.float32([[80,0,0],[0,80,0],[0,0,80]])
            imgpts, _ = cv2.projectPoints(
                axes_3d, rvec, tvec, camera_matrix, dist_coeffs
            )
            imgpts = imgpts.reshape(-1, 2).astype(int)
            cv2.line(color_img, nose_pt, tuple(imgpts[0]), (0, 0, 255), 2)
            cv2.line(color_img, nose_pt, tuple(imgpts[1]), (0, 255, 0), 2)
            cv2.line(color_img, nose_pt, tuple(imgpts[2]), (255, 0, 0), 2)

            # ── Iris centers (for HUD/CSV pixel positions) ────────────────────
            left_center,  _ = get_iris_center(face_landmarks, LEFT_IRIS,  w, h)
            right_center, _ = get_iris_center(face_landmarks, RIGHT_IRIS, w, h)
            lx, ly = left_center
            rx, ry = right_center

            eye_x = (lx + rx) // 2
            eye_y = (ly + ry) // 2

            # ── L2CS gaze estimation ──────────────────────────────────────────
            # Compute face bbox from landmarks → pass to L2CS for face-cropped inference
            face_bbox = get_face_bbox(face_landmarks, w, h, padding=0.15)

            gaze_result = gaze_model.estimate_with_vector(color_img, face_bbox)

            if gaze_result is not None:
                eye_yaw_deg, eye_pitch_deg, gaze_vec = gaze_result
            else:
                eye_yaw_deg  = 0.0
                eye_pitch_deg = 0.0
                gaze_vec = np.array([0.0, 0.0, 1.0])

            # Debug print every frame
            print(
                f"[{cam_label}] L2CS  yaw={eye_yaw_deg:+.2f}°  "
                f"pitch={eye_pitch_deg:+.2f}°  "
                f"vec=({gaze_vec[0]:+.3f}, {gaze_vec[1]:+.3f}, {gaze_vec[2]:+.3f})"
            )

            # ── Combined gaze angles ──────────────────────────────────────────
            final_yaw_deg   = head_yaw_deg   + eye_yaw_deg
            final_pitch_deg = head_pitch_deg + eye_pitch_deg

            # ── Depth at eye midpoint ─────────────────────────────────────────
            if (
                eye_x < 0 or eye_y < 0 or
                eye_x >= depth_img.shape[1] or
                eye_y >= depth_img.shape[0]
            ):
                continue

            patch = depth_img[
                max(0, eye_y - 2):min(depth_img.shape[0], eye_y + 3),
                max(0, eye_x - 2):min(depth_img.shape[1], eye_x + 3)
            ]
            valid = patch[patch > 0]
            if len(valid) == 0:
                continue

            depth_raw = np.median(valid)
            if depth_raw == 0:
                continue

            depth_m = float(depth_raw) * float(depth_scale)
            if depth_m < 0.15 or depth_m > 2.0:
                continue

            X, Y, Z = pixel_to_3d(eye_x, eye_y, depth_m, fx, fy, cx, cy)

            # ── Exponential smoothing ─────────────────────────────────────────
            if prev_X is None:
                smooth_X, smooth_Y, smooth_Z = X, Y, Z
            else:
                smooth_X = alpha * prev_X + (1 - alpha) * X
                smooth_Y = alpha * prev_Y + (1 - alpha) * Y
                smooth_Z = alpha * prev_Z + (1 - alpha) * Z

            prev_X, prev_Y, prev_Z = smooth_X, smooth_Y, smooth_Z

            # ── Gaze direction labels ─────────────────────────────────────────
            head_horizontal = (
                "LEFT"  if head_yaw_deg < -15 else
                "RIGHT" if head_yaw_deg >  15 else
                "CENTER"
            )
            head_vertical = (
                "UP"   if head_pitch_deg < -10 else
                "DOWN" if head_pitch_deg >  10 else
                "CENTER"
            )

            h_dir = (
                "LEFT"  if final_yaw_deg < -12 else
                "RIGHT" if final_yaw_deg >  12 else
                "CENTER"
            )
            v_dir = (
                "UP"   if final_pitch_deg < -10 else
                "DOWN" if final_pitch_deg >  10 else
                "CENTER"
            )

            if h_dir == "CENTER" and v_dir == "CENTER":
                gaze_direction = "CENTER"
            elif h_dir == "CENTER":
                gaze_direction = v_dir
            elif v_dir == "CENTER":
                gaze_direction = h_dir
            else:
                gaze_direction = f"{v_dir}_{h_dir}"

            # ── HUD ───────────────────────────────────────────────────────────
            try:
                draw_hud(
                    color_img,
                    smooth_X, smooth_Y, smooth_Z,
                    head_horizontal, head_vertical,
                    gaze_direction,
                    lx, ly,
                    rx, ry,
                    head_yaw_deg, head_pitch_deg,
                    eye_yaw_deg, eye_pitch_deg,
                    final_yaw_deg, final_pitch_deg,
                    horizontal_ratio=eye_yaw_deg,   # was gaze_x
                    vertical_ratio=eye_pitch_deg,    # was gaze_y
                )
            except Exception as e:
                print("[HUD ERROR]", e)
                traceback.print_exc()

            pkt.eye_overlay = color_img

            # ── Save frame every 5 ────────────────────────────────────────────
            if pkt.frame_id % 5 == 0:
                frame_path = eye_dir / f"eye_{pkt.frame_id:06d}.jpg"
                if not cv2.imwrite(str(frame_path), color_img):
                    print(f"[SAVE ERROR] Failed to save {frame_path}")

            # ── Log ───────────────────────────────────────────────────────────
            msg = (
                f"Eye3D => X={smooth_X:.3f} Y={smooth_Y:.3f} Z={smooth_Z:.3f} | "
                f"gaze yaw={eye_yaw_deg:+.1f}° pitch={eye_pitch_deg:+.1f}°"
            )
            print(f"[{cam_label}] {msg}")
            logger.info(msg)
            logger.periodic_flush()

            csv_file.write(
                f"{pkt.t_ns},{pkt.frame_id},"
                f"{lx},{ly},"
                f"{rx},{ry},"
                f"{X:.5f},{Y:.5f},{Z:.5f},"
                f"{smooth_X:.5f},{smooth_Y:.5f},{smooth_Z:.5f},"
                f"{head_horizontal},{head_vertical},"
                f"{gaze_direction},CENTER,"
                f"{final_yaw_deg:.2f},{final_pitch_deg:.2f},"
                f"{head_yaw_deg:.2f},{head_pitch_deg:.2f},"
                f"{eye_yaw_deg:.3f},{eye_pitch_deg:.3f},"
                f"{gaze_vec[0]:.4f},{gaze_vec[1]:.4f},{gaze_vec[2]:.4f}\n"
            )

            flush_counter += 1
            if flush_counter >= 30:
                csv_file.flush()
                flush_counter = 0

        except Exception as e:
            print(f"[EYE TRACK ERROR] {e}")
            traceback.print_exc()
            continue

    # ── Cleanup ───────────────────────────────────────────────────────────────
    csv_file.close()
    face_mesh.close()
    logger.close()

    print(f"[INFO] Eye tracking (L2CS) stopped for {cam_label}")