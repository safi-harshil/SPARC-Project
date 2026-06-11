#!/usr/bin/env python3
"""
test_l2cs_webcam.py
-------------------
Standalone test for L2CS-Net using a regular webcam (no RealSense needed).

Run this FIRST to confirm L2CS is working before integrating into
the full eye_tracking_worker_l2cs.py pipeline.

Usage:
    python test_l2cs_webcam.py

Press Q to quit.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gaze_models.l2cs_gaze import L2CSGazeEstimator


MODEL_PATH = "models/L2CSNet_gaze360.pkl"
DRAW_ARROW = True


def draw_gaze_arrow(img, cx, cy, yaw_deg, pitch_deg, length=80):
    """Draw a gaze direction arrow centered at (cx, cy)."""
    yaw_rad   = np.radians(yaw_deg)
    pitch_rad = np.radians(pitch_deg)

    dx = int(length * np.sin(yaw_rad) * np.cos(pitch_rad))
    dy = int(length * np.sin(pitch_rad))

    end = (cx + dx, cy + dy)
    cv2.arrowedLine(img, (cx, cy), end, (0, 255, 0), 3, tipLength=0.3)


def main():
    estimator = L2CSGazeEstimator(
        model_path=MODEL_PATH,
        arch="ResNet50",
        use_face_crop=False,   # Full frame mode for quick test
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    print("[TEST] Running — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        result = estimator.estimate_with_vector(frame)

        if result is not None:
            yaw_deg, pitch_deg, gaze_vec = result

            label = (
                f"Yaw: {yaw_deg:+.1f}  Pitch: {pitch_deg:+.1f}"
            )
            vec_label = (
                f"Vec: ({gaze_vec[0]:+.3f}, "
                f"{gaze_vec[1]:+.3f}, "
                f"{gaze_vec[2]:+.3f})"
            )

            cv2.putText(
                frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )
            cv2.putText(
                frame, vec_label, (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2
            )

            if DRAW_ARROW:
                draw_gaze_arrow(frame, cx, cy, yaw_deg, pitch_deg)

            # Direction label
            h_dir = "LEFT" if yaw_deg < -10 else "RIGHT" if yaw_deg > 10 else "CENTER"
            v_dir = "UP"   if pitch_deg < -8 else "DOWN" if pitch_deg > 8 else "CENTER"
            cv2.putText(
                frame, f"{v_dir} {h_dir}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2
            )

        else:
            cv2.putText(
                frame, "No face detected", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )

        cv2.imshow("L2CS Gaze Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()