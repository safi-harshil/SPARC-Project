#!/usr/bin/env python3
# emotion_processor.py — centralized logging
import time
import threading
import queue
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp

from control_flags import pause_event, stop_event
from logger_utils import get_emotion_logger, log_exception
from types_shared import FramePacket
from preview_grid import PreviewGrid
from emotion_mapping import map_to_valence_arousal  # assumed present

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


def emotion_worker(
    cam_label: str,
    q: "queue.Queue[FramePacket]",
    out_dir: Path,
    stride: int,
    csv_flush_every: int,
    log_flush_sec: int,
    history_len: int,
    preview: Optional[PreviewGrid],
):
    """
    Computes valence & arousal per frame using MediaPipe Face Mesh geometry,
    writes CSV, and (optionally) updates the unified preview. Honors pause/stop.
    """
    cam_dir = out_dir / cam_label
    (cam_dir / "CSV").mkdir(parents=True, exist_ok=True)
    (cam_dir / "logs").mkdir(parents=True, exist_ok=True)
    (cam_dir / "color_mp").mkdir(parents=True, exist_ok=True)

    logger = get_emotion_logger(cam_dir, flush_sec=log_flush_sec)
    csv_stream = EmotionCSVStream(cam_dir / "CSV" / "emotion_rt.csv", flush_every=csv_flush_every)
    csv_stream.write_header_if_needed()

    try:
        with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_model:

            logger.info("Emotion worker started")
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

                # pause support
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
                cv2.putText(
                    annotated, f"Valence: {val:+.2f}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
                )
                cv2.putText(
                    annotated, f"Arousal: {aro:.2f}", (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
                )

                if preview is not None:
                    preview.update_emo_frame(annotated)
                    preview.push_valence_arousal(val, aro)

                logger.periodic_flush()

    except Exception as e:
        log_exception(logger, "[FATAL] Emotion worker crashed", e)
    finally:
        csv_stream.close()
        logger.periodic_flush(force=True)
