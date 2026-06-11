#!/usr/bin/env python3
# capture_worker.py — robust RealSense capture with warm-up, retries, and auto-reset

import time
import threading
from pathlib import Path
from typing import Optional
import queue as pyqueue

import numpy as np
import cv2
import pyrealsense2 as rs

from tunables import FALLBACK_DEPTH_SCALE
from control_flags import pause_event, stop_event
from logger_utils import DebouncedLogger
from types_shared import FramePacket
from camera_utils import write_camera_info  # assumed present

# Tunables for robustness
STARTUP_DISCARD = 30              # discard first N frames for auto-exposure/convergence
WAIT_TIMEOUT_WARM_MS = 2000       # 2.0 s during warm-up
WAIT_TIMEOUT_STEADY_MS = 1200     # 1.2 s during steady-state
MAX_TIMEOUTS_BEFORE_RESET = 60    # consecutive timeouts before we reset pipeline
WARN_EVERY_TIMEOUTS = 5

# Keep a shared notion of cam2's depth scale for cross-cam consistency
_CAM2_DEPTH_SCALE_LOCK = threading.Lock()
_CAM2_DEPTH_SCALE: Optional[float] = None  # learned during run from cam2


def _safe_put(q: Optional["pyqueue.Queue"], pkt: FramePacket, backpressure: str):
    if q is None:
        return
    try:
        if backpressure == "block":
            q.put(pkt, timeout=0.01)
        else:
            q.put_nowait(pkt)
    except pyqueue.Full:
        # drop-then-insert (keeps freshest)
        try:
            _ = q.get_nowait()
        except Exception:
            pass
        try:
            q.put_nowait(pkt)
        except Exception:
            pass


def capture_worker(
    serial: str,
    cam_label: str,
    out_dir: Path,
    duration_sec: float,
    save_every: int,
    filters_on: bool,
    q_mov: Optional["pyqueue.Queue[FramePacket]"],
    q_emo: Optional["pyqueue.Queue[FramePacket]"],
    backpressure: str,
    q_obj: Optional["pyqueue.Queue[FramePacket]"] = None,
    q_eye: Optional["pyqueue.Queue[FramePacket]"] = None,  # ✅ NEW (optional)
):
    """
    Captures color+depth from a RealSense, aligns depth to color, optionally filters,
    publishes FramePacket(s) to movement / emotion / object queues, and optionally saves raw frames.
    Honors global pause/stop (SPACE / ESC) via control_flags.
    """

    cam_dir = out_dir / cam_label
    color_dir = cam_dir / "color"
    depth_dir = cam_dir / "depth"
    (cam_dir / "logs").mkdir(parents=True, exist_ok=True)
    color_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    logger = DebouncedLogger(cam_dir / "logs" / "capture_rt.log", flush_interval_sec=5)

    # Build pipeline & config
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    def _start_pipeline() -> Optional[rs.pipeline_profile]:
        try:
            return pipeline.start(cfg)
        except Exception as e:
            logger.warn(f"Failed to start RealSense {cam_label} ({serial}): {e}")
            logger.periodic_flush()
            return None

    profile = _start_pipeline()
    if profile is None:
        print(f"[ERROR] Failed to start RealSense {cam_label} ({serial}).")
        return

    align = rs.align(rs.stream.color)
    if filters_on:
        spatial = rs.spatial_filter()
        temporal = rs.temporal_filter()
        hole = rs.hole_filling_filter()
    else:
        spatial = temporal = hole = None

    # Write camera info & intrinsics json (returns a scale; we may override below)
    info_txt = cam_dir / f"camera_info_{serial}.txt"
    info_json = cam_dir / f"camera_intrinsics_{serial}.json"
    intr_json, depth_scale_m = write_camera_info(profile, serial, cam_label, info_txt, info_json)

    # Prefer cam2's scale for others if available; ensure fallback constant
    with _CAM2_DEPTH_SCALE_LOCK:
        global _CAM2_DEPTH_SCALE
        if cam_label == "cam2":
            # If cam2's reported scale is invalid, fall back and publish the fallback
            if (depth_scale_m is None) or (depth_scale_m <= 0):
                depth_scale_m = FALLBACK_DEPTH_SCALE
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
        fx, fy, cx, cy = 604.7, 604.9, 313.8, 252.7  # conservative fallback
    else:
        fx = float(intr_json[color_key]["fx"])
        fy = float(intr_json[color_key]["fy"])
        cx = float(intr_json[color_key]["cx"])
        cy = float(intr_json[color_key]["cy"])

    print(
        f"[INFO] 🎥 {cam_label} → depth_scale={depth_scale_m:.12f} m/unit | "
        f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}"
    )

    frame_id = 0
    active_elapsed = 0.0
    last_tick = time.monotonic()

    timeouts_in_row = 0
    discarded = 0
    warmup = True

    try:
        while not stop_event.is_set() and active_elapsed < duration_sec:
            # Pause support (SPACE)
            if pause_event.is_set():
                last_tick = time.monotonic()
                time.sleep(0.02)
                continue

            # Try fast non-blocking fetch first
            fs = None
            try:
                fs = pipeline.poll_for_frames()
                if not fs:
                    # fall back to blocking with timeout (warm vs steady)
                    timeout_ms = WAIT_TIMEOUT_WARM_MS if warmup else WAIT_TIMEOUT_STEADY_MS
                    try:
                        fs = pipeline.wait_for_frames(timeout_ms)
                    except Exception as _:
                        fs = None
                # Handle no-frames case as timeout
                if not fs:
                    timeouts_in_row += 1
                    if timeouts_in_row % WARN_EVERY_TIMEOUTS == 0:
                        logger.warn(
                            f"{cam_label}: no frames for {timeouts_in_row} consecutive attempts "
                            f"(warmup={warmup}, timeout_ms={WAIT_TIMEOUT_WARM_MS if warmup else WAIT_TIMEOUT_STEADY_MS})"
                        )
                    # Don’t charge active time when we starve
                    last_tick = time.monotonic()
                    # Auto-reset if starved for long
                    if timeouts_in_row >= MAX_TIMEOUTS_BEFORE_RESET:
                        logger.warn(f"{cam_label}: resetting pipeline after {timeouts_in_row} timeouts.")
                        try:
                            pipeline.stop()
                        except Exception:
                            pass
                        time.sleep(0.2)
                        profile = _start_pipeline()
                        if profile is None:
                            logger.warn(f"{cam_label}: restart failed; will keep retrying.")
                            time.sleep(0.3)
                        else:
                            logger.info(f"{cam_label}: pipeline restarted.")
                            timeouts_in_row = 0
                            discarded = 0
                            warmup = True
                    time.sleep(0.005)
                    continue  # retry loop
                else:
                    timeouts_in_row = 0
            except Exception as e:
                # Unexpected device error: log and try to continue
                logger.warn(f"{cam_label}: exception fetching frames: {e}")
                last_tick = time.monotonic()
                time.sleep(0.01)
                continue

            # Align & fetch frames
            try:
                aligned = align.process(fs)
                c = aligned.get_color_frame()
                d = aligned.get_depth_frame()
                if not c or not d:
                    # treat as soft miss
                    continue
            except Exception:
                continue

            if filters_on:
                try:
                    d = spatial.process(d)
                    d = temporal.process(d)
                    d = hole.process(d)
                except Exception:
                    pass

            # Warm-up discard
            if warmup:
                discarded += 1
                if discarded < STARTUP_DISCARD:
                    # keep UI responsive, don’t count active time
                    last_tick = time.monotonic()
                    continue
                else:
                    warmup = False
                    logger.info(f"{cam_label}: warm-up completed after {discarded} frames.")

            color_img = np.asanyarray(c.get_data())
            depth_img = np.asanyarray(d.get_data())

            # Save raw
            if save_every > 0 and (frame_id % save_every == 0):
                try:
                    cv2.imwrite(str(color_dir / f"frame_{frame_id:06d}.png"), color_img)
                    cv2.imwrite(
                        str(depth_dir / f"frame_{frame_id:06d}.tiff"),
                        depth_img,
                        [cv2.IMWRITE_TIFF_COMPRESSION, 1],
                    )
                except Exception as e:
                    logger.warn(f"{cam_label}: failed to save frame {frame_id}: {e}")

            # Publish to processing pipelines (if enabled for this cam)
            pkt = FramePacket(
                cam_label=cam_label,
                frame_id=frame_id,
                t_ns=time.monotonic_ns(),
                rs_ts_ms=c.get_timestamp() if hasattr(c, "get_timestamp") else 0.0,
                color=color_img,
                depth=depth_img,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                depth_scale_m=depth_scale_m,
            )
            _safe_put(q_mov, pkt, backpressure)
            _safe_put(q_emo, pkt, backpressure)
            _safe_put(q_obj, pkt, backpressure)  # ✅ NEW: feed object lane
            _safe_put(q_eye, pkt, backpressure)  # ✅ NEW: feed eye lane

            frame_id += 1

            now = time.monotonic()
            active_elapsed += (now - last_tick)
            last_tick = now

            if stop_event.is_set():
                break

    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        logger.info(
            f"Finished capture {cam_label}: frames={frame_id}, active_elapsed={active_elapsed:.2f}s, "
            f"warmup_discarded={discarded}, last_timeouts={timeouts_in_row}"
        )
        logger.periodic_flush(force=True)
