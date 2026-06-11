#!/usr/bin/env python3

import pyrealsense2 as rs
import cv2
import threading
import numpy as np
import os

CAMERA_SERIALS_FILE = "camera_serials.txt"
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_camera_serials():
    serial_map = {}
    if not os.path.exists(CAMERA_SERIALS_FILE):
        print(f"[ERROR] File '{CAMERA_SERIALS_FILE}' not found.")
        return serial_map

    with open(CAMERA_SERIALS_FILE, "r") as f:
        for line in f:
            if ":" in line:
                label, serial = line.strip().split(":")
                serial_map[serial.strip()] = label.strip()
    return serial_map


class CameraStream:
    def __init__(self, serial, label):
        self.serial = serial
        self.label = label  # e.g., "cam1", "cam2"

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial)
        # Enable BOTH color and depth streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Align depth to color
        self.align = rs.align(rs.stream.color)
        self.colorizer = rs.colorizer()

        # Frame buffers
        self.color_frames = []
        self.depth_frames = []

        self.last_ts = None
        self.fps = 0.0
        self.dropped = 0

        self.lock = threading.Lock()
        self.running = False          # controls main loop in update()
        self._stopping = False        # tells update() we're shutting down

        # View mode: False = RGB, True = depth
        self.show_depth = False

    def start(self) -> bool:
        """Start the RealSense pipeline. Returns True on success."""
        try:
            self.pipeline.start(self.config)
            self.running = True
            self._stopping = False
            return True
        except Exception as e:
            print(f"[ERROR] {self.label}: failed to start pipeline: {e}")
            self.running = False
            self._stopping = False
            return False

    def stop(self):
        """Request clean shutdown of this camera stream."""
        if not self.running and not self._stopping:
            return

        # Tell update() we’re shutting down so it can swallow expected errors
        self._stopping = True
        self.running = False

        try:
            self.pipeline.stop()
        except Exception as e:
            # Only warn if something truly unexpected happens here
            print(f"[WARN] {self.label}: error while stopping pipeline: {e}")

    def toggle_view(self):
        """Toggle between RGB and depth view for this camera."""
        self.show_depth = not self.show_depth

    def update(self):
        """Background thread: grab frames and update FPS/dropped counts."""
        while self.running:
            try:
                frameset = self.pipeline.wait_for_frames()
                # Align depth to color
                aligned_frames = self.align.process(frameset)
                color_frame = aligned_frames.get_color_frame()
                depth_frame = aligned_frames.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                ts = color_frame.get_timestamp()
                color_img = np.asanyarray(color_frame.get_data())
                # Colorize depth for visualization
                depth_color_frame = self.colorizer.colorize(depth_frame)
                depth_img = np.asanyarray(depth_color_frame.get_data())

                with self.lock:
                    self.color_frames.append(color_img)
                    self.depth_frames.append(depth_img)

                    # Keep only recent frames
                    if len(self.color_frames) > 5:
                        self.color_frames = self.color_frames[-5:]
                    if len(self.depth_frames) > 5:
                        self.depth_frames = self.depth_frames[-5:]

                    if self.last_ts is not None:
                        time_diff = ts - self.last_ts
                        expected_diff = 1000.0 / 30.0
                        if time_diff > expected_diff * 1.5:
                            self.dropped += 1
                        self.fps = 1000.0 / time_diff if time_diff > 0 else 0.0

                    self.last_ts = ts

            except Exception as e:
                # If we're stopping, this error is expected (pipeline.stop() wakes wait_for_frames)
                if not self._stopping:
                    print(f"[ERROR] {self.label}: {e}")
                break

    def get_latest_frame(self):
        """Return latest frame according to current view mode (RGB/depth)."""
        with self.lock:
            if self.show_depth:
                if self.depth_frames:
                    return self.depth_frames[-1].copy()
            else:
                if self.color_frames:
                    return self.color_frames[-1].copy()

            # Fallback if no frames yet
            return np.zeros((480, 640, 3), dtype=np.uint8)


def draw_overlay(img, label, fps, dropped, show_depth):
    overlay = img.copy()
    mode = "DEPTH" if show_depth else "RGB"
    cv2.putText(overlay, f"{label} [{mode}]", (10, 25), FONT, 0.8, (0, 255, 255), 2)
    cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 50), FONT, 0.7, (255, 255, 255), 1)
    cv2.putText(overlay, f"Dropped: {dropped}", (10, 75), FONT, 0.7, (0, 0, 255), 1)
    return overlay


def create_preview_grid(cams, cols=2):
    overlays = []
    for cam in cams:
        frame = cam.get_latest_frame()
        overlays.append(draw_overlay(frame, cam.label, cam.fps, cam.dropped, cam.show_depth))

    if not overlays:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    while len(overlays) % cols != 0:
        overlays.append(np.zeros_like(overlays[0]))

    rows = len(overlays) // cols
    grid_rows = [np.hstack(overlays[i * cols:(i + 1) * cols]) for i in range(rows)]
    grid = np.vstack(grid_rows)
    return grid


def main():
    serial_map = load_camera_serials()
    if not serial_map:
        print("[ERROR] No known camera serials found. Check 'camera_serials.txt'.")
        return

    ctx = rs.context()
    connected_serials = [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]

    print(f"[INFO] Connected RealSense devices: {connected_serials}")

    cameras = []
    threads = []

    # Create and start streams only for known cameras
    for serial in connected_serials:
        if serial in serial_map:
            cam = CameraStream(serial, serial_map[serial])
            if not cam.start():
                # Failed to start; skip this camera
                continue
            t = threading.Thread(target=cam.update, daemon=True)
            t.start()
            cameras.append(cam)
            threads.append(t)

    if not cameras:
        print("[ERROR] No known cameras connected (or none started successfully).")
        return

    print("[INFO] Controls:")
    print("  ESC  → exit preview")
    print("  1..9 → toggle RGB/DEPTH for corresponding grid slot")

    try:
        while True:
            grid = create_preview_grid(cameras, cols=2)
            cv2.imshow("RealSense Grid View", grid)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("[INFO] ESC pressed. Exiting preview...")
                break

            # Number keys 1..9 map to cameras[0..8]
            if ord('1') <= key <= ord('9'):
                idx = key - ord('1')
                if 0 <= idx < len(cameras):
                    cameras[idx].toggle_view()
                    print(f"[INFO] Toggled view for {cameras[idx].label} "
                          f"→ {'DEPTH' if cameras[idx].show_depth else 'RGB'}")

    except KeyboardInterrupt:
        print("\n[INFO] Preview interrupted by user (Ctrl+C).")
    finally:
        # Clean shutdown of all camera streams
        for cam in cameras:
            cam.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
