#!/usr/bin/env python3
# preview_grid.py — robust UI loop & clean shutdown + centralized logging
import time
import sys
from collections import deque
from pathlib import Path
import numpy as np
import cv2
from tunables import DEFAULT_CELL_W, DEFAULT_CELL_H, PLOT_UPDATE_INTERVAL
from logger_utils import DebouncedLogger, get_preview_logger

class PreviewGrid:
    def __init__(
        self,
        title="Unified Preview",
        history_len=600,
        target_fps=30,
        logger: DebouncedLogger | None = None,
        total_duration_sec: float | None = None,
        enable_event_overlay: bool = True,     # NEW (defaults preserve old behavior)
        enable_object_overlay: bool = True,    # NEW (defaults preserve old behavior)
        show_cv2_window: bool = True,
    ):
        self.title = title
        self.history_len = int(max(120, history_len))
        self.target_dt = 1.0 / float(max(5, target_fps))
        import threading
        self.lock = threading.Lock()
        self._run_flag = threading.Event()
        self._is_open = threading.Event()
        self._thread = None
        self.cell_h, self.cell_w = DEFAULT_CELL_H, DEFAULT_CELL_W
        self.hand_frame = None
        self.emo_frame = None

        # Movement (cumulative) history + timestamps
        self.r0_cum_hist = deque(maxlen=self.history_len)
        self.r0_time_hist = deque(maxlen=self.history_len)
        self._t0_plot = None
        self.r0_last_avg_speed = 0.0

        # Affect history
        self.va_hist = deque(maxlen=self.history_len)

        self.logger = logger or get_preview_logger(Path("."))

        # Total duration to lock X-axis
        self.total_duration_sec = (
            self._guess_total_duration_from_argv()
            if total_duration_sec is None
            else float(max(1.0, total_duration_sec))
        )

        # Short-lived badges and persistent event marks
        self._active_events = []
        self._event_marks = []

        # per-frame object state and masks (for overlay)
        self._obj_states = {"untouched": {}, "checking": {}}
        self._obj_masks = {}       # {obj: np.ndarray mask (binary 0/255), ROI-sized or full-frame}
        self._obj_crop_box = None  # (x_min, y_min, x_max, y_max) in movement-frame coords

        # Qt notes UI pull (latest grid + Mode-2 timeline)
        self._last_grid = None
        self._grid_frame_idx = 0
        self._timeline_t0 = time.monotonic()
        self._paused_accum = 0.0
        self._pause_started = None

        # NEW: overlay toggles (no behavior change unless caller disables)
        self.enable_event_overlay = bool(enable_event_overlay)
        self.enable_object_overlay = bool(enable_object_overlay)

        self.show_cv2_window = bool(show_cv2_window)

    def _guess_total_duration_from_argv(self, default: float = 60.0) -> float:
        try:
            args = sys.argv
            if "--duration-sec" in args:
                i = args.index("--duration-sec")
                if i + 1 < len(args):
                    return float(args[i + 1])
        except Exception:
            pass
        return float(default)

    # ---------- external ----------
    def start(self):
        if self._thread:
            return
        self._run_flag.set()
        self._is_open.set()
        import threading
        self._thread = threading.Thread(target=self._loop, daemon=True, name="preview-grid")
        self._thread.start()
        try:
            self.logger.info("Preview window started")
        finally:
            self.logger.periodic_flush()

    def reopen_window(self):
        self._is_open.set()
        try:
            self.logger.info("Preview window reopened")
        finally:
            self.logger.periodic_flush()

    def close_window(self):
        self._is_open.clear()
        try:
            cv2.destroyWindow(self.title)
        except Exception:
            pass
        try:
            self.logger.info("Preview window closed")
        finally:
            self.logger.periodic_flush()

    def stop(self):
        self._run_flag.clear()
        self._is_open.clear()
        try:
            cv2.destroyWindow(self.title)
        except Exception:
            pass
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
        self._thread = None
        try:
            self.logger.info("Preview window stopped")
        finally:
            self.logger.periodic_flush(force=True)

    def set_total_duration(self, duration_sec: float):
        with self.lock:
            self.total_duration_sec = float(max(1.0, duration_sec))

    def update_hand_frame(self, img):
        with self.lock:
            self.hand_frame = img.copy()
            self._maybe_set_cell_size(img)

    def update_emo_frame(self, img):
        with self.lock:
            self.emo_frame = img.copy()
            self._maybe_set_cell_size(img)

    def push_r0_cumulative(self, cum, avg):
        now = time.monotonic()
        with self.lock:
            if self._t0_plot is None:
                self._t0_plot = now
            self.r0_cum_hist.append(float(cum))
            self.r0_time_hist.append(now)
            self.r0_last_avg_speed = float(avg)

    def push_valence_arousal(self, val, aro):
        with self.lock:
            self.va_hist.append((float(val), float(aro)))

    # --- event overlays + persistent marks ---
    def annotate_event(self, cam_label: str, kind: str, text: str, ttl_s: float = 6.0):
        if not self.enable_event_overlay:
            return
        now = time.monotonic()
        exp = now + max(0.5, float(ttl_s))
        with self.lock:
            self._active_events = [e for e in self._active_events if e["expires"] > now]
            self._active_events.append({"text": str(text), "kind": str(kind), "expires": exp})
            self._event_marks.append({"t_abs": now, "kind": str(kind), "text": str(text)})

    # --- called by object worker each frame ---
    def update_objects(self, states: dict, masks: dict | None = None, crop_box: tuple | None = None):
        """states = {'untouched': {obj: bool}, 'checking': {obj: bool}}
           masks  = {obj: mask (uint8), ROI-sized or full-frame}
           crop_box = (x_min, y_min, x_max, y_max) in movement-frame coords (optional)"""
        with self.lock:
            self._obj_states = {
                "untouched": dict(states.get("untouched", {})),
                "checking": dict(states.get("checking", {})),
            }
            if masks:
                self._obj_masks = {k: v.copy() for k, v in masks.items() if v is not None}
            else:
                self._obj_masks.clear()
            self._obj_crop_box = tuple(crop_box) if crop_box is not None else None

    # --- compatibility wrapper for workers that send a single dict ---
    def update_object_state(self, cam_label: str, obj_state: dict):
        """
        Accepts the unified dict returned by ObjectInteraction.ingest_frame():
            {
              'untouched': {...}, 'checking': {...},
              'masks': {...}, 'crop_box': (x_min,y_min,x_max,y_max)
            }
        or (older): {'object_state': {...}} wrapping the same keys.
        """
        if not obj_state:
            return
        payload = obj_state.get("object_state", obj_state)
        states = {
            "untouched": payload.get("untouched", payload.get("untouched_out", {})),
            "checking":  payload.get("checking",  payload.get("checking_out",  {})),
        }
        masks = payload.get("masks")
        crop  = payload.get("crop_box")
        self.update_objects(states, masks=masks, crop_box=crop)

    # ── Qt notes UI pull helpers ─────────────────────────────────────────
    def get_latest_grid(self):
        """Return latest composed grid frame (BGR uint8) for Qt UI to display."""
        with self.lock:
            return None if self._last_grid is None else self._last_grid.copy()

    def get_timeline_seconds(self) -> int:
        """
        Playback timeline:
        - advances while pause_event is NOT set
        - freezes while pause_event IS set
        """
        from control_flags import pause_event
        now = time.monotonic()

        # latch pause start
        if pause_event.is_set():
            if self._pause_started is None:
                self._pause_started = now
        else:
            # on resume: commit paused interval into accumulator
            if self._pause_started is not None:
                self._paused_accum += (now - self._pause_started)
                self._pause_started = None

        # IMPORTANT: while paused, also subtract the current pause duration
        paused_live = 0.0
        if pause_event.is_set() and self._pause_started is not None:
            paused_live = now - self._pause_started

        t = (now - self._timeline_t0) - (self._paused_accum + paused_live)
        if t < 0:
            t = 0.0
        return int(t)

    def _store_grid(self, grid: np.ndarray):
        with self.lock:
            self._last_grid = grid
            self._grid_frame_idx += 1

    def _maybe_set_cell_size(self, img):
        h, w = img.shape[:2]
        maxw = 640
        scale = min(1.0, maxw / float(w)) if w > 0 else 1.0
        self.cell_w, self.cell_h = int(w * scale), int(h * scale)

    # ---------- object legend + mask overlay ----------
    def _overlay_object_masks(self, img_full):
        """Overlay semi-transparent colored object masks on the ORIGINAL movement frame (pre-resize),
        byte-aligned with the renderer behavior."""
        if img_full is None:
            return
        if not self.enable_object_overlay:
            return
        with self.lock:
            masks = dict(self._obj_masks)
            crop_box = self._obj_crop_box
        if not masks:
            return

        # BGR colors exactly as renderer
        colors = {
            "red":      (0, 0, 255),
            "green":    (0, 255, 0),
            "gray":     (255, 255, 255),
            "yellow_1": (0, 255, 255),
            "yellow_2": (0, 128, 255),
            "gold":     (128, 0, 255),
        }
        objects = list(colors.keys())

        if crop_box:
            x_min, y_min, x_max, y_max = crop_box
            crop_w, crop_h = x_max - x_min, y_max - y_min

            for obj in objects:
                m = masks.get(obj)
                if m is None:
                    continue

                # If mask shape mismatches crop size, attempt to fallback crop or skip
                if m.shape[:2] != (crop_h, crop_w):
                    # If mask is full-frame we can crop
                    if m.shape[0] >= y_max and m.shape[1] >= x_max:
                        m = m[y_min:y_max, x_min:x_max]
                    else:
                        continue

                color = colors.get(obj, (255, 255, 255))

                colored_roi = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
                colored_roi[m > 0] = color

                roi = img_full[y_min:y_max, x_min:x_max]
                blended = cv2.addWeighted(roi, 1.0, colored_roi, 0.4, 0)

                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    cv2.drawContours(blended, [c], -1, color, thickness=2)

                img_full[y_min:y_max, x_min:x_max] = blended

        else:
            H, W = img_full.shape[:2]
            for obj in objects:
                m = masks.get(obj)
                if m is None:
                    continue
                if m.shape[:2] != (H, W):
                    continue

                color = colors.get(obj, (255, 255, 255))
                colored = np.zeros((H, W, 3), dtype=np.uint8)
                colored[m > 0] = color

                blended = cv2.addWeighted(img_full, 1.0, colored, 0.4, 0)

                contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    cv2.drawContours(blended, [c], -1, color, thickness=2)

                img_full[:] = blended

    def _draw_transparent_circle(self, img_bgr, center, radius, color_bgr, alpha=0.45, outline=True):
        overlay = img_bgr.copy()
        cv2.circle(overlay, center, radius, color_bgr, thickness=-1)
        cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0, dst=img_bgr)
        if outline:
            cv2.circle(img_bgr, center, radius, color_bgr, 1)

    def _render_object_legend(self, img):
        """Overlay vertical legend of per-object states on the movement image."""
        if img is None:
            return
        if not self.enable_object_overlay:
            return
        order = ["red", "green", "gray", "yellow_1", "yellow_2", "gold"]
        colors = {
            "red": (0, 0, 255),
            "green": (0, 255, 0),
            "gray": (255, 255, 255),
            "yellow_1": (0, 255, 255),
            "yellow_2": (0, 128, 255),
            "gold": (128, 0, 255),
        }
        with self.lock:
            unt = dict(self._obj_states.get("untouched", {}))
            chk = dict(self._obj_states.get("checking", {}))
        radius = 8
        start_x = 20
        start_y = 60
        vspacing = 28
        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, obj in enumerate(order):
            cx = start_x
            cy = start_y + i * vspacing
            color = colors.get(obj, (255, 255, 255))
            is_check = bool(chk.get(obj, False))
            is_un = bool(unt.get(obj, False))
            if is_check:
                self._draw_transparent_circle(img, (cx, cy), radius, color, alpha=0.45, outline=True)
            elif is_un:
                cv2.circle(img, (cx, cy), radius, color, -1)
            else:
                cv2.circle(img, (cx, cy), radius, color, 2)
            cv2.putText(img, obj, (cx + radius + 8, cy + 5), font, 0.5, color, 1, cv2.LINE_AA)

    # ---------- badges ----------
    def _badge(self, img, text, corner="tl", pad=8, opacity=0.4):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        bw, bh = tw + 18, th + 14
        H, W = img.shape[:2]
        if corner == "tl":
            x0, y0 = pad, pad
        elif corner == "tr":
            x0, y0 = W - bw - pad, pad
        elif corner == "bl":
            x0, y0 = pad, H - bh - pad
        else:
            x0, y0 = W - bw - pad, H - bh - pad
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + bw, y0 + bh), (0, 0, 0), -1)
        cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
        cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (180, 180, 180), 1)
        cv2.putText(img, text, (x0 + 9, y0 + bh - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    # ---------- cumulative plot ----------
    def _render_plot_line(self, values, label, avg_speed=0.0):
        W, H = self.cell_w, self.cell_h
        canvas = np.zeros((H, W, 3), np.uint8)

        x0, y0, x1, y1 = 60, 30, W - 30, H - 50
        width, height = max(1, x1 - x0), max(1, y1 - y0)
        cx = x0 + width // 2

        cv2.rectangle(canvas, (x0, y0), (x1, y1), (80, 80, 80), 1)
        cv2.line(canvas, (x0, y1), (x1, y1), (100, 100, 100), 1)    # X
        cv2.line(canvas, (x0, y0), (x0, y1), (100, 100, 100), 1)    # Y
        self._badge(canvas, label, "tr")

        vals = list(values)
        times = list(self.r0_time_hist)
        if not vals:
            return canvas

        if times and len(vals) > len(times):
            vals = vals[-len(times):]
        if len(times) > len(vals):
            times = times[-len(vals):]

        if vals and vals[0] != 0.0:
            vals = [0.0] + vals
            if times:
                times = [times[0]] + times

        if self._t0_plot is None:
            self._t0_plot = times[0] if times else time.monotonic()
        t0 = self._t0_plot
        t_last = times[-1] if times else t0
        elapsed = max(0.0, t_last - t0)
        total_T = max(1e-9, self.total_duration_sec)
        elapsed = min(elapsed, total_T)

        x_tip = int(cx + (elapsed / total_T) * (x1 - cx))

        current = float(vals[-1])
        target_ratio_y = 0.8 + 0.2 * (elapsed / total_T)
        target_ratio_y = float(np.clip(target_ratio_y, 0.8, 1.0))
        eps = 1e-6
        if current <= eps:
            y_max = 1.0
        else:
            y_max = max(current / target_ratio_y, eps)

        def map_y(v: float) -> int:
            clamped = np.clip(v / y_max, 0.0, 1.0)
            return int(y1 - clamped * height)

        def map_x_t(t_sec: float) -> int:
            if elapsed <= 1e-9:
                return x_tip
            frac = np.clip(t_sec / elapsed, 0.0, 1.0)
            return int(x0 + frac * (x_tip - x0))

        prev = None
        for v, t_abs in zip(vals, times):
            t_rel = max(0.0, t_abs - t0)
            x = map_x_t(t_rel)
            y = map_y(v)
            if prev is not None:
                cv2.line(canvas, prev, (x, y), (0, 255, 255), 2)
            prev = (x, y)

        tip_y = map_y(current)
        cv2.circle(canvas, (x_tip, tip_y), 3, (255, 255, 255), -1)
        cv2.putText(canvas, f"avg:{avg_speed:.2f} mm/s", (x_tip + 10, tip_y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        tick_y = y1 + 18
        for t_val, label_txt in ((0.0, "0"), (elapsed, f"{elapsed:.0f}s"), (total_T, f"{total_T:.0f}s")):
            if t_val <= elapsed:
                x_tick = map_x_t(t_val)
            else:
                rem = (t_val - elapsed) / max(1e-9, (total_T - elapsed))
                x_tick = int(x_tip + rem * (x1 - x_tip))
            x_tick = int(np.clip(x_tick, x0, x1))
            cv2.line(canvas, (x_tick, y1), (x_tick, y1 + 6), (130, 130, 130), 1)
            cv2.putText(canvas, label_txt, (x_tick - 10, tick_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        for v_val, label_txt in ((0.0, "0"), (current, f"{current:.0f}")):
            y = map_y(v_val)
            cv2.line(canvas, (x0 - 6, y), (x0, y), (130, 130, 130), 1)
            cv2.putText(canvas, label_txt, (x0 - 50, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Event triangles ONLY if enabled
        if self.enable_event_overlay:
            with self.lock:
                marks = list(self._event_marks)
            tri = 7
            for m in marks:
                t_e = float(np.clip(m["t_abs"] - t0, 0.0, elapsed))
                if times:
                    diffs = np.abs(np.array(times) - (t0 + t_e))
                    idx = int(np.argmin(diffs))
                    v_e = float(vals[idx])
                else:
                    v_e = current
                mx = map_x_t(t_e)
                my = map_y(v_e)
                txt = m.get("text", "")
                if "high" in txt.lower():
                    pts = np.array([[mx, my - tri], [mx - tri, my + tri], [mx + tri, my + tri]], np.int32)
                    color = (0, 0, 255)
                else:
                    pts = np.array([[mx, my + tri], [mx - tri, my - tri], [mx + tri, my - tri]], np.int32)
                    color = (255, 255, 0)
                cv2.fillConvexPoly(canvas, pts, color)

        return canvas

    # ---------- valence/arousal ----------
    def _render_va(self, tuples_va):
        W, H = self.cell_w, self.cell_h
        canvas = np.zeros((H, W, 3), np.uint8)
        x0, y0, x1, y1 = 60, 30, W - 30, H - 50
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (80, 80, 80), 1)
        cv2.line(canvas, (x0, y1), (x1, y1), (100, 100, 100), 1)
        cv2.line(canvas, (x0, y0), (x0, y1), (100, 100, 100), 1)
        self._badge(canvas, "Valence (Green)  Arousal (Orange)", "tr")
        if not tuples_va:
            return canvas

        def vx(i, n):
            den = max(1, n - 1)
            return x0 + int(i * (x1 - x0 - 1) / den)

        def vy_val(v):
            return int(y1 - ((v + 1.0) * 0.5) * (y1 - y0))

        def vy_aro(a):
            return int(y1 - (a) * (y1 - y0))

        prev_v = prev_a = None
        n = len(tuples_va)
        for i, (v, a) in enumerate(tuples_va):
            x = vx(i, n)
            yv, ya = vy_val(v), vy_aro(a)
            if prev_v:
                cv2.line(canvas, prev_v, (x, yv), (0, 255, 0), 2)
            if prev_a:
                cv2.line(canvas, prev_a, (x, ya), (0, 165, 255), 2)
            prev_v, prev_a = (x, yv), (x, ya)
        return canvas

    # ---------- compose ----------
    def _compose_grid(self, hand, emo, mov_plot, va_plot):
        def fit(img):
            if img is None:
                return np.zeros((self.cell_h, self.cell_w, 3), np.uint8)
            return cv2.resize(img, (self.cell_w, self.cell_h))

        # IMPORTANT: object overlay only if enabled
        hand_proc = None if hand is None else hand.copy()
        if hand_proc is not None and self.enable_object_overlay:
            self._overlay_object_masks(hand_proc)
            self._render_object_legend(hand_proc)

        tl = fit(hand_proc) if hand_proc is not None else fit(hand)
        tr = fit(emo)
        bl = fit(mov_plot)
        br = fit(va_plot)

        self._badge(tl, "Movement Camera", "tr")
        self._badge(tr, "Emotion Camera", "tr")
        self._badge(bl, "Cumulative R_0", "bl")
        self._badge(br, "Affect Traces", "br")

        grid = np.zeros((self.cell_h * 2, self.cell_w * 2, 3), np.uint8)
        grid[0:self.cell_h, 0:self.cell_w] = tl
        grid[0:self.cell_h, self.cell_w:] = tr
        grid[self.cell_h:, 0:self.cell_w] = bl
        grid[self.cell_h:, self.cell_w:] = br

        # Event badge only if enabled
        if self.enable_event_overlay:
            now = time.monotonic()
            with self.lock:
                self._active_events = [e for e in self._active_events if e["expires"] > now]
                events = list(self._active_events)
            if events:
                e = events[-1]
                text = e["text"]
                x_off, y_off = 0, self.cell_h
                overlay = grid.copy()
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
                bw, bh = tw + 36, th + 22
                x0, y0 = x_off + 10, y_off + 10
                cv2.rectangle(overlay, (x0, y0), (x0 + bw, y0 + bh), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.45, grid, 1 - 0.45, 0, grid)
                cv2.rectangle(grid, (x0, y0), (x0 + bw, y0 + bh), (200, 200, 200), 1)
                sym_c = (0, 0, 255) if "high" in text.lower() else (255, 255, 0)
                cv2.circle(grid, (x0 + 14, y0 + bh // 2), 7, sym_c, -1)
                cv2.putText(grid, text, (x0 + 28, y0 + bh - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        return grid

    # ---------- loop ----------
    def _loop(self):
        from control_flags import stop_event
        stall_t0 = time.monotonic()
        while self._run_flag.is_set():
            try:
                if self._is_open.is_set():
                    with self.lock:
                        hand = self.hand_frame.copy() if self.hand_frame is not None else None
                        emo = self.emo_frame.copy() if self.emo_frame is not None else None
                        r0 = list(self.r0_cum_hist)
                        avg = self.r0_last_avg_speed
                        va = list(self.va_hist)

                    mov = self._render_plot_line(r0, "R_0 cumulative (mm)", avg)
                    va_plot = self._render_va(va)
                    grid = self._compose_grid(hand, emo, mov, va_plot)

                    # store latest frame for Qt Notes UI
                    self._store_grid(grid)

                    if self.show_cv2_window:
                        cv2.imshow(self.title, grid)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            self.close_window()
                        elif key == 27:
                            print("[⛔] ESC (window) → stopping.")
                            stop_event.set()
                    else:
                        # still pump events lightly so OpenCV doesn't choke if imported elsewhere
                        cv2.waitKey(1)

                    stall_t0 = time.monotonic()
                else:
                    # Keep pumping OpenCV events lightly
                    cv2.waitKey(1)
                    if (time.monotonic() - stall_t0) > 2.0:
                        self.logger.warn("Preview hidden for >2s; UI idle")
                        self.logger.periodic_flush()
                        stall_t0 = time.monotonic()

                time.sleep(self.target_dt)
            except Exception as e:
                self.logger.warn(f"Preview loop transient error: {e}")
                self.logger.periodic_flush()
                time.sleep(self.target_dt)
