#!/usr/bin/env python3
# object_worker.py — wraps ObjectInteraction in a thread worker (now forwards masks to PreviewGrid)

from __future__ import annotations
import time
from pathlib import Path
from collections import deque
from typing import Dict, Optional

import numpy as np
import cv2  # ✅ needed for resizing masks

from types_shared import FramePacket
from object_interaction import ObjectInteraction, SaveTargets
from event_triggers import ObjectUntouchedTrigger
from tunables import (
    OBJECT_TRIGGER_WINDOW_SEC,
    OBJECT_TRIGGER_MIN_PCT,
    OBJ_P_START,
    OBJ_Q_END,
    OBJ_COLORS,
)

# Internal per-cam, time-windowed state buffer
class _WindowBuffer:
    """Keeps last W seconds of per-object states with timestamps."""
    def __init__(self, window_sec: float):
        self.window_sec = float(max(0.1, window_sec))
        # each entry: (t_mono, untouched: Dict[str,bool], checking: Dict[str,bool])
        self.buf = deque()

    def push(self, t: float, untouched: Dict[str, bool], checking: Dict[str, bool]):
        self.buf.append((float(t), dict(untouched), dict(checking)))
        self._gc()

    def _gc(self):
        if not self.buf:
            return
        cutoff = self.buf[-1][0] - self.window_sec
        while self.buf and self.buf[0][0] < cutoff:
            self.buf.popleft()

    def pct_untouched(self, obj_order) -> Dict[str, float]:
        counts = {o: 0 for o in obj_order}
        good   = {o: 0 for o in obj_order}
        for _, unt, _ in self.buf:
            for o in obj_order:
                if o in unt:
                    counts[o] += 1
                    if bool(unt[o]):
                        good[o] += 1
        pct = {}
        for o in obj_order:
            c = counts[o]
            pct[o] = (float(good[o]) / float(c)) if c > 0 else 0.0
        return pct

    def latest_state_label(self, obj_order) -> Dict[str, str]:
        if not self.buf:
            return {o: "unknown" for o in obj_order}
        _, unt, chk = self.buf[-1]
        out = {}
        for o in obj_order:
            u = bool(unt.get(o, False))
            c = bool(chk.get(o, False))
            if u:
                out[o] = "untouched"
            elif c:
                out[o] = "checking"
            else:
                out[o] = "moving"
        return out


def object_worker(
    cam_label: str,
    q_obj,
    out_dir: Path,
    log_flush_sec: int,
    trigger: Optional[ObjectUntouchedTrigger],  # may be None if disabled
    colors: Optional[list[str]] = None,
    *,
    preview=None,  # ✅ PreviewGrid handle (optional)
):
    """
    Consumes FramePacket(s) from q_obj, runs ObjectInteraction in real time,
    aggregates a sliding time window, calls ObjectUntouchedTrigger.update(...),
    and forwards mask overlays to PreviewGrid if provided.
    """
    cam_dir = out_dir / cam_label
    csv_dir = cam_dir / "CSV"
    csv_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = cam_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # ---- Normalize colors for ObjectInteraction ----
    # If caller provided yellow_1/yellow_2, condense to a single "yellow" token.
    configured = (colors[:] if colors else OBJ_COLORS[:])
    colors_for_oi: list[str] = []
    saw_yellow = False
    for c in configured:
        lc = c.lower()
        if lc in ("yellow", "yellow_1", "yellow_2"):
            if not saw_yellow:
                colors_for_oi.append("yellow")
                saw_yellow = True
        else:
            colors_for_oi.append(lc)

    # Initialize interaction
    oi = ObjectInteraction(
        fps=30,  # timestamps are monotonic; exact fps here is not critical
        colors=colors_for_oi,
        start_time_sec=4.0,
        p_start=int(OBJ_P_START),
        q_end=int(OBJ_Q_END),
        ref_frame=100,
        depth_units=None,  # ✅ will be set from first packet
        save=SaveTargets(csv_dir=csv_dir, overlay_dir=None, logs_dir=logs_dir),
        cam_label=cam_label,
    )

    # Time-windowed buffer
    win = _WindowBuffer(OBJECT_TRIGGER_WINDOW_SEC)

    t0: Optional[float] = None
    last_flush = time.monotonic()
    depth_units_initialized = False  # ✅ ensure depths.csv gets values

    def _place_masks_fullframe(local_masks: Dict[str, np.ndarray],
                               crop_box, full_H: int, full_W: int) -> Dict[str, np.ndarray]:
        """Return dict of HxW uint8 masks aligned to full color frame."""
        if not local_masks:
            return {}
        if crop_box is None:
            # already full-frame or unknown ROI → resize to full frame
            out = {}
            for k, v in local_masks.items():
                if v is None or v.size == 0:
                    continue
                if v.shape[:2] != (full_H, full_W):
                    v = cv2.resize(v, (full_W, full_H), interpolation=cv2.INTER_NEAREST)
                out[k] = v
            return out

        x_min, y_min, x_max, y_max = crop_box
        h, w = (y_max - y_min), (x_max - x_min)
        out = {}
        for k, m in local_masks.items():
            if m is None or m.size == 0:
                continue
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            canvas = np.zeros((full_H, full_W), dtype=np.uint8)
            canvas[y_min:y_max, x_min:x_max] = m
            out[k] = canvas
        return out

    # ✅ Normalize states to fixed legend keys expected by PreviewGrid
    LEGEND_KEYS = ("red", "green", "gray", "yellow_1", "yellow_2", "gold")

    def _expand_states_for_preview(raw_unt: Dict[str, bool],
                                   raw_chk: Dict[str, bool],
                                   masks: Dict[str, np.ndarray]) -> tuple[Dict[str, bool], Dict[str, bool]]:
        """
        Expand single 'yellow' into 'yellow_1'/'yellow_2' for both untouched/checking.
        If masks contain yellow_1/yellow_2, use their presence as hint; otherwise copy the same flag to both.
        Always return dicts with LEGEND_KEYS.
        """
        unt = {k: False for k in LEGEND_KEYS}
        chk = {k: False for k in LEGEND_KEYS}

        # Direct keys first
        for k in ("red", "green", "gray", "gold"):
            if k in raw_unt: unt[k] = bool(raw_unt.get(k, False))
            if k in raw_chk: chk[k] = bool(raw_chk.get(k, False))

        # Handle yellow family
        if "yellow_1" in raw_unt or "yellow_2" in raw_unt or "yellow" in raw_unt:
            yv = bool(raw_unt.get("yellow", False))
            unt["yellow_1"] = bool(raw_unt.get("yellow_1", yv))
            unt["yellow_2"] = bool(raw_unt.get("yellow_2", yv))
        if "yellow_1" in raw_chk or "yellow_2" in raw_chk or "yellow" in raw_chk:
            yv = bool(raw_chk.get("yellow", False))
            chk["yellow_1"] = bool(raw_chk.get("yellow_1", yv))
            chk["yellow_2"] = bool(raw_chk.get("yellow_2", yv))

        # If masks only have a single 'yellow', mirroring already handled above.
        return unt, chk

    # Helper: single-step process
    def _step(pkt: FramePacket):
        nonlocal t0, last_flush, depth_units_initialized
        t_abs = pkt.t_ns / 1e9 if pkt.t_ns else time.monotonic()
        if t0 is None:
            t0 = t_abs

        # ✅ Initialize depth units once from live packet so depths.csv fills
        if not depth_units_initialized:
            try:
                if hasattr(oi, "set_depth_units"):
                    oi.set_depth_units(float(pkt.depth_scale_m))
                elif hasattr(oi, "depth_units"):
                    oi.depth_units = float(pkt.depth_scale_m)  # fallback attribute
                depth_units_initialized = True
            except Exception:
                # keep going even if the method/attr doesn't exist
                pass

        # Ingest frame
        states = oi.ingest_frame(
            color_full=pkt.color,
            depth_full=pkt.depth,
            fnum=int(pkt.frame_id),
            mediapipe_overlay=None,
            save_overlay=False
        )

        # Window update (use raw first; window isn't tied to preview legend)
        win.push(t_abs, states.get("untouched", {}), states.get("checking", {}))

        # Forward overlays to preview (if available)
        local_masks = states.get("masks", {}) or {}
        crop_box    = states.get("crop_box", None)

        if preview is not None:
            try:
                full_masks = _place_masks_fullframe(local_masks, crop_box, pkt.color.shape[0], pkt.color.shape[1])
                # 🔑 Normalize/expand to legend keys so circles render correctly
                unt_norm, chk_norm = _expand_states_for_preview(
                    states.get("untouched", {}),
                    states.get("checking", {}),
                    full_masks
                )
                preview.update_objects({"untouched": unt_norm, "checking": chk_norm}, masks=full_masks)
            except Exception:
                # keep preview resilient
                pass

        # Trigger updates (no overlay pop for object trigger by design)
        if trigger is not None and hasattr(oi, "untouched_out"):
            # obj_order for trigger can follow OI’s own naming
            order_for_trigger = getattr(oi, "obj_order", list(local_masks.keys()))
            pct = win.pct_untouched(order_for_trigger)
            _ = win.latest_state_label(order_for_trigger)  # stl kept if you need it for debugging

            # threshold currently unused for the trigger decision (kept for debug/telemetry)
            _thr = OBJECT_TRIGGER_MIN_PCT if OBJECT_TRIGGER_MIN_PCT <= 1.0 else (OBJECT_TRIGGER_MIN_PCT / 100.0)
            elapsed_s = (t_abs - t0) if t0 is not None else 0.0

            try:
                # 🔁 Merge confirmed + live-open spans so the trigger sees ongoing intervals
                if hasattr(oi, "get_untouched_spans_for_trigger"):
                    merged_spans = oi.get_untouched_spans_for_trigger(int(pkt.frame_id))
                else:
                    merged_spans = getattr(oi, "untouched_out", {})

                trigger.update(
                    cam_label=cam_label,
                    elapsed_time_s=elapsed_s,
                    frame_idx=int(pkt.frame_id),
                    fps=30.0,  # cadence window math
                    untouched_out=merged_spans,
                    preview_grid=None,
                )
            except Exception:
                pass

        # Periodic flush cadence (logger in ObjectInteraction handles CSV flush)
        now = time.monotonic()
        if (now - last_flush) >= max(1, int(log_flush_sec)):
            last_flush = now

    # Main loop
    while True:
        if getattr(q_obj, "closed", False):
            break
        try:
            pkt: FramePacket = q_obj.get(timeout=0.05)
        except Exception:
            if hasattr(q_obj, "closed") and getattr(q_obj, "closed", False):
                break
            continue
        try:
            _step(pkt)
        except Exception:
            # Silent resilience
            pass

    # finalize untouched intervals
    try:
        oi.finalize()
    except Exception:
        pass
