#!/usr/bin/env python3
# object_interaction.py — real-time friendly with depth mm stats via ROI+binary mask
import os, re, cv2, csv, math
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

# ───────────────────────── Tunable defaults ─────────────────────────
DEFAULT_DEPTH_UNITS = 0.0010000000474974513  # meters per unit
YELLOW_TRACK_START  = 101  # kept for compatibility; not used for gating anymore
GRAY_JUMP_START     = 101
GRAY_JUMP_THRESH    = 79.4  # px
MIN_CONTOUR_AREA    = 65
AREA_KEEP_FRAC      = 0.35

# ───────────────────────── Utilities ─────────────────────────
INT_RE = re.compile(r"(\d+)")
def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(INT_RE, s)]

def parse_frame_num(fname: str) -> Optional[int]:
    m = re.search(r"frame_(\d+)\.(png|jpg|jpeg|tif|tiff)$", fname, re.IGNORECASE)
    return int(m.group(1)) if m else None

# ───────────────────────── ArUco crop helpers ─────────────────────────
def build_aruco_detector():
    import cv2.aruco as aruco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    return aruco.ArucoDetector(aruco_dict, parameters)

def detect_crop_box(img_bgr: np.ndarray, detector) -> Optional[Tuple[int,int,int,int]]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) < 3 or len(ids) > 4:
        return None
    selected_points = []
    for i, corner in enumerate(corners):
        marker_id = ids[i][0]
        pts = corner.reshape((4,2))
        sel = pts[0] if marker_id in [1,2,3] else (pts[1] if marker_id == 0 else None)
        if sel is not None:
            selected_points.append(sel)
    if len(selected_points) < 2:
        return None
    pts = np.array(selected_points, dtype=int)
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    return int(x_min), int(y_min), int(x_max), int(y_max)

# ───────────────────────── Masking utilities ─────────────────────────
def union_significant_contours(binary_mask: np.ndarray, min_area: int, area_frac_keep: float):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return None
    contours.sort(key=cv2.contourArea, reverse=True)
    largest = cv2.contourArea(contours[0])
    keep = [c for c in contours if cv2.contourArea(c) >= area_frac_keep * largest]
    if not keep:
        keep = [contours[0]]
    out = np.zeros_like(binary_mask)
    cv2.drawContours(out, keep, -1, 255, thickness=cv2.FILLED)
    return out

def compute_masks(image_bgr: np.ndarray, hsv: np.ndarray, colors: List[str],
                  frame_idx: int, prev_two_yellow: Optional[Tuple[Tuple[int,int],Tuple[int,int]]]) -> Tuple[Dict[str,np.ndarray], Optional[Tuple[Tuple[int,int],Tuple[int,int]]]]:
    kernel = np.ones((3,3), np.uint8)
    masks = {}

    if "red" in colors:
        m1 = cv2.inRange(hsv, np.array([0,120,110]),   np.array([10,240,235]))
        m2 = cv2.inRange(hsv, np.array([165,120,110]), np.array([180,240,235]))
        m  = cv2.bitwise_or(m1, m2)
        m  = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
        mf = union_significant_contours(m, MIN_CONTOUR_AREA+15, AREA_KEEP_FRAC)
        if mf is not None: masks["red"] = mf

    if "green" in colors:
        m  = cv2.inRange(hsv, np.array([40,70,70]), np.array([70,150,190]))
        m  = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
        mf = union_significant_contours(m, MIN_CONTOUR_AREA+15, AREA_KEEP_FRAC)
        if mf is not None: masks["green"] = mf

    if "gray" in colors:
        m  = cv2.inRange(hsv, np.array([10,0,90]), np.array([100,50,160]))
        m  = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filt = []
        for c in contours:
            if not any((pt[0][0] < 60 and pt[0][1] > 300) for pt in c):
                if cv2.contourArea(c) > MIN_CONTOUR_AREA: filt.append(c)
        if filt:
            c  = max(filt, key=cv2.contourArea)
            mf = np.zeros_like(m); cv2.drawContours(mf, [c], -1, 255, -1)
            masks["gray"] = mf

    # ── YELLOW: start tracking only when BOTH are seen once; otherwise label-by-order (no seeding) ──
    if "yellow" in colors:
        my = cv2.inRange(hsv, np.array([18,175,160]), np.array([30,255,255]))
        my = cv2.morphologyEx(cv2.morphologyEx(my, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(my, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA/2]

        cents: List[Tuple[int,int]] = []
        for c in contours:
            M = cv2.moments(c)
            if M["m00"] > 0:
                cents.append((int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])))

        def contour_to_mask(c):
            mf = np.zeros_like(my); cv2.drawContours(mf, [c], -1, 255, -1); return mf

        if prev_two_yellow is None:
            for i, c in enumerate(contours[:2]):
                masks[f"yellow_{i+1}"] = contour_to_mask(c)
            if len(cents) == 2:
                prev_two_yellow = (cents[0], cents[1])  # seed next frame
        else:
            if len(cents) == 2:
                (py1x,py1y),(py2x,py2y) = prev_two_yellow
                (c0x,c0y),(c1x,c1y)     = cents[0], cents[1]
                d0y1 = math.hypot(c0x - py1x, c0y - py1y)
                d0y2 = math.hypot(c0x - py2x, c0y - py2y)
                y1_idx, y2_idx = (0,1) if d0y1 <= d0y2 else (1,0)
                masks["yellow_1"] = contour_to_mask(contours[y1_idx])
                masks["yellow_2"] = contour_to_mask(contours[y2_idx])
                prev_two_yellow   = (cents[y1_idx], cents[y2_idx])
            elif len(cents) == 1:
                (py1x,py1y),(py2x,py2y) = prev_two_yellow
                (cx,cy) = cents[0]
                d1 = math.hypot(cx - py1x, cy - py1y)
                d2 = math.hypot(cx - py2x, cy - py2y)
                if d1 <= d2:
                    label = "yellow_1"; prev_two_yellow = ((cx,cy),(py2x,py2y))
                else:
                    label = "yellow_2"; prev_two_yellow = ((py1x,py1y),(cx,cy))
                if contours:
                    masks[label] = contour_to_mask(contours[0])

    if "gold" in colors:
        m  = cv2.inRange(hsv, np.array([18,100,120]), np.array([23,190,190]))
        m  = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]
        if contours:
            c  = max(contours, key=cv2.contourArea)
            mf = np.zeros_like(m); cv2.drawContours(mf, [c], -1, 255, -1)
            masks["gold"] = mf

    return masks, prev_two_yellow

def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    return x,y,w,h

def com_from_bbox(xywh: Tuple[int,int,int,int]) -> Tuple[int,int]:
    x,y,w,h = xywh
    return x + w//2, y + h//2

# ───────────────────────── Depth helpers ─────────────────────────
def depth_to_mm(depth_raw: np.ndarray, depth_units: float):
    return (depth_raw.astype(np.float32) * depth_units * 1000.0)

def compute_z_stats_mm(depth_mm: np.ndarray, mask: np.ndarray) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    valid = (mask > 0) & (depth_mm > 0)
    if not np.any(valid): return None, None, None
    vals = depth_mm[valid]; vals = vals[np.isfinite(vals)]
    if vals.size == 0: return None, None, None
    return float(np.min(vals)), float(np.max(vals)), float(np.median(vals))

# ───────────────────────── Untouched state machine ─────────────────────────
def depth_ok_for(obj_name: str, z_mm: float) -> bool:
    if obj_name.startswith("yellow"): return 720 <= z_mm <= 760
    else:                              return 680 <= z_mm <= 760

class UntouchedState:
    def __init__(self):
        self.interval_active = False
        self.bbox = None
        self.x_start = None
        self.last_good = None
        self.check_start = None
        self.check_end = None
        self.saw_any_bad = False
        self.ref_wh = None

def update_untouched_states(
    states: Dict[str, UntouchedState],
    obj_order: List[str],
    fnum: int,
    fps: int,
    start_time: float,
    p: int,
    q: int,
    xy_dict: Dict[str, Optional[Tuple[int,int]]],
    z_dict: Dict[str, Optional[float]],
    untouched_out: Dict[str, List[List[int]]],
    checking_out: Dict[str, List[List[int]]]
):
    start_frame = int(start_time * fps)
    for obj in obj_order:
        st = states[obj]
        xy = xy_dict.get(obj)
        z  = z_dict.get(obj)

        if st.interval_active:
            if st.check_start is not None and st.check_end is not None:
                if fnum <= st.check_end:
                    if xy is not None and z is not None:
                        cx, cy = xy
                        xmin, ymin, xmax, ymax = st.bbox
                        inside = (xmin <= cx <= xmax) and (ymin <= cy <= ymax)
                        z_ok   = depth_ok_for(obj, z)
                        if inside and z_ok:
                            checking_out[obj].append([st.check_start, fnum - 1])
                            st.check_start = None; st.check_end = None; st.saw_any_bad = False
                            st.last_good = fnum
                        else:
                            st.saw_any_bad = True
                else:
                    if st.saw_any_bad:
                        end_frame = st.last_good if st.last_good is not None else (st.x_start - 1)
                        if end_frame is not None and st.x_start is not None and end_frame >= st.x_start:
                            untouched_out[obj].append([st.x_start - p, end_frame])
                        checking_out[obj].append([st.check_start, st.check_end])
                        st.interval_active = False; st.bbox = None; st.x_start = None; st.last_good = None
                        st.check_start = None; st.check_end = None; st.saw_any_bad = False
                    else:
                        checking_out[obj].append([st.check_start, st.check_end])
                        st.check_start = None; st.check_end = None; st.saw_any_bad = False
                        st.last_good = fnum
                continue

            if xy is None or z is None:
                st.last_good = fnum; continue
            cx, cy = xy
            xmin, ymin, xmax, ymax = st.bbox
            inside = (xmin <= cx <= xmax) and (ymin <= cy <= ymax)
            z_ok   = depth_ok_for(obj, z)
            if inside and z_ok:
                st.last_good = fnum
            else:
                st.check_start = fnum; st.check_end = fnum + q - 1; st.saw_any_bad = True
            continue

        if fnum < start_frame: continue
        if xy is None:         continue
        if st.ref_wh is None:  continue
        continue

# ───────────────────────── Real-time friendly processor ─────────────────────────
@dataclass
class SaveTargets:
    csv_dir: Path                  # <cam>/CSV
    overlay_dir: Optional[Path]    # e.g., color_mp (None to disable saving)
    logs_dir: Optional[Path] = None  # <cam>/logs (auto-filled if None)

class ObjectInteraction:
    def __init__(
        self,
        fps: int,
        colors: List[str],
        start_time_sec: float,
        p_start: int,
        q_end: int,
        ref_frame: int = 100,
        depth_units: float = DEFAULT_DEPTH_UNITS,
        save: Optional[SaveTargets] = None,
        cam_label: str = "cam?"
    ):
        self.save = save
        if self.save:
            # CSVs
            self.save.csv_dir.mkdir(parents=True, exist_ok=True)
            self.xy_csv_path   = self.save.csv_dir / "xy_com.csv"
            self.depths_path   = self.save.csv_dir / "depths.csv"
            self._xy_writer    = csv.DictWriter(open(self.xy_csv_path, "w", newline=""),
                                                fieldnames=["filename","red","green","gray","yellow_1","yellow_2","gold"])
            self._xy_writer.writeheader()
            depth_cols = ["frame"] + [f"{k}_{suf}" for k in ["red","green","gray","yellow_1","yellow_2","gold"] for suf in ("min_mm","max_mm","com_mm")] + ["mean_com_mm"]
            self._z_writer     = csv.DictWriter(open(self.depths_path, "w", newline=""), fieldnames=depth_cols)
            self._z_writer.writeheader()

            # LOGS dir (default to <cam>/logs beside CSV)
            logs_dir = self.save.logs_dir if self.save.logs_dir else (self.save.csv_dir.parent / "logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            self.summary_path  = logs_dir / "untouched_intervals_xyz.log"

            # Ensure overlay dir exists if provided
            if self.save.overlay_dir is not None:
                self.save.overlay_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._xy_writer = None; self._z_writer = None
            self.summary_path = None  # no file output if no save target

        self.fps = fps
        self.colors = colors[:]
        self.start_time = start_time_sec
        self.p = p_start
        self.q = q_end
        self.ref_frame = ref_frame
        self.depth_units = depth_units
        self.cam_label = cam_label

        self.obj_order = []
        for c in self.colors:
            self.obj_order += (["yellow_1","yellow_2"] if c == "yellow" else [c])
        for k in ["red","green","gray","yellow_1","yellow_2","gold"]:
            if k not in self.obj_order: self.obj_order.append(k)  # fixed CSV columns

        self.states        = {obj: UntouchedState() for obj in self.obj_order}
        self.untouched_out = {obj: [] for obj in self.obj_order}
        self.checking_out  = {obj: [] for obj in self.obj_order}
        self.warmup_good_count = {obj: 0 for obj in self.obj_order}

        # NEW: keep a provisional open span per object for triggers (not persisted)
        self._live_open: Dict[str, Optional[Tuple[int,int]]] = {obj: None for obj in self.obj_order}

        self.prev_two_yellow = None
        self.prev_gray_accept_xy = None
        self.prev_gray_accept_frame = None

        self.crop_box: Optional[Tuple[int,int,int,int]] = None
        self._aruco = build_aruco_detector()

    def set_crop_box(self, box: Optional[Tuple[int,int,int,int]]):
        self.crop_box = box

    def set_depth_units(self, units: float):
        try:
            self.depth_units = float(units)
        except Exception:
            pass

    def _crop_img(self, img: np.ndarray) -> np.ndarray:
        if not self.crop_box: return img
        x_min,y_min,x_max,y_max = self.crop_box
        return img[y_min:y_max, x_min:x_max]

    def _maybe_detect_crop(self, color_full: np.ndarray, fnum: int):
        if self.crop_box is not None: return
        if 50 <= fnum <= 100 or fnum >= 101:
            box = detect_crop_box(color_full, self._aruco)
            if box is not None:
                self.crop_box = box

    # NEW: merge confirmed spans + live open interval (if any) for triggers
    def get_untouched_spans_for_trigger(self, current_frame: int) -> Dict[str, List[List[int]]]:
        merged: Dict[str, List[List[int]]] = {}
        for obj in self.obj_order:
            spans = [s[:] for s in self.untouched_out.get(obj, [])]
            live = self._live_open.get(obj)
            if live is not None:
                a, b = live
                # extend b to at least current_frame to be generous for coverage
                b = max(b, int(current_frame))
                if spans and a <= spans[-1][1] + 1:
                    # overlaps/adjacent → extend last
                    spans[-1][1] = max(spans[-1][1], b)
                    spans[-1][0] = min(spans[-1][0], a)
                else:
                    spans.append([a, b])
            merged[obj] = spans
        return merged

    def ingest_frame(
        self,
        color_full: np.ndarray,
        depth_full: Optional[np.ndarray],
        fnum: int,
        *,
        mediapipe_overlay: Optional[np.ndarray] = None,
        save_overlay: bool = False
    ) -> Dict[str, Dict[str, bool]]:
        """
        Returns instantaneous states + masks + crop_box (for preview grid).
        """
        # 1) maybe set crop
        self._maybe_detect_crop(color_full, fnum)
        color = self._crop_img(color_full.copy())
        hsv   = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)

        # 2) masks & yellow tracking
        masks, self.prev_two_yellow = compute_masks(color, hsv, self.colors, fnum, self.prev_two_yellow)

        # 3) bboxes on cropped
        xy_bb = {k: None for k in ["red","green","gray","yellow_1","yellow_2","gold"]}
        for obj in xy_bb.keys():
            if obj in masks:
                bb = bbox_from_mask(masks[obj])
                if bb is not None: xy_bb[obj] = bb

        # 4) seed ref size at ref_frame
        if fnum == self.ref_frame:
            for obj in self.obj_order:
                bb = xy_bb.get(obj)
                if bb is not None:
                    x,y,w,h = bb
                    self.states[obj].ref_wh = (float(w), float(h))

        # 5) xy_com.csv
        row_xy = {"filename": f"frame_{fnum:04d}.png"}
        for obj in ["red","green","gray","yellow_1","yellow_2","gold"]:
            bb = xy_bb.get(obj)
            if bb is None:
                row_xy[obj] = ""
                continue
            cx, cy = com_from_bbox(bb)
            x,y,w,h = bb
            val = f"({cx},{cy},{w},{h})"
            if obj == "gray":
                if fnum >= GRAY_JUMP_START and self.prev_gray_accept_xy is not None:
                    jump = math.hypot(cx - self.prev_gray_accept_xy[0], cy - self.prev_gray_accept_xy[1])
                    if jump > GRAY_JUMP_THRESH:
                        row_xy[obj] = ""
                    else:
                        row_xy[obj] = val
                        self.prev_gray_accept_xy = (cx, cy); self.prev_gray_accept_frame = fnum
                else:
                    row_xy[obj] = val
                    self.prev_gray_accept_xy = (cx, cy); self.prev_gray_accept_frame = fnum
            else:
                row_xy[obj] = val
        if self._xy_writer: self._xy_writer.writerow(row_xy)

        # 6) depths.csv
        row_z = {"frame": f"frame_{fnum:04d}"}
        xy_curr_for_state = {}
        z_curr_for_state  = {}
        com_list = []

        if depth_full is not None:
            depth_roi = depth_full
            if self.crop_box is not None:
                x_min,y_min,x_max,y_max = self.crop_box
                depth_roi = depth_roi[y_min:y_max, x_min:x_max]
            if depth_roi.ndim == 3:
                depth_roi = cv2.cvtColor(depth_roi, cv2.COLOR_BGR2GRAY)
            if depth_roi.dtype != np.uint16:
                depth_roi = depth_roi.astype(np.uint16)
            depth_mm = depth_to_mm(depth_roi, self.depth_units)
        else:
            depth_mm = None

        h,w = color.shape[:2]
        for obj in ["red","green","gray","yellow_1","yellow_2","gold"]:
            mask_bin = masks[obj] if obj in masks else np.zeros((h,w), dtype=np.uint8)
            if depth_mm is None:
                zmin=zmax=zcom=None
            else:
                zmin,zmax,zcom = compute_z_stats_mm(depth_mm, mask_bin)
            row_z[f"{obj}_min_mm"] = "" if zmin is None else f"{zmin:.3f}"
            row_z[f"{obj}_max_mm"] = "" if zmax is None else f"{zmax:.3f}"
            row_z[f"{obj}_com_mm"] = "" if zcom is None else f"{zcom:.3f}"
            if zcom is not None: com_list.append(zcom)

            bb = xy_bb.get(obj)
            xy_curr_for_state[obj] = None if bb is None else com_from_bbox(bb)
            z_curr_for_state[obj]  = zcom

        row_z["mean_com_mm"] = "" if not com_list else f"{np.mean(com_list):.3f}"
        if self._z_writer: self._z_writer.writerow(row_z)

        # 7) start intervals warmup (unchanged)
        for obj in self.obj_order:
            st = self.states[obj]
            xy = xy_curr_for_state[obj]; zc = z_curr_for_state[obj]
            if st.ref_wh is None:              self.warmup_good_count[obj] = 0; continue
            if xy is None or zc is None:       self.warmup_good_count[obj] = 0; continue
            cx, cy = xy; w_ref, h_ref = st.ref_wh
            cand_xmin, cand_xmax = cx - w_ref/2.0, cx + w_ref/2.0
            cand_ymin, cand_ymax = cy - h_ref/2.0, cy + h_ref/2.0
            inside = (cand_xmin <= cx <= cand_xmax) and (cand_ymin <= cy <= cand_ymax)
            z_ok   = depth_ok_for(obj, zc)
            self.warmup_good_count[obj] = (self.warmup_good_count[obj] + 1) if (inside and z_ok) else 0
            if (not st.interval_active) and (fnum >= int(self.start_time*self.fps)) and (self.warmup_good_count[obj] >= max(1, self.p//2)):
                st.interval_active = True
                st.bbox = (cand_xmin, cand_ymin, cand_xmax, cand_ymax)
                st.x_start = fnum
                st.last_good = fnum
                st.check_start = None

        # 8) strict evaluator
        update_untouched_states(
            states=self.states,
            obj_order=self.obj_order,
            fnum=fnum,
            fps=self.fps,
            start_time=self.start_time,
            p=self.p,
            q=self.q,
            xy_dict=xy_curr_for_state,
            z_dict=z_curr_for_state,
            untouched_out=self.untouched_out,
            checking_out=self.checking_out
        )

        # NEW: update provisional open span (not persisted)
        for obj in self.obj_order:
            st = self.states[obj]
            if st.interval_active and st.check_start is None:
                if st.x_start is not None:
                    live_start = max(0, int(st.x_start) - int(self.p))
                    live_end   = int(st.last_good) if st.last_good is not None else int(fnum)
                    self._live_open[obj] = (live_start, live_end)
            else:
                self._live_open[obj] = None

        # 9) overlay saving (unchanged)
        if save_overlay and self.save and self.save.overlay_dir is not None:
            overlay = (mediapipe_overlay.copy()
                       if mediapipe_overlay is not None else color_full.copy())

            COLOR_MAP = {
                "red":      (0, 0, 255),
                "green":    (0, 255, 0),
                "gray":     (255, 255, 255),
                "yellow_1": (0, 255, 255),
                "yellow_2": (0, 128, 255),
                "gold":     (128, 0, 255),
            }
            draw_order = ["red","green","gray","yellow_1","yellow_2","gold"]

            if self.crop_box:
                x_min,y_min,x_max,y_max = self.crop_box
                crop_w, crop_h = x_max - x_min, y_max - y_min

                for obj in draw_order:
                    m = masks.get(obj, None)
                    if m is None:
                        continue

                    # ensure mask is ROI-sized; if full-frame, crop to ROI; else skip
                    if m.shape[:2] != (crop_h, crop_w):
                        if m.shape[0] >= y_max and m.shape[1] >= x_max:
                            m = m[y_min:y_max, x_min:x_max]
                        else:
                            continue

                    color_bgr = COLOR_MAP.get(obj, (255,255,255))

                    colored_roi = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
                    colored_roi[m > 0] = color_bgr

                    roi = overlay[y_min:y_max, x_min:x_max]
                    blended = cv2.addWeighted(roi, 1.0, colored_roi, 0.4, 0)

                    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        cmax = max(cnts, key=cv2.contourArea)
                        cv2.drawContours(blended, [cmax], -1, color_bgr, thickness=2)

                    overlay[y_min:y_max, x_min:x_max] = blended

            out_path = self.save.overlay_dir / f"frame_{fnum:04d}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), overlay)

        # 10) return instantaneous state + masks + crop_box (LIVE-AWARE)
        def in_spans(spans, idx): 
            return any(a <= idx <= b for (a, b) in spans)

        state_now_untouched = {}
        state_now_checking  = {}

        for obj in self.obj_order:
            u = in_spans(self.untouched_out.get(obj, []), fnum)
            c = in_spans(self.checking_out.get(obj,  []), fnum)

            st = self.states.get(obj)
            if st and st.interval_active:
                if st.check_start is not None and st.check_end is not None:
                    if st.check_start <= fnum <= st.check_end:
                        c = True; u = False
                else:
                    if st.x_start is not None:
                        live_u_start = max(0, int(st.x_start) - int(self.p))
                        if live_u_start <= fnum:
                            u = True; c = False

            state_now_untouched[obj] = bool(u)
            state_now_checking[obj]  = bool(c)

        return {
            "untouched": state_now_untouched,
            "checking": state_now_checking,
            "masks": masks,            # CROPPED coords
            "crop_box": self.crop_box  # FULL-frame coords or None
        }

    def finalize(self) -> Dict[str, List[List[int]]]:
        last_frame = 0
        for spans in self.untouched_out.values():
            for _, b in spans:
                last_frame = max(last_frame, b)
        for obj, st in self.states.items():
            if st.interval_active:
                if st.check_start is not None and st.check_end is not None:
                    bucket_end = st.check_end
                    if st.saw_any_bad:
                        end_frame = st.last_good if st.last_good is not None else (st.x_start - 1 if st.x_start else bucket_end)
                        if st.x_start is not None and end_frame is not None and end_frame >= st.x_start:
                            self.untouched_out[obj].append([st.x_start - self.p, end_frame])
                    else:
                        self.checking_out[obj].append([st.check_start, bucket_end])
                        st.last_good = last_frame
                    st.check_start = None; st.check_end = None; st.saw_any_bad = False
                if st.interval_active:
                    end_frame = st.last_good if st.last_good is not None else (st.x_start - 1 if st.x_start else last_frame)
                    if st.x_start is not None and end_frame is not None and end_frame >= st.x_start:
                        self.untouched_out[obj].append([st.x_start - self.p, end_frame])

        # Write realtime summary to <run_root>/<cam>/logs/untouched_intervals_xyz.log
        try:
            if self.summary_path is not None:
                with open(self.summary_path, "w", encoding="utf-8") as f:
                    f.write("📌 Untouched intervals (confirmed):\n")
                    for obj in ["red","green","gray","yellow_1","yellow_2","gold"]:
                        spans = self.untouched_out.get(obj, [])
                        if not spans:
                            f.write(f"{obj}: None\n")
                        else:
                            parts = [f"[{a}, {b}]" for a, b in spans]
                            f.write(f"{obj}: {', '.join(parts)}\n")

                    f.write("\n🕒 Checking windows (q-buckets):\n")
                    for obj in ["red","green","gray","yellow_1","yellow_2","gold"]:
                        spans = self.checking_out.get(obj, [])
                        if not spans:
                            f.write(f"{obj}: None\n")
                        else:
                            parts = [f"[{a}, {b}]" for a, b in spans]
                            f.write(f"{obj}: {', '.join(parts)}\n")
        except Exception:
            pass  # avoid affecting shutdown

        return self.untouched_out
