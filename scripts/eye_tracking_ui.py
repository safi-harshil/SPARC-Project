"""
eye_tracking_ui.py
------------------
Premium HUD for L2CS-Net gaze tracking pipeline.

DESIGN:
  ┌──────────────────────────────────────────────────────────────┐
  │  HEAD  Yaw=+22.4°  Pitch=-166.3°  │  EYE  │  GAZE         │  ← top banner
  ├──────────────────┬─────────────────────────┬────────────────┤
  │  READOUT panel   │                         │  GAZE          │
  │  • HEAD Yaw      │   live camera feed      │  ANALYTICS     │
  │  • HEAD Pitch    │   (iris rings drawn      │  • gaze grid   │
  │  • FPS           │    on eyes here)         │  • mini charts │
  │  • DISTANCE      │                         │  • system info │
  │  • STATUS        │                         │                │
  └──────────────────┴─────────────────────────┴────────────────┘
  │              [SPACE] Pause  [ESC] Stop  [Q] Close           │
  └──────────────────────────────────────────────────────────────┘

KEY CHANGES vs old code:
  • REMOVED the ugly plain red dot on eye centre
  • ADDED beautiful iris ring (cyan glow circle on each eye)
  • REMOVED raw red arrow gaze pointer
  • ADDED glowing cyan gaze dot on the gaze analytics grid
  • ADDED "Gaze Statistics" mini sparkline bars
  • ADDED "System Info" icon row at the bottom of right panel
  • ADDED [ACTIVE POSE] status badge in sidebar
  • Panels use rounded-corner simulation with gradient-edge glow
  • All colours match the reference dark-teal (#00C8E0) palette
"""

import cv2
import numpy as np
import time
import math
import collections

START_TIME = time.time()

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (BGR)
# ─────────────────────────────────────────────────────────────────────────────
TEAL     = (200, 200,   0)   # amber  — top-level section labels (matches ref)
TEAL_ACC = (220, 180,   0)   # accent teal for borders & rules
CYAN     = (255, 220,   0)   # bright cyan — gaze point, iris ring
CHEAD    = (180, 230,  80)   # soft yellow-green — head values
CGAZE    = (255, 210,   0)   # cyan-white — gaze values
CZ       = (220, 180,   0)   # teal — Z / gaze section
COK      = ( 80, 255,  80)   # bright green — FPS / OK
CWARN    = (  0, 150, 255)   # orange — warning
WH       = (230, 230, 230)   # off-white — body text
DIM      = (130, 130, 130)   # dim gray — secondary text
PANEL_BG = 15                # panel fill intensity

# Iris / eye overlay colours
IRIS_OUTER = (200, 200,   0)   # teal ring
IRIS_INNER = (255, 240,   0)   # bright teal pupil dot
IRIS_GLOW  = (120,  90,   0)   # soft glow ring

# Gaze grid colours
GRID_BG    = (30, 40, 50)
GRID_LINE  = (60, 80, 100)
GRID_CROSS = (100, 120, 160)
GAZE_DOT   = (255, 210,   0)   # cyan gaze dot

# History buffer for mini sparklines
_YAW_HIST   = collections.deque(maxlen=60)
_PITCH_HIST = collections.deque(maxlen=60)


# ─────────────────────────────────────────────────────────────────────────────
# LOW-LEVEL PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def _clip_rect(img, x1, y1, x2, y2):
    H, W = img.shape[:2]
    return (max(0, int(x1)), max(0, int(y1)),
            min(W - 1, int(x2)), min(H - 1, int(y2)))


def panel(img, x1, y1, x2, y2, alpha=0.72, fill=PANEL_BG):
    """Semi-transparent dark panel, clips to image bounds."""
    x1, y1, x2, y2 = _clip_rect(img, x1, y1, x2, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = img[y1:y2, x1:x2]
    ov  = np.full_like(roi, fill)
    cv2.addWeighted(ov, alpha, roi, 1.0 - alpha, 0, roi)
    img[y1:y2, x1:x2] = roi


def glowing_panel(img, x1, y1, x2, y2, accent=(200, 180, 0), alpha=0.72):
    """Panel with a glowing top-border accent rule."""
    panel(img, x1, y1, x2, y2, alpha=alpha)
    x1i, y1i, x2i, _ = _clip_rect(img, x1, y1, x2, y2)
    # 3-px glow: dim outer, bright inner
    cv2.line(img, (x1i, y1i),     (x2i, y1i),     _dim(accent, 0.45), 1)
    cv2.line(img, (x1i, y1i + 1), (x2i, y1i + 1), accent,             2)


def _dim(color, factor):
    return tuple(max(0, min(255, int(c * factor))) for c in color)


def T(img, text, pos, scale, color, thick=1, shadow=True):
    """Text with drop-shadow."""
    x, y = int(pos[0]), int(pos[1])
    f = cv2.FONT_HERSHEY_SIMPLEX
    if shadow:
        cv2.putText(img, text, (x + 1, y + 1), f, scale,
                    (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), f, scale, color, thick, cv2.LINE_AA)


def Ts(img, text, pos, scale, color, thick=1):
    """Text WITHOUT shadow (for inside panels)."""
    T(img, text, pos, scale, color, thick, shadow=False)


def hline(img, x1, x2, y, color, thick=1):
    cv2.line(img, (int(x1), int(y)), (int(x2), int(y)), color, thick, cv2.LINE_AA)


def vline(img, x, y1, y2, color, thick=1):
    cv2.line(img, (int(x), int(y1)), (int(x), int(y2)), color, thick, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# IRIS RING  (the beautiful eye overlay from the reference image)
# ─────────────────────────────────────────────────────────────────────────────

def draw_iris_ring(img, cx, cy, r, yaw_deg, pitch_deg, S=1.0):
    """
    Draw a glowing teal iris ring at (cx,cy) with radius r.
    A small cyan dot shows gaze direction offset inside the ring.
    Matches the reference image's eye overlay exactly.
    """
    cx, cy, r = int(cx), int(cy), max(6, int(r))

    # ── outer glow ring (semi-transparent, soft) ──────────────────────────
    glow_r = r + max(3, int(4 * S))
    overlay = img.copy()
    cv2.circle(overlay, (cx, cy), glow_r, IRIS_GLOW, 2, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

    # ── main iris ring ─────────────────────────────────────────────────────
    cv2.circle(img, (cx, cy), r, IRIS_OUTER, max(1, int(1.5 * S)), cv2.LINE_AA)

    # ── inner thin ring ────────────────────────────────────────────────────
    cv2.circle(img, (cx, cy), max(2, r - max(2, int(3 * S))),
               _dim(IRIS_OUTER, 0.55), 1, cv2.LINE_AA)

    # ── gaze pupil dot offset inside the ring ─────────────────────────────
    max_off = int(r * 0.45)
    off_x = int(np.clip(math.sin(math.radians(yaw_deg))   * max_off, -max_off, max_off))
    off_y = int(np.clip(math.sin(math.radians(pitch_deg)) * max_off, -max_off, max_off))
    px, py = cx + off_x, cy + off_y

    dot_r = max(3, int(r * 0.28))
    cv2.circle(img, (px, py), dot_r + 1, _dim(IRIS_INNER, 0.4), -1, cv2.LINE_AA)  # glow
    cv2.circle(img, (px, py), dot_r,     IRIS_INNER,             -1, cv2.LINE_AA)  # core

    # ── specular highlight ─────────────────────────────────────────────────
    hl_x = px - max(1, dot_r // 3)
    hl_y = py - max(1, dot_r // 3)
    cv2.circle(img, (hl_x, hl_y), max(1, dot_r // 3), WH, -1, cv2.LINE_AA)

    # ── thin line from ring centre to pupil ───────────────────────────────
    cv2.line(img, (cx, cy), (px, py), _dim(IRIS_OUTER, 0.7),
             max(1, int(S)), cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# GAZE ANALYTICS GRID
# ─────────────────────────────────────────────────────────────────────────────

def draw_gaze_grid(img, x1, y1, size, yaw_deg, pitch_deg, S=1.0):
    """
    Draw the 'Current Gaze Point' grid with a glowing cyan dot.
    x1, y1 = top-left of grid; size = pixel side length.
    """
    x2, y2 = x1 + size, y1 + size

    # background fill
    panel(img, x1, y1, x2, y2, alpha=0.80, fill=20)
    cv2.rectangle(img, (x1, y1), (x2, y2), (80, 100, 140), 1, cv2.LINE_AA)

    # 4×4 grid lines
    step = size // 4
    for i in range(1, 4):
        gx = x1 + i * step
        gy = y1 + i * step
        cv2.line(img, (gx, y1), (gx, y2), GRID_LINE, 1, cv2.LINE_AA)
        cv2.line(img, (x1, gy), (x2, gy), GRID_LINE, 1, cv2.LINE_AA)

    # centre crosshair
    cx, cy = x1 + size // 2, y1 + size // 2
    hline(img, x1 + 2, x2 - 2, cy, GRID_CROSS, 1)
    vline(img, cx, y1 + 2, y2 - 2, GRID_CROSS, 1)
    cv2.circle(img, (cx, cy), 2, (180, 180, 180), -1, cv2.LINE_AA)

    # gaze dot — mapped from ±60° → grid space
    half = size // 2 - int(10 * S)
    dot_x = int(cx + np.clip(yaw_deg   / 60.0, -1.0, 1.0) * half)
    dot_y = int(cy - np.clip(pitch_deg / 60.0, -1.0, 1.0) * half)  # pitch up = negative Y

    # glow rings
    for gr, ga in [(10, 0.25), (7, 0.40), (5, 0.55)]:
        ov = img.copy()
        cv2.circle(ov, (dot_x, dot_y), int(gr * S), GAZE_DOT, -1, cv2.LINE_AA)
        cv2.addWeighted(ov, ga, img, 1 - ga, 0, img)

    # solid dot
    cv2.circle(img, (dot_x, dot_y), max(4, int(5 * S)), GAZE_DOT, -1, cv2.LINE_AA)
    cv2.circle(img, (dot_x, dot_y), max(2, int(3 * S)), WH,       -1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# MINI SPARKLINE CHART
# ─────────────────────────────────────────────────────────────────────────────

def draw_sparkline(img, x1, y1, w, h, data, color, label=""):
    """Draw a mini sparkline chart inside a small box."""
    x2, y2 = x1 + w, y1 + h
    panel(img, x1, y1, x2, y2, alpha=0.75, fill=18)
    cv2.rectangle(img, (x1, y1), (x2, y2), _dim(color, 0.4), 1, cv2.LINE_AA)

    if len(data) < 2:
        return

    vals = list(data)
    mn, mx = min(vals), max(vals)
    rng = max(mx - mn, 1.0)

    pts = []
    for i, v in enumerate(vals):
        px = x1 + int(i / (len(vals) - 1) * (w - 4)) + 2
        py = y2 - 4 - int((v - mn) / rng * (h - 8))
        pts.append((px, py))

    for i in range(len(pts) - 1):
        cv2.line(img, pts[i], pts[i + 1], color, 1, cv2.LINE_AA)

    # fill area under curve (semi-transparent)
    if len(pts) >= 2:
        poly = np.array(
            [(x1 + 2, y2 - 4)] + pts + [(x2 - 2, y2 - 4)],
            dtype=np.int32
        )
        ov = img.copy()
        cv2.fillPoly(ov, [poly], _dim(color, 0.25))
        cv2.addWeighted(ov, 0.45, img, 0.55, 0, img)

    if label:
        Ts(img, label, (x1 + 3, y1 + 10), 0.28, _dim(color, 0.9), 1)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM INFO ICONS  (bottom of right panel — matching reference)
# ─────────────────────────────────────────────────────────────────────────────

def draw_system_icons(img, x1, y, icon_size, S):
    """Draw simple system-info icon row: phone, battery, cpu, cloud."""
    icons = [
        ("[ ]", WH),           # phone / device
        ("[+]", COK),          # battery
        ("*",   CGAZE),        # CPU / processor
        ("~",   DIM),          # cloud / upload
    ]
    sp = max(icon_size + 8, int((icon_size + 14) * S))
    for i, (sym, col) in enumerate(icons):
        Ts(img, sym, (x1 + i * sp, y), 0.36 * S, col, 1)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HUD
# ─────────────────────────────────────────────────────────────────────────────

def draw_hud(
    img,
    smooth_X,
    smooth_Y,
    smooth_Z,
    head_horizontal,
    head_vertical,
    gaze_direction,
    lx, ly,
    rx, ry,
    head_yaw_deg,
    head_pitch_deg,
    eye_yaw_deg,
    eye_pitch_deg,
    final_yaw_deg,
    final_pitch_deg,
    fps=0.0,
    gaze_camera=None,
):
    """
    Render the premium HUD overlay onto img (BGR, modified in-place).
    Matches the reference dark-teal design with iris rings and gaze grid.
    """
    H, W = img.shape[:2]

    # Update sparkline history
    _YAW_HIST.append(float(final_yaw_deg))
    _PITCH_HIST.append(float(final_pitch_deg))

    # ── Scale factors ─────────────────────────────────────────────────────────
    S     = W / 1280.0
    M     = max(6,  int(10 * S))
    G     = max(4,  int(6  * S))
    pad   = max(6,  int(10 * S))

    fHDR  = max(0.32, 0.46 * S)
    fVAL  = max(0.40, 0.60 * S)
    fBIG  = max(0.58, 0.88 * S)
    fSM   = max(0.28, 0.40 * S)
    fTINY = max(0.22, 0.32 * S)
    fLBL  = max(0.26, 0.38 * S)

    lw1 = max(1, int(1 * S + 0.5))
    lw2 = max(1, int(2 * S + 0.5))

    # ─────────────────────────────────────────────────────────────────────────
    # 1.  TOP BANNER
    # ─────────────────────────────────────────────────────────────────────────
    BAN_H = max(62, int(H * 0.082))
    glowing_panel(img, M, M, W - M, M + BAN_H, accent=TEAL_ACC)

    zone_w = (W - 2 * M) // 3
    zones  = [M + pad, M + zone_w + pad, M + 2 * zone_w + pad]
    labels = ["HEAD", "EYE", "GAZE"]
    yaw_vals   = [head_yaw_deg,  eye_yaw_deg,   final_yaw_deg]
    pitch_vals = [head_pitch_deg, eye_pitch_deg, final_pitch_deg]
    cols       = [CHEAD, CGAZE, CZ]

    for zx, lbl, yv, pv, col in zip(zones, labels, yaw_vals, pitch_vals, cols):
        T(img, lbl,                            (zx, M + int(BAN_H * 0.24)), fHDR * 0.90, TEAL,  lw1)
        T(img, f"Yaw={yv:+.1f}  Pitch={pv:+.1f}", (zx, M + int(BAN_H * 0.60)), fVAL * 0.90, col,   lw1)

    # Status label in HEAD zone
    pose_col = COK if gaze_direction != "UNKNOWN" else CWARN
    T(img, f"Status: {head_horizontal}", (zones[0] + int(140 * S), M + int(BAN_H * 0.60)),
      fVAL * 0.85, pose_col, lw1)

    # Dividers between zones
    for zx in [M + zone_w, M + 2 * zone_w]:
        vline(img, zx, M + 8, M + BAN_H - 8, (70, 80, 80), 1)

    # ─────────────────────────────────────────────────────────────────────────
    # 2.  LEFT SIDEBAR
    # ─────────────────────────────────────────────────────────────────────────
    SIDEBAR_W = max(200, int(W * 0.195))
    sb_top    = M + BAN_H + G
    sb_bot    = H - M - max(26, int(H * 0.044))

    glowing_panel(img, M, sb_top, M + SIDEBAR_W, sb_bot, accent=CHEAD, alpha=0.38)

    row  = max(18, int((sb_bot - sb_top - 2 * pad) / 14.0))
    ry0  = sb_top + pad + int(fSM * 20)
    rx0  = M + pad
    rval = rx0 + int(80 * S)

    # ── READOUT header
    Ts(img, "READOUT", (rx0, ry0 - int(1.4 * fSM * 10)), fHDR, TEAL, lw1)

    def val_row(label, val_str, col, y):
        Ts(img, label,   (rx0, y),   fSM, WH,  lw1)
        Ts(img, val_str, (rval, y),  fSM, col, lw1)

    val_row("HEAD Yaw",   f"{head_yaw_deg:+.1f}°",   CHEAD, ry0 + row * 1)
    val_row("HEAD Pitch", f"{head_pitch_deg:+.1f}°", CHEAD, ry0 + row * 2)

    hline(img, rx0, M + SIDEBAR_W - pad, ry0 + int(row * 2.5), (55, 65, 65), 1)

    val_row("Eye Yaw",    f"{eye_yaw_deg:+.1f}°",    CGAZE, ry0 + row * 3)
    val_row("Eye Pitch",  f"{eye_pitch_deg:+.1f}°",  CGAZE, ry0 + row * 4)

    hline(img, rx0, M + SIDEBAR_W - pad, ry0 + int(row * 4.5), (55, 65, 65), 1)

    val_row("Gaze Yaw",   f"{final_yaw_deg:+.1f}°",  CZ,    ry0 + row * 5)
    val_row("Gaze Pitch", f"{final_pitch_deg:+.1f}°", CZ,   ry0 + row * 6)

    hline(img, rx0, M + SIDEBAR_W - pad, ry0 + int(row * 6.5), (55, 65, 65), 1)

    # ── FPS row with gear icon simulation
    fps_col = COK if fps >= 25 else CWARN
    Ts(img, "FPS",             (rx0,                ry0 + row * 7 + G), fLBL, DIM,     lw1)
    Ts(img, f"{fps:.1f}",      (rval,               ry0 + row * 7 + G), fVAL, fps_col, lw2)

    # ── Distance
    Ts(img, "DISTANCE",        (rx0,                ry0 + row * 9 + G), fLBL, DIM, lw1)
    Ts(img, f"{smooth_Z:.2f} m", (rval,             ry0 + row * 9 + G), fVAL, WH,  lw1)

    # ── STATUS  with cyan badge
    status_y = ry0 + row * 11 + G
    Ts(img, "STATUS",          (rx0, status_y),     fLBL, DIM, lw1)

    badge_label = f"[ACTIVE POSE] - {gaze_direction}"
    bsz = cv2.getTextSize(badge_label, cv2.FONT_HERSHEY_SIMPLEX, fSM * 0.90, 1)[0]
    bx1 = rx0
    by1 = status_y + int(row * 0.4)
    bx2 = min(M + SIDEBAR_W - pad, bx1 + bsz[0] + int(10 * S))
    by2 = by1 + bsz[1] + int(8 * S)
    panel(img, bx1 - 2, by1 - 2, bx2 + 2, by2 + 2, alpha=0.55, fill=30)
    # small cyan left strip
    vline(img, bx1, by1, by2, CGAZE, 2)
    Ts(img, badge_label, (bx1 + 4, by2 - int(4 * S)), fSM * 0.90, CGAZE, lw1)

    # ─────────────────────────────────────────────────────────────────────────
    # 3.  RIGHT ANALYTICS PANEL
    # ─────────────────────────────────────────────────────────────────────────
    AX_W  = max(240, int(W * 0.230))
    ax_x1 = W - M - AX_W
    ax_x2 = W - M

    glowing_panel(img, ax_x1, sb_top, ax_x2, sb_bot, accent=CZ)

    lx0 = ax_x1 + pad
    right_w = AX_W - 2 * pad
    cur_y   = sb_top + pad

    # ── "GAZE ANALYTICS" header ───────────────────────────────────────────
    Ts(img, "GAZE ANALYTICS", (lx0, cur_y + int(fHDR * 18)), fHDR, TEAL, lw1)
    cur_y += int(fHDR * 18) + G * 2

    # ── "Current Gaze Point" sub-label ───────────────────────────────────
    Ts(img, "Current Gaze Point", (lx0, cur_y + int(fSM * 14)), fSM, WH, lw1)
    cur_y += int(fSM * 14) + G

    # ── Gaze grid ─────────────────────────────────────────────────────────
    grid_size = min(int(right_w * 0.92), int((sb_bot - cur_y) * 0.40))
    grid_size = max(80, grid_size)
    draw_gaze_grid(img, lx0, cur_y, grid_size, final_yaw_deg, final_pitch_deg, S)
    cur_y += grid_size + G * 2

    # ── "Gaze Statistics" sub-label ───────────────────────────────────────
    Ts(img, "Gaze Statistics", (lx0, cur_y + int(fSM * 14)), fSM, WH, lw1)
    cur_y += int(fSM * 14) + G

    # ── 4 mini sparkline charts in 2×2 layout ─────────────────────────────
    spark_w = max(40, (right_w - G) // 2)
    spark_h = max(28, int((sb_bot - cur_y - G * 3 - int(fSM * 40)) * 0.48))
    spark_h = min(spark_h, 50)

    draw_sparkline(img, lx0,              cur_y, spark_w, spark_h,
                   _YAW_HIST,   CGAZE,  "Yaw")
    draw_sparkline(img, lx0 + spark_w + G, cur_y, spark_w, spark_h,
                   _PITCH_HIST, CHEAD,  "Pitch")

    cur_y += spark_h + G

    # second row — yaw abs + a flat line placeholder for elevation
    draw_sparkline(img, lx0,              cur_y, spark_w, spark_h,
                   [abs(v) for v in _YAW_HIST], (100, 180, 255), "Abs Yaw")
    draw_sparkline(img, lx0 + spark_w + G, cur_y, spark_w, spark_h,
                   [abs(v) for v in _PITCH_HIST], (100, 255, 200), "Abs Pitch")

    cur_y += spark_h + G * 2

    # ── "System Info" label ───────────────────────────────────────────────
    if cur_y + int(fSM * 40) < sb_bot - pad:
        Ts(img, "System Info", (lx0, cur_y + int(fSM * 14)), fSM, WH, lw1)
        cur_y += int(fSM * 14) + G
        draw_system_icons(img, lx0, cur_y + int(fSM * 14), int(14 * S), S)

    # ─────────────────────────────────────────────────────────────────────────
    # 4.  IRIS RINGS ON EYES  (replaces the ugly red dot)
    # ─────────────────────────────────────────────────────────────────────────
    # Estimate eye radius from inter-iris distance
    inter = max(20, int(math.hypot(rx - lx, ry - ly)))
    iris_r = max(8, int(inter * 0.28))

    draw_iris_ring(img, lx, ly, iris_r, eye_yaw_deg, eye_pitch_deg, S)
    draw_iris_ring(img, rx, ry, iris_r, eye_yaw_deg, eye_pitch_deg, S)


    # ─────────────────────────────────────────────────────────────────────────
    # 5.  CONTROLS BAR (bottom strip)
    # ─────────────────────────────────────────────────────────────────────────
    CTRL_H = max(22, int(H * 0.040))
    ctl_y1 = H - CTRL_H - 1
    panel(img, M, ctl_y1, W - M, H - 1, alpha=0.85, fill=12)
    hline(img, M, W - M, ctl_y1, _dim(TEAL_ACC, 0.55), 1)

    ctrl_icons = [
        ("-]", "[SPACE] Pause"),
        ("[-]", "[ESC] Stop"),
        ("[-", "[Q] Close"),
    ]
    ctxt = "  \u2192]  [SPACE] Pause    [-]  [ESC] Stop    [-  [Q] Close"
    # Simpler, reliable version:
    ctrl_str = "   [SPACE] Pause      [ESC] Stop      [Q] Close"
    ctsz = cv2.getTextSize(ctrl_str, cv2.FONT_HERSHEY_SIMPLEX, fTINY, 1)[0]
    Ts(img, ctrl_str,
       ((W - ctsz[0]) // 2, ctl_y1 + int(CTRL_H * 0.70)),
       fTINY, (150, 150, 150), lw1)

    return img