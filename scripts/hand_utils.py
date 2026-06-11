#!/usr/bin/env python3
from typing import Optional, Dict, List, Tuple
import numpy as np
import cv2
import mediapipe as mp

from tunables import (
    WRIST_ID, NUM_LANDMARKS, WRIST_SEP_PX, COLOR_LEFT, COLOR_RIGHT, COLOR_TEXT,
    HANDS, KEYPOINTS_MOV
)

mp_hands = mp.solutions.hands

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

def collapse_overlap_mediapipe(dets, frame_id, logger):
    """If both detected but wrists too close, keep higher score."""
    if len(dets) == 2:
        w0 = dets[0]['pts'][WRIST_ID]; w1 = dets[1]['pts'][WRIST_ID]
        d = float(np.hypot(w0[0]-w1[0], w0[1]-w1[1]))
        if d < WRIST_SEP_PX:
            keep_idx = 0 if dets[0]['score'] >= dets[1]['score'] else 1
            logger.warn(f"Frame {frame_id}: wrist distance {d:.1f}px < {WRIST_SEP_PX}px → collapse 2→1 (keep {keep_idx}).")
            return [dets[keep_idx]]
    return dets

def collapse_overlap_raw(raw_pts, prev_L, prev_R, logger, frame_id):
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
    cv2.putText(img, f"id:{frame_id}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
    return img
