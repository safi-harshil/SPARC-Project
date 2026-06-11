# eye_tracking_worker_l2cs.py
# ---------------------------
# RealSense eye-tracking worker using L2CS-Net for ML gaze estimation.


# ═══════════════════════════════════════════════════════════════════
# APPROACH / THEORY  (for presentation to professor)
# ═══════════════════════════════════════════════════════════════════


# The gaze vector is a 3D unit vector in CAMERA space that points in
# the direction the person is looking.  It is built in two stages:


#   STAGE 1 — Head pose  (MediaPipe FaceMesh + OpenCV solvePnPRansac)
#   ─────────────────────────────────────────────────────────────────
#   Six stable 2D facial landmarks (nose tip, chin, left/right eye
#   corners, left/right mouth corners) are matched against a known
#   generic 3-D face model.  solvePnPRansac returns a rotation vector
#   rvec and translation vector tvec.
#   cv2.Rodrigues(rvec) → rotation matrix R (3×3).
#   R transforms vectors FROM face/model space INTO camera space.
#   cv2.RQDecomp3x3(R) → Euler angles (head_yaw, head_pitch).


#   STAGE 2 — Eye gaze  (L2CS-Net deep-learning model)
#   ─────────────────────────────────────────────────────────────────
#   L2CS-Net (ResNet50, trained on Gaze360) takes a face-crop image
#   and regresses two angles: eye_yaw and eye_pitch.
#   These angles describe where the *eyeball* is pointing relative to
#   the face/head coordinate frame, NOT in camera space.


#   STAGE 3 — Combining into a camera-space gaze vector  ← KEY FIX
#   ─────────────────────────────────────────────────────────────────
#   WRONG approach (previous code):
#       final_yaw   = head_yaw + eye_yaw          ← plain angle addition
#       final_pitch = head_pitch + eye_pitch       ← wrong, double-counts


#   WHY it is wrong: L2CS is trained on images of faces at all head
#   orientations; its output is in face space, NOT in camera space.
#   Adding head angles on top of L2CS angles double-applies the head
#   rotation and gives nonsense whenever the person turns their head.


#   CORRECT approach (this code):
#       1. Build a unit gaze vector in FACE space from L2CS angles:
#              gaze_face = [ -sin(yaw)·cos(pitch),
#                             -sin(pitch),
#                              cos(yaw)·cos(pitch) ]


#       2. Rotate it into CAMERA space using the PnP rotation matrix R:
#              gaze_camera = R @ gaze_face


#       3. Extract display angles from the camera-space vector:
#              final_yaw   = atan2(-gaze_camera[0],  gaze_camera[2])
#              final_pitch = arcsin(-gaze_camera[1])


#   This is the standard approach used in academic gaze-estimation
#   papers (e.g. Gaze360, ETH-XGaze) and matches what the article
#   https://medium.com/@olga.mindlina describes when it explains that
#   the gaze vector must live in the CAMERA coordinate system.


#   STAGE 4 — 3-D eye POSITION  (RealSense depth)
#   ─────────────────────────────────────────────────────────────────
#   The mid-point of the two iris centres is back-projected to 3-D
#   using the RealSense depth frame and camera intrinsics:
#       X = (u - cx) · depth / fx
#       Y = (v - cy) · depth / fy
#       Z = depth   (metres, along optical axis)


#   All position values are smoothed with an exponential moving average
#   (alpha=0.8) because the depth sensor is noisy.
#   Gaze angles are smoothed separately (alpha=0.6, lighter) so that
#   quick eye movements remain visible.


# ═══════════════════════════════════════════════════════════════════
# Pipeline per frame:
#   1. MediaPipe FaceMesh      → 468 landmarks (iris-refined)
#   2. solvePnPRansac           → rotation matrix R, head yaw/pitch
#   3. get_face_bbox()          → tight face crop with padding
#   4. L2CS-Net                 → eye_yaw, eye_pitch (face space)
#   5. EMA smooth L2CS output
#   6. Build gaze_face vector from L2CS angles
#   7. gaze_camera = R @ gaze_face  (camera-space unit vector)  ← KEY
#   8. final_yaw/pitch from gaze_camera for display / CSV
#   9. RealSense depth          → 3-D eye position (metres)
#  10. EMA smooth 3-D position
#  11. draw_hud()               → annotated overlay
#  12. CSV logger               → all values per frame
# """


import time
from pathlib import Path
import traceback


import cv2
import numpy as np
import mediapipe as mp


from logger_utils import get_eye_logger
from eye_tracking_ui import draw_hud
from control_flags import stop_event, pause_event
from gaze_models.l2cs_gaze import L2CSGazeEstimator




# ── Model path ────────────────────────────────────────────────────────────────
MODEL_PATH = str(Path(__file__).parent / "models/L2CSNet_gaze360.pkl")
print(f"[DEBUG] Model path: {MODEL_PATH}")


# ── MediaPipe iris landmark indices ───────────────────────────────────────────
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


# ── 3-D face model points (mm, generic head) for PnP ─────────────────────────
# These are world-frame 3-D coordinates of 6 stable landmarks.
# The origin is at the nose tip, Z points towards the camera.
FACE_MODEL_POINTS = np.array([
   ( 0.0,    0.0,   0.0),   # nose tip        lm 1
   ( 0.0,  -63.6, -12.5),   # chin            lm 152
   (-43.3,  32.7, -26.0),   # left eye outer  lm 33
   ( 43.3,  32.7, -26.0),   # right eye outer lm 263
   (-28.9, -28.9, -24.1),   # left mouth      lm 61
   ( 28.9, -28.9, -24.1),   # right mouth     lm 291
], dtype=np.float64)


dist_coeffs   = np.zeros((4, 1), dtype=np.float64)
camera_matrix = None   # built from RealSense intrinsics on first frame




# ── Helpers ───────────────────────────────────────────────────────────────────


def pixel_to_3d(u, v, depth_m, fx, fy, cx, cy):
   """Back-project a pixel + depth to a 3-D point in camera space."""
   return (
       (u - cx) * depth_m / fx,
       (v - cy) * depth_m / fy,
       depth_m,
   )




def get_iris_center(face_landmarks, iris_indices, w, h):
   """Return (centre_xy, list_of_points) for a set of iris landmark indices."""
   pts = [
       (int(face_landmarks.landmark[i].x * w),
        int(face_landmarks.landmark[i].y * h))
       for i in iris_indices
   ]
   return np.mean(pts, axis=0).astype(np.int32), pts




def get_face_bbox(face_landmarks, w, h, padding=0.15):
   """Tight bounding box around all face landmarks with fractional padding."""
   xs = [lm.x * w for lm in face_landmarks.landmark]
   ys = [lm.y * h for lm in face_landmarks.landmark]
   x1, x2 = int(min(xs)), int(max(xs))
   y1, y2 = int(min(ys)), int(max(ys))
   px = int((x2 - x1) * padding)
   py = int((y2 - y1) * padding)
   return (
       max(0, x1 - px), max(0, y1 - py),
       min(w, x2 + px), min(h, y2 + py),
   )




def l2cs_to_face_vector(yaw_deg, pitch_deg):
   """
   Convert L2CS yaw/pitch angles (degrees, face-space) to a unit 3-D vector
   in face/head coordinate space.


   Convention (matches L2CS-Net / Gaze360 training convention):
       +yaw   → looking right  (in face frame)
       +pitch → looking down   (in face frame)
       forward (straight ahead) = +Z axis of face frame


   Returns np.ndarray shape (3,) — a normalised unit vector.
   """
   yaw_r   = np.deg2rad(yaw_deg)
   pitch_r = np.deg2rad(pitch_deg)


   gx = -np.sin(yaw_r) * np.cos(pitch_r)   # X: left is negative
   gy = -np.sin(pitch_r)                    # Y: up is negative
   gz =  np.cos(yaw_r)  * np.cos(pitch_r)  # Z: forward is positive


   vec = np.array([gx, gy, gz], dtype=np.float64)
   vec /= np.linalg.norm(vec) + 1e-9        # safety normalise
   return vec




def camera_vector_to_angles(gaze_camera):
   """
   Convert a unit gaze vector in CAMERA space to yaw and pitch in degrees.


   yaw   = rotation around camera Y axis (+ = right)
   pitch = rotation around camera X axis (+ = down)
   """
   gx, gy, gz = gaze_camera
   yaw_deg   = np.degrees(np.arctan2(-gx,  gz))
   pitch_deg = np.degrees(np.arcsin( np.clip(-gy, -1.0, 1.0)))
   return float(yaw_deg), float(pitch_deg)




# ── Worker ────────────────────────────────────────────────────────────────────


def eye_tracking_worker(cam_label, q_eye, out_dir):


   global camera_matrix


   print(f"[INFO] Eye tracking (L2CS, corrected gaze vector) started for {cam_label}")


   # ── Directories ───────────────────────────────────────────────────────────
   cam_dir = Path(out_dir) / cam_label
   cam_dir.mkdir(parents=True, exist_ok=True)
   eye_dir = cam_dir / "eye_tracking"
   eye_dir.mkdir(parents=True, exist_ok=True)


   # ── Logger ────────────────────────────────────────────────────────────────
   logger = get_eye_logger(cam_dir, flush_sec=5)
   logger.info("Eye tracking (L2CS, corrected gaze vector) started")


   # ── CSV ───────────────────────────────────────────────────────────────────
   csv_path = eye_dir / "eye_tracking.csv"
   csv_file = open(csv_path, "w", buffering=1)
   csv_file.write(
       "timestamp_ns,frame_id,"
       "left_iris_x_px,left_iris_y_px,"
       "right_iris_x_px,right_iris_y_px,"
       "X,Y,Z,"
       "smooth_X,smooth_Y,smooth_Z,"
       "head_horizontal,head_vertical,"
       "gaze_direction,"
       "final_yaw_deg,final_pitch_deg,"
       "head_yaw_deg,head_pitch_deg,"
       "eye_yaw_deg,eye_pitch_deg,"
       "gaze_vec_x,gaze_vec_y,gaze_vec_z\n"
       # gaze_vec_x/y/z = camera-space unit gaze vector (the main deliverable)
   )
   csv_file.flush()


   # ── MediaPipe FaceMesh ────────────────────────────────────────────────────
   mp_face_mesh = mp.solutions.face_mesh
   face_mesh = mp_face_mesh.FaceMesh(
       max_num_faces=1,
       refine_landmarks=True,        # enables iris landmarks 468-477
       min_detection_confidence=0.5,
       min_tracking_confidence=0.5,
   )


   # ── L2CS-Net ──────────────────────────────────────────────────────────────
   gaze_model = L2CSGazeEstimator(
       model_path=MODEL_PATH,
       arch="ResNet50",
       use_face_crop=True,
   )
   print(f"[INFO] L2CS model loaded for {cam_label}")


   # ── Per-frame EMA state ───────────────────────────────────────────────────
   prev_X = prev_Y = prev_Z = None
   alpha_pos  = 0.8   # heavy smoothing for noisy depth
   alpha_gaze = 0.6   # lighter smoothing so eye movements stay responsive


   # Smoothed L2CS angles (face-space) — smoothing happens BEFORE rotation
   prev_eye_yaw   = None
   prev_eye_pitch = None


   t_prev      = time.time()
   fps_display = 0.0
   flush_counter = 0


   # ─────────────────────────────────────────────────────────────────────────
   while not stop_event.is_set():


       if pause_event.is_set():
           time.sleep(0.02)
           continue


       try:
           pkt = q_eye.get(timeout=0.1)
       except Exception:
           continue


       if pkt is None:
           continue


       # ── Live FPS ──────────────────────────────────────────────────────────
       t_now       = time.time()
       fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
       t_prev      = t_now


       try:
           color_img   = pkt.color.copy()
           depth_img   = pkt.depth
           fx, fy      = pkt.fx, pkt.fy
           cx, cy      = pkt.cx, pkt.cy
           depth_scale = pkt.depth_scale_m


           if camera_matrix is None:
               camera_matrix = np.array(
                   [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                   dtype=np.float64,
               )


           img_h, img_w = color_img.shape[:2]


           # ── MediaPipe FaceMesh ────────────────────────────────────────────
           rgb    = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
           mp_res = face_mesh.process(rgb)


           if not mp_res.multi_face_landmarks:
               continue


           face_landmarks = mp_res.multi_face_landmarks[0]


           # ── Stage 1: Head pose via solvePnPRansac ─────────────────────────
           # Map 6 stable face landmarks to their known 3-D model positions.
           # solvePnPRansac returns rvec (rotation vector) and tvec.
           # Rodrigues(rvec) → R  which rotates face-space → camera-space.
           image_points = np.array([
               [face_landmarks.landmark[1].x   * img_w, face_landmarks.landmark[1].y   * img_h],
               [face_landmarks.landmark[152].x * img_w, face_landmarks.landmark[152].y * img_h],
               [face_landmarks.landmark[33].x  * img_w, face_landmarks.landmark[33].y  * img_h],
               [face_landmarks.landmark[263].x * img_w, face_landmarks.landmark[263].y * img_h],
               [face_landmarks.landmark[61].x  * img_w, face_landmarks.landmark[61].y  * img_h],
               [face_landmarks.landmark[291].x * img_w, face_landmarks.landmark[291].y * img_h],
           ], dtype=np.float64)


           ok, rvec, tvec, _ = cv2.solvePnPRansac(
               FACE_MODEL_POINTS, image_points,
               camera_matrix, dist_coeffs,
               flags=cv2.SOLVEPNP_ITERATIVE,
           )
           if not ok:
               continue


           # R rotates vectors FROM face/model space INTO camera space
           R, _ = cv2.Rodrigues(rvec)


           # Head Euler angles for display / CSV (not used in gaze calculation)
           angles, *_ = cv2.RQDecomp3x3(R)
           head_pitch_deg = angles[0]   # + = looking down
           head_yaw_deg   = angles[1]   # + = looking right


           # Draw RGB pose axes on nose tip
           nose_pt   = tuple(image_points[0].astype(int))
           imgpts, _ = cv2.projectPoints(
               np.float32([[80,0,0],[0,80,0],[0,0,80]]),
               rvec, tvec, camera_matrix, dist_coeffs,
           )
           imgpts = imgpts.reshape(-1, 2).astype(int)
           cv2.line(color_img, nose_pt, tuple(imgpts[0]), (0,   0, 255), 2)  # X red
           cv2.line(color_img, nose_pt, tuple(imgpts[1]), (0, 255,   0), 2)  # Y green
           cv2.line(color_img, nose_pt, tuple(imgpts[2]), (255,  0,   0), 2) # Z blue

           # ── Iris centres ──────────────────────────────────────────────────
           left_center,  _ = get_iris_center(face_landmarks, LEFT_IRIS,  img_w, img_h)
           right_center, _ = get_iris_center(face_landmarks, RIGHT_IRIS, img_w, img_h)
           lx, ly = left_center
           rx, ry = right_center
           eye_x  = (lx + rx) // 2
           eye_y  = (ly + ry) // 2


           # ── Stage 2: L2CS-Net eye gaze (face space) ───────────────────────
           face_bbox   = get_face_bbox(face_landmarks, img_w, img_h, padding=0.15)
           gaze_result = gaze_model.estimate_with_vector(color_img, face_bbox)


           if gaze_result is not None:
               raw_yaw, raw_pitch, _ = gaze_result   # _ = L2CS raw vec (face space, not used)
           else:
               raw_yaw   = 0.0
               raw_pitch = 0.0


           # EMA smooth the raw L2CS face-space angles
           # (smoothing before rotation keeps the math clean)
           if prev_eye_yaw is None:
               eye_yaw_deg   = raw_yaw
               eye_pitch_deg = raw_pitch
           else:
               eye_yaw_deg   = alpha_gaze * prev_eye_yaw   + (1.0 - alpha_gaze) * raw_yaw
               eye_pitch_deg = alpha_gaze * prev_eye_pitch + (1.0 - alpha_gaze) * raw_pitch


           prev_eye_yaw   = eye_yaw_deg
           prev_eye_pitch = eye_pitch_deg


           # ── Stage 3: Rotate gaze into camera space ────────────────────────
           #
           # KEY FIX vs previous code:
           #   OLD (wrong): final_yaw   = head_yaw   + eye_yaw
           #                final_pitch = head_pitch + eye_pitch
           #   This double-counts head rotation — L2CS already "knows" the
           #   head pose because it sees the face crop.
           #
           #   CORRECT:
           #   1. Build a unit vector from the L2CS angles (face space)
           #   2. Apply R (from PnP) to rotate it into camera space
           #   3. Derive final angles from the camera-space vector
           #
           gaze_face   = l2cs_to_face_vector(eye_yaw_deg, eye_pitch_deg)
           gaze_camera = R @ gaze_face                         # camera-space unit vector
           gaze_camera = gaze_camera / (np.linalg.norm(gaze_camera) + 1e-9)


           final_yaw_deg, final_pitch_deg = camera_vector_to_angles(gaze_camera)


           # Debug print
           print(
               f"[{cam_label}] "
               f"head yaw={head_yaw_deg:+.1f}° pitch={head_pitch_deg:+.1f}° | "
               f"eye  yaw={eye_yaw_deg:+.1f}° pitch={eye_pitch_deg:+.1f}° | "
               f"gaze_cam=({gaze_camera[0]:+.3f},{gaze_camera[1]:+.3f},{gaze_camera[2]:+.3f}) | "
               f"final yaw={final_yaw_deg:+.1f}° pitch={final_pitch_deg:+.1f}° | "
               f"fps={fps_display:.1f}"
           )


           # ── Stage 4: RealSense depth → 3-D eye position ───────────────────
           if (
               eye_x < 0 or eye_y < 0
               or eye_x >= depth_img.shape[1]
               or eye_y >= depth_img.shape[0]
           ):
               continue


           patch = depth_img[
               max(0, eye_y - 2):min(depth_img.shape[0], eye_y + 3),
               max(0, eye_x - 2):min(depth_img.shape[1], eye_x + 3),
           ]
           valid = patch[patch > 0]
           if len(valid) == 0:
               continue


           depth_raw = np.median(valid)
           if depth_raw == 0:
               continue


           depth_m = float(depth_raw) * float(depth_scale)
           if not (0.15 <= depth_m <= 2.0):
               continue


           X, Y, Z = pixel_to_3d(eye_x, eye_y, depth_m, fx, fy, cx, cy)


           if prev_X is None:
               smooth_X, smooth_Y, smooth_Z = X, Y, Z
           else:
               smooth_X = alpha_pos * prev_X + (1.0 - alpha_pos) * X
               smooth_Y = alpha_pos * prev_Y + (1.0 - alpha_pos) * Y
               smooth_Z = alpha_pos * prev_Z + (1.0 - alpha_pos) * Z


           prev_X, prev_Y, prev_Z = smooth_X, smooth_Y, smooth_Z


           # ── Direction labels (from camera-space final angles) ──────────────
           head_horizontal = (
               "LEFT"  if head_yaw_deg   < -15 else
               "RIGHT" if head_yaw_deg   >  15 else "CENTER"
           )
           head_vertical = (
               "UP"    if head_pitch_deg < -10 else
               "DOWN"  if head_pitch_deg >  10 else "CENTER"
           )
           h_dir = (
               "LEFT"  if final_yaw_deg   < -12 else
               "RIGHT" if final_yaw_deg   >  12 else "CENTER"
           )
           v_dir = (
               "UP"    if final_pitch_deg < -10 else
               "DOWN"  if final_pitch_deg >  10 else "CENTER"
           )


           if   h_dir == "CENTER" and v_dir == "CENTER": gaze_direction = "CENTER"
           elif h_dir == "CENTER":                        gaze_direction = v_dir
           elif v_dir == "CENTER":                        gaze_direction = h_dir
           else:                                          gaze_direction = f"{v_dir}_{h_dir}"


           # ── HUD ───────────────────────────────────────────────────────────
           try:
               draw_hud(
                   color_img,
                   smooth_X, smooth_Y, smooth_Z,
                   head_horizontal, head_vertical,
                   gaze_direction,
                   lx, ly, rx, ry,
                   head_yaw_deg, head_pitch_deg,
                   eye_yaw_deg, eye_pitch_deg,
                   final_yaw_deg, final_pitch_deg,
                   fps=fps_display,
                   gaze_camera=gaze_camera,
               )
           except Exception as e:
               print("[HUD ERROR]", e)
               traceback.print_exc()


           pkt.eye_overlay = color_img


           # ── Save every 5th frame ──────────────────────────────────────────
           if pkt.frame_id % 5 == 0:
               fp = eye_dir / f"eye_{pkt.frame_id:06d}.jpg"
               if not cv2.imwrite(str(fp), color_img):
                   print(f"[SAVE ERROR] {fp}")


           # ── Terminal log ──────────────────────────────────────────────────
           msg = (
               f"Eye3D X={smooth_X:.3f} Y={smooth_Y:.3f} Z={smooth_Z:.3f} | "
               f"head yaw={head_yaw_deg:+.1f}° pitch={head_pitch_deg:+.1f}° | "
               f"eye  yaw={eye_yaw_deg:+.1f}° pitch={eye_pitch_deg:+.1f}° | "
               f"final yaw={final_yaw_deg:+.1f}° pitch={final_pitch_deg:+.1f}° | "
               f"gaze_cam=({gaze_camera[0]:+.3f},{gaze_camera[1]:+.3f},{gaze_camera[2]:+.3f}) | "
               f"fps={fps_display:.1f}"
           )
           logger.info(msg)
           logger.periodic_flush()


           # gaze_vec_x/y/z = the CORRECT camera-space unit gaze vector
           csv_file.write(
               f"{pkt.t_ns},{pkt.frame_id},"
               f"{lx},{ly},{rx},{ry},"
               f"{X:.5f},{Y:.5f},{Z:.5f},"
               f"{smooth_X:.5f},{smooth_Y:.5f},{smooth_Z:.5f},"
               f"{head_horizontal},{head_vertical},"
               f"{gaze_direction},"
               f"{final_yaw_deg:.2f},{final_pitch_deg:.2f},"
               f"{head_yaw_deg:.2f},{head_pitch_deg:.2f},"
               f"{eye_yaw_deg:.3f},{eye_pitch_deg:.3f},"
               f"{gaze_camera[0]:.4f},{gaze_camera[1]:.4f},{gaze_camera[2]:.4f}\n"
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
