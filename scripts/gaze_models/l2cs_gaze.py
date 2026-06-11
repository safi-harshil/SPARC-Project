"""
L2CS-Net Gaze Estimator
-----------------------
Wraps the L2CS Pipeline to return (yaw_deg, pitch_deg) per frame.

Usage:
    estimator = L2CSGazeEstimator("models/L2CSNet_gaze360.pkl")
    result = estimator.estimate(color_bgr_frame)
    if result:
        yaw_deg, pitch_deg = result
"""

from pathlib import Path
import numpy as np
import torch
import cv2
import traceback

from l2cs import Pipeline


class L2CSGazeEstimator:
    """
    Wraps the L2CS Pipeline.

    Parameters
    ----------
    model_path : str | Path
        Path to L2CSNet_gaze360.pkl checkpoint.
    arch : str
        Backbone architecture. "ResNet50" is the standard checkpoint.
    use_face_crop : bool
        If True, pass a face-cropped image instead of the full frame.
        Face crops improve accuracy significantly. Requires a face bbox.
    """

    def __init__(
        self,
        model_path: str,
        arch: str = "ResNet50",
        use_face_crop: bool = False,
    ):
        self.use_face_crop = use_face_crop
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"[L2CS] Loading model from {model_path} on {self.device}")

        self.pipeline = Pipeline(
            weights=Path(model_path),
            arch=arch,
            device=self.device,
        )

        print("[L2CS] Model loaded successfully")

    def estimate(self, frame_bgr: np.ndarray, face_bbox=None):
        """
        Estimate gaze from a BGR frame (as returned by OpenCV / RealSense).

        Parameters
        ----------
        frame_bgr : np.ndarray
            Full color frame (H x W x 3, BGR uint8).
        face_bbox : tuple | None
            Optional (x1, y1, x2, y2) bounding box for face crop.
            Only used when use_face_crop=True.

        Returns
        -------
        (yaw_deg, pitch_deg) : tuple[float, float] | None
            Gaze angles in degrees. None if no face detected.
            - yaw_deg  > 0  →  looking RIGHT
            - yaw_deg  < 0  →  looking LEFT
            - pitch_deg > 0  →  looking DOWN
            - pitch_deg < 0  →  looking UP
        """
        if self.use_face_crop and face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            # Add 20% padding around the face crop
            h, w = frame_bgr.shape[:2]
            pad_x = int((x2 - x1) * 0.2)
            pad_y = int((y2 - y1) * 0.2)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            input_img = frame_bgr[y1:y2, x1:x2]
        else:
            input_img = frame_bgr

        try:
            results = self.pipeline.step(input_img)

        except Exception:
            traceback.print_exc()
            return None

        if results is None or len(results.yaw) == 0:
            return None

        # L2CS returns radians; convert to degrees
        yaw_deg   = float(results.yaw[0])   * (180.0 / np.pi)
        pitch_deg = float(results.pitch[0]) * (180.0 / np.pi)

        return yaw_deg, pitch_deg

    def estimate_with_vector(self, frame_bgr: np.ndarray, face_bbox=None):
        """
        Same as estimate() but also returns the 3D unit gaze vector
        in camera coordinates.

        Returns
        -------
        (yaw_deg, pitch_deg, gaze_vec_3d) | None
            gaze_vec_3d : np.ndarray shape (3,)
                Unit vector [x, y, z] in camera space.
                x = right, y = down, z = into screen (away from camera).
        """
        result = self.estimate(frame_bgr, face_bbox)

        if result is None:
            return None

        yaw_deg, pitch_deg = result

        yaw_rad   = np.radians(yaw_deg)
        pitch_rad = np.radians(pitch_deg)

        # Spherical → Cartesian (camera convention: +Z into scene)
        x = np.sin(yaw_rad) * np.cos(pitch_rad)
        y = np.sin(pitch_rad)
        z = np.cos(yaw_rad) * np.cos(pitch_rad)

        gaze_vec = np.array([x, y, z], dtype=np.float64)
        gaze_vec /= np.linalg.norm(gaze_vec)  # ensure unit length

        return yaw_deg, pitch_deg, gaze_vec