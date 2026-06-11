"""
gaze_estimator.py

Loads a pretrained gaze estimation model and returns:

yaw_deg
pitch_deg
gaze_vector

Usage:

estimator = GazeEstimator()

yaw, pitch, vec = estimator.predict(frame, eye_bbox)
"""

import cv2
import torch
import numpy as np

from torchvision import transforms


class GazeEstimator:

    def __init__(self):

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = torch.load(
            "models/L2CSNet_gaze360.pkl",
            map_location=self.device
        )

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])

    def gaze_vector(self, yaw_deg, pitch_deg):

        yaw = np.radians(yaw_deg)
        pitch = np.radians(pitch_deg)

        x = -np.cos(pitch) * np.sin(yaw)
        y = -np.sin(pitch)
        z = -np.cos(pitch) * np.cos(yaw)

        vec = np.array([x,y,z])

        norm = np.linalg.norm(vec)

        if norm > 0:
            vec /= norm

        return vec

    def predict(self, eye_crop):

        rgb = cv2.cvtColor(
            eye_crop,
            cv2.COLOR_BGR2RGB
        )

        tensor = self.transform(rgb)
        tensor = tensor.unsqueeze(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():

            output = self.model(tensor)

        yaw_deg = float(output[0][0])
        pitch_deg = float(output[0][1])

        vec = self.gaze_vector(
            yaw_deg,
            pitch_deg
        )

        return yaw_deg, pitch_deg, vec