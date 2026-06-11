import cv2
import torch
from l2cs import Pipeline

pipe = Pipeline(
    weights="models/L2CSNet_gaze360.pkl",
    arch="ResNet50",
    device=torch.device("cpu")
)

img = cv2.imread(
    "/home/robotics/Desktop/SPARC-Project/scripts/image.png"
)

if img is None:
    print("Failed to load image")
    exit()

print("Image shape:", img.shape)

result = pipe.step(img)

print("Yaw:", result.yaw)
print("Pitch:", result.pitch)
print("Faces detected:", len(result.yaw))