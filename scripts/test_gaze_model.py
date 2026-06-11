import cv2
import torch
from pathlib import Path

from l2cs import Pipeline


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

gaze_pipeline = Pipeline(
    weights=Path("models/L2CSNet_gaze360.pkl"),
    arch="ResNet50",
    device=device
)

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = gaze_pipeline.step(frame)

    print(results)

    cv2.imshow("Webcam", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()