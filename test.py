from ultralytics import YOLO
from ultralytics.yolo.v8.detect.predict import DetectionPredictor

import cv2

model = YOLO("y8best.pt")

results = model.predict(source="iiith3.mp4", show=True)
print(results)