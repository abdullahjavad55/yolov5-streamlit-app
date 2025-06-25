# yolo_detector.py
import torch
import cv2
import numpy as np
from PIL import Image
import os

# Load the YOLOv5 model from Ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_image(image_path):
    results = model(image_path)
    results.render()
    return Image.fromarray(results.ims[0])

def detect_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = "output_detected.mp4"
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        results.render()
        frame = results.ims[0]
        out.write(frame)

    cap.release()
    out.release()
    return output_path
