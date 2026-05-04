import torch
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image
import cv2
import numpy as np

def test_models():
    print("Testing YOLOv8...")
    yolo_model = YOLO("yolov8n.pt")
    print("YOLOv8 loaded.")

    print("Testing DepthAnything V2...")
    # Using transformers pipeline for depth
    depth_pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=0 if torch.cuda.is_available() else -1)
    print("DepthAnything V2 loaded.")

    print("Testing YOLOP via Torch Hub...")
    try:
        yolop_model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
        yolop_model.eval()
        if torch.cuda.is_available():
            yolop_model.cuda()
        print("YOLOP loaded.")
    except Exception as e:
        print(f"Error loading YOLOP: {e}")

if __name__ == "__main__":
    test_models()
