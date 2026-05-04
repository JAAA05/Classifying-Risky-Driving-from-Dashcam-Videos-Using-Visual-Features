import cv2
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
from tqdm import tqdm
import os
import pandas as pd
import sys

# Ensure the parent directory (src) is in the path so we can import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.lane_experts import enhance_low_light, OpenCVLaneDetector, ensemble_lane_detector

class VideoProcessor:
    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"Initializing models on {self.device}...")
        
        # YOLOv8 for vehicle detection and tracking
        self.yolo = YOLO("yolov8n.pt")
        
        # DepthAnything V2 for depth estimation
        depth_device = self.device
        if self.device == "cuda":
            depth_device = 0 # Use first GPU
            
        self.depth_pipe = pipeline(
            task="depth-estimation", 
            model="depth-anything/Depth-Anything-V2-Small-hf", 
            device=depth_device
        )
        
        # YOLOP for lane detection
        self.yolop = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
        self.yolop.to(self.device).eval()

        # Traditional CV Lane Detector
        self.cv_lane_detector = OpenCVLaneDetector()
        
    def process_video(self, video_path, limit=None, show_progress=True):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if limit:
            total_frames = min(total_frames, limit)
            
        dt = 1.0 / (fps if fps > 0 else 30.0)
        
        features = []
        vehicle_classes = [2, 3, 5, 7] # car, motorcycle, bus, truck
        
        pbar = None
        if show_progress:
            pbar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}")
        
        frame_idx = 0
        prev_lane_center = None
        track_history = {} # track_id -> {depth: [val], pos: [val]}
        
        while cap.isOpened() and frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # 1. Depth Estimation
            depth_output = self.depth_pipe(pil_img)
            depth_map = depth_output['predicted_depth'].squeeze().cpu().numpy()
            
            # 2. YOLO Tracking
            results = self.yolo.track(frame, persist=True, classes=vehicle_classes, verbose=False, device=self.device)
            
            frame_ttcs = []
            frame_rel_vels = [] # relative velocity (change in depth)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box.astype(int)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    # ROI for depth
                    h_box, w_box = y2 - y1, x2 - x1
                    roi_y1 = max(0, cy - h_box // 4)
                    roi_y2 = min(height, cy + h_box // 4)
                    roi_x1 = max(0, cx - w_box // 4)
                    roi_x2 = min(width, cx + w_box // 4)
                    
                    if roi_y2 > roi_y1 and roi_x2 > roi_x1:
                        roi_depth = depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
                        curr_d = np.median(roi_depth)
                        
                        if track_id not in track_history:
                            track_history[track_id] = {'depth': [], 'pos': []}
                        
                        track_history[track_id]['depth'].append(curr_d)
                        track_history[track_id]['pos'].append((cx, cy))
                        
                        if len(track_history[track_id]['depth']) > 1:
                            d_prev = track_history[track_id]['depth'][-2]
                            diff = curr_d - d_prev
                            frame_rel_vels.append(diff / dt)
                            
                            if diff > 0:
                                ttc = (d_prev * dt) / diff
                                frame_ttcs.append(ttc)
            
            # 3. Lane Detection (Ensemble)
            # 3a. Pre-processing: Enhancement for low light
            frame_enhanced = frame_rgb
            avg_brightness = np.mean(frame_rgb)
            if avg_brightness < 80: # Dark frame heuristic
                frame_enhanced = enhance_low_light(frame) # uses BGR
                frame_enhanced = cv2.cvtColor(frame_enhanced, cv2.COLOR_BGR2RGB)

            # 3b. YOLOP Expert
            img_yolop = cv2.resize(frame_enhanced, (640, 640))
            img_yolop_tensor = torch.from_numpy(img_yolop).permute(2, 0, 1).float().to(self.device) / 255.0
            img_yolop_tensor = img_yolop_tensor.unsqueeze(0)
            
            with torch.no_grad():
                _, _, ll_seg_out = self.yolop(img_yolop_tensor)
            
            yolop_mask = torch.softmax(ll_seg_out, dim=1)[0, 1, :, :].cpu().numpy()
            yolop_mask = (yolop_mask > 0.5).astype(np.uint8)

            # 3c. OpenCV Expert
            img_cv = cv2.resize(frame_enhanced, (640, 640))
            img_cv_bgr = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            cv_mask = self.cv_lane_detector.get_lane_mask(img_cv_bgr)
            cv_mask = (cv_mask > 0).astype(np.uint8)

            # 3d. Voting / Ensemble
            lane_mask = ensemble_lane_detector(yolop_mask, cv_mask)
            
            # Lane center analysis
            bottom_slice = lane_mask[400:, :]
            lane_pixels = np.where(bottom_slice > 0)
            
            lane_departure = 0
            has_lane = 1
            if len(lane_pixels[1]) > 0:
                curr_lane_center = np.mean(lane_pixels[1]) / 640.0
                if prev_lane_center is not None:
                    lane_departure = curr_lane_center - prev_lane_center # lateral velocity
                prev_lane_center = curr_lane_center
            else:
                curr_lane_center = 0.5
                has_lane = 0
                prev_lane_center = None # reset
            
            # 4. Aggregate Features
            min_ttc = min(frame_ttcs) if frame_ttcs else 100.0
            max_rel_vel = max(frame_rel_vels) if frame_rel_vels else 0.0
            
            # 5. Proof of Concept: "WHY" Label
            why_label = "normal"
            if min_ttc < 1.5:
                why_label = "tailgating"
            elif abs(lane_departure) > 0.1:
                why_label = "lane_departure"
            elif max_rel_vel > 20.0:
                why_label = "aggressive_closing"
            
            features.append({
                'frame': frame_idx,
                'min_ttc': min_ttc,
                'max_rel_vel': max_rel_vel,
                'lane_departure': lane_departure,
                'lane_center': curr_lane_center,
                'has_lane': has_lane,
                'has_objects': 1 if frame_ttcs else 0,
                'why': why_label
            })
            
            frame_idx += 1
            if pbar:
                pbar.update(1)
            
        cap.release()
        if pbar:
            pbar.close()
        
        # Convert to DataFrame and apply smoothing
        df = pd.DataFrame(features)
        if not df.empty:
            df['min_ttc_smooth'] = df['min_ttc'].rolling(window=5, min_periods=1).mean()
            df['lane_departure_smooth'] = df['lane_departure'].rolling(window=5, min_periods=1).mean()
            
        return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    processor = VideoProcessor()
    df = processor.process_video(args.video, limit=args.limit)
    df.to_csv(args.output, index=False)
    print(f"Saved features to {args.output}")
