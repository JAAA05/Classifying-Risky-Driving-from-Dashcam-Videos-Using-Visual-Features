import cv2
import torch
import numpy as np
import pandas as pd
import joblib
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
from tqdm import tqdm
import os
import time
import sys
import os

# Ensure the parent directory (src) is in the path so we can import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.lane_experts import enhance_low_light, OpenCVLaneDetector, ensemble_lane_detector

# Add models directory to path for feature extraction logic
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
try:
    from train_classifier import extract_video_features
except ImportError:
    # Fallback if structure is different
    def extract_video_features(df):
        # Basic fallback if import fails
        return {}

class Visualizer:
    def __init__(self, device=None, model_path="src/models/classifier.joblib"):
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
        
        # YOLOv8
        self.yolo = YOLO("yolov8n.pt")
        
        # DepthAnything V2
        depth_device = self.device
        if self.device == "cuda":
            depth_device = 0
        self.depth_pipe = pipeline(
            task="depth-estimation", 
            model="depth-anything/Depth-Anything-V2-Small-hf", 
            device=depth_device
        )
        
        # YOLOP
        self.yolop = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
        self.yolop.to(self.device).eval()

        # Traditional CV Lane Detector
        self.cv_lane_detector = OpenCVLaneDetector()
        
        # Classifier
        self.clf = None
        if os.path.exists(model_path):
            try:
                self.clf = joblib.load(model_path)
                print(f"Loaded classifier from {model_path}")
            except Exception as e:
                print(f"Error loading classifier: {e}")
        else:
            print(f"Classifier not found at {model_path}")

    def draw_gauge(self, img, value, label, pos, size=(200, 20), color=(255, 255, 255)):
        x, y = pos
        w, h = size
        cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
        
        # Center line
        cv2.line(img, (x + w//2, y), (x + w//2, y + h), (200, 200, 200), 1)
        
        # Value marker
        marker_x = int(x + w//2 + (value * w//2))
        marker_x = max(x, min(x + w, marker_x))
        cv2.line(img, (marker_x, y - 5), (marker_x, y + h + 5), (0, 0, 255), 2)
        
        cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def process_video(self, video_path, output_path=None, show=False, limit=None):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if limit:
            total_frames = min(total_frames, limit)
            
        dt = 1.0 / (fps if fps > 0 else 30.0)
        
        # Dashboard width
        dash_w = 350
        full_width = width + dash_w
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (full_width, height))
        else:
            out = None
            
        vehicle_classes = [2, 3, 5, 7]
        pbar = tqdm(total=total_frames, desc=f"Visualizing {os.path.basename(video_path)}")
        
        frame_idx = 0
        prev_lane_center = None
        track_history = {}
        all_features = []
        
        while cap.isOpened() and frame_idx < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            display_frame = frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            # 1. Depth Estimation
            depth_output = self.depth_pipe(pil_img)
            depth_map = depth_output['predicted_depth'].squeeze().cpu().numpy()
            
            # 2. YOLO Tracking
            results = self.yolo.track(frame, persist=True, classes=vehicle_classes, verbose=False, device=self.device)
            
            frame_ttcs = []
            frame_rel_vels = []
            closest_vehicle_pos = None
            min_ttc = 100.0
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().numpy()
                
                # Resize depth for ROI sampling
                depth_map_full = cv2.resize(depth_map, (width, height))
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box.astype(int)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    h_box, w_box = y2 - y1, x2 - x1
                    roi_y1 = max(0, cy - h_box // 4)
                    roi_y2 = min(height, cy + h_box // 4)
                    roi_x1 = max(0, cx - w_box // 4)
                    roi_x2 = min(width, cx + w_box // 4)
                    
                    if roi_y2 > roi_y1 and roi_x2 > roi_x1:
                        roi_depth = depth_map_full[roi_y1:roi_y2, roi_x1:roi_x2]
                        curr_d = np.median(roi_depth)
                        
                        if track_id not in track_history:
                            track_history[track_id] = {'depth': [], 'pos': []}
                        
                        track_history[track_id]['depth'].append(curr_d)
                        track_history[track_id]['pos'].append((cx, cy))
                        
                        curr_ttc = 100.0
                        if len(track_history[track_id]['depth']) > 1:
                            d_prev = track_history[track_id]['depth'][-2]
                            diff = curr_d - d_prev
                            rel_vel = diff / dt
                            frame_rel_vels.append(rel_vel)
                            
                            if diff > 0:
                                curr_ttc = (d_prev * dt) / diff
                                frame_ttcs.append(curr_ttc)
                        
                        if curr_ttc < min_ttc:
                            min_ttc = curr_ttc
                            closest_vehicle_pos = (cx, cy)
                        
                        # Draw Box
                        color = (0, 255, 0) if curr_ttc > 2.0 else (0, 0, 255)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        label = f"ID:{track_id} D:{curr_d:.1f}"
                        if curr_ttc < 100:
                            label += f" TTC:{curr_ttc:.1f}s"
                        cv2.putText(display_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw TTC connection line to closest vehicle if dangerous
            if closest_vehicle_pos and min_ttc < 5.0:
                ego_pos = (width // 2, height - 50)
                color = (0, 0, 255) if min_ttc < 1.5 else (0, 165, 255)
                cv2.line(display_frame, ego_pos, closest_vehicle_pos, color, 2)
                cv2.putText(display_frame, f"TTC: {min_ttc:.1f}s", 
                            ((ego_pos[0]+closest_vehicle_pos[0])//2, (ego_pos[1]+closest_vehicle_pos[1])//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 3. Lane Detection (Ensemble)
            # 3a. Pre-processing: Enhancement for low light
            frame_enhanced = frame_rgb
            avg_brightness = np.mean(frame_rgb)
            is_dark = avg_brightness < 80
            if is_dark:
                frame_enhanced_bgr = enhance_low_light(frame)
                frame_enhanced = cv2.cvtColor(frame_enhanced_bgr, cv2.COLOR_BGR2RGB)

            # 3b. YOLOP Expert
            img_yolop = cv2.resize(frame_enhanced, (640, 640))
            img_yolop_tensor = torch.from_numpy(img_yolop).permute(2, 0, 1).float().to(self.device) / 255.0
            img_yolop_tensor = img_yolop_tensor.unsqueeze(0)
            
            with torch.no_grad():
                _, da_seg_out, ll_seg_out = self.yolop(img_yolop_tensor)
            
            da_mask = (torch.softmax(da_seg_out, dim=1)[0, 1, :, :] > 0.5).cpu().numpy().astype(np.uint8)
            da_mask_full = cv2.resize(da_mask, (width, height))
            
            yolop_mask = (torch.softmax(ll_seg_out, dim=1)[0, 1, :, :] > 0.5).cpu().numpy().astype(np.uint8)
            yolop_mask_full = cv2.resize(yolop_mask, (width, height))

            # 3c. OpenCV Expert
            img_cv = cv2.resize(frame_enhanced, (640, 640))
            img_cv_bgr = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            cv_mask = self.cv_lane_detector.get_lane_mask(img_cv_bgr)
            cv_mask = (cv_mask > 0).astype(np.uint8)
            cv_mask_full = cv2.resize(cv_mask, (width, height))

            # 3d. Voting / Ensemble
            lane_mask = ensemble_lane_detector(yolop_mask, cv_mask)
            lane_mask_full = cv2.resize(lane_mask, (width, height))
            
            # Overlay masks
            # Drivable area (green)
            display_frame[da_mask_full > 0] = display_frame[da_mask_full > 0] * 0.8 + np.array([0, 100, 0], dtype=np.uint8) * 0.2
            # YOLOP Lane (blue - for comparison)
            display_frame[yolop_mask_full > 0] = display_frame[yolop_mask_full > 0] * 0.5 + np.array([255, 0, 0], dtype=np.uint8) * 0.5
            # Final Ensemble Lane (yellow)
            display_frame[lane_mask_full > 0] = np.array([0, 255, 255], dtype=np.uint8)
            
            # Lane departure analysis (using ensemble mask)
            bottom_slice = lane_mask[400:, :]
            lane_pixels = np.where(bottom_slice > 0)
            lane_departure = 0
            curr_lane_center = 0.5
            has_lane = 0
            if len(lane_pixels[1]) > 0:
                curr_lane_center = np.mean(lane_pixels[1]) / 640.0
                has_lane = 1
                if prev_lane_center is not None:
                    lane_departure = curr_lane_center - prev_lane_center
                prev_lane_center = curr_lane_center
            else:
                prev_lane_center = None
            
            # Gauge for lane departure
            self.draw_gauge(display_frame, lane_departure * 5.0, "LANE DEVIATION", (20, height - 40), color=(255, 255, 0))

            # 4. Feature Accumulation
            max_rel_vel = max(frame_rel_vels) if frame_rel_vels else 0.0
            
            why_label = "normal"
            if min_ttc < 1.5: why_label = "tailgating"
            elif abs(lane_departure) > 0.1: why_label = "lane_departure"
            elif max_rel_vel > 20.0: why_label = "aggressive_closing"
            
            feat_row = {
                'frame': frame_idx,
                'min_ttc': min_ttc,
                'max_rel_vel': max_rel_vel,
                'lane_departure': lane_departure,
                'lane_center': curr_lane_center,
                'has_lane': has_lane,
                'has_objects': 1 if frame_ttcs else 0,
                'why': why_label
            }
            all_features.append(feat_row)
            
            # Smoothing
            df_hist = pd.DataFrame(all_features)
            df_hist['min_ttc_smooth'] = df_hist['min_ttc'].rolling(window=5, min_periods=1).mean()
            df_hist['lane_departure_smooth'] = df_hist['lane_departure'].rolling(window=5, min_periods=1).mean()
            
            current_ttc_smooth = df_hist['min_ttc_smooth'].iloc[-1]
            current_ld_smooth = df_hist['lane_departure_smooth'].iloc[-1]

            # 5. Dashboard Construction
            canvas = np.zeros((height, full_width, 3), dtype=np.uint8)
            canvas[:, :width] = display_frame
            
            # Sidebar
            sidebar = canvas[:, width:]
            cv2.rectangle(sidebar, (0, 0), (dash_w, height), (30, 30, 30), -1)
            
            y_off = 40
            cv2.putText(sidebar, "DASHCAM DEBUGGER", (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_off += 50
            
            # Live Classification
            cv2.putText(sidebar, "LIVE CLASSIFICATION:", (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_off += 30
            
            main_class = "WAITING..."
            conf = 0.0
            status_color = (150, 150, 150)
            
            if self.clf and len(df_hist) > 10:
                agg_feat = extract_video_features(df_hist)
                X = pd.DataFrame([agg_feat])
                pred_prob = self.clf.predict_proba(X)[0]
                idx = np.argmax(pred_prob)
                main_class = "RECKLESS" if idx == 1 else "NORMAL"
                conf = pred_prob[idx]
                status_color = (0, 0, 255) if main_class == "RECKLESS" else (0, 255, 0)
            
            cv2.putText(sidebar, f"{main_class}", (40, y_off), cv2.FONT_HERSHEY_SIMPLEX, 1.2, status_color, 3)
            y_off += 40
            cv2.putText(sidebar, f"Confidence: {conf*100:.1f}%", (40, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
            y_off += 60
            
            # Heuristic Alerts
            cv2.putText(sidebar, "HEURISTIC ALERTS:", (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_off += 30
            
            alerts = []
            if current_ttc_smooth < 1.5: alerts.append(("TAILGATING", (0, 0, 255)))
            if abs(current_ld_smooth) > 0.05: alerts.append(("LANE DEVIATION", (0, 165, 255)))
            if max_rel_vel > 15.0: alerts.append(("RAPID CLOSING", (0, 0, 255)))
            
            if not alerts:
                cv2.putText(sidebar, "NO ALERTS", (40, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_off += 30
            else:
                for a_txt, a_col in alerts:
                    cv2.putText(sidebar, f"! {a_txt}", (40, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.7, a_col, 2)
                    y_off += 30
            
            y_off += 30
            # Metrics
            cv2.putText(sidebar, "METRICS:", (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_off += 30
            
            metrics = [
                (f"Frame: {frame_idx}/{total_frames}", (255,255,255)),
                (f"Min TTC: {min_ttc:.2f}s", (255,255,255) if min_ttc > 2 else (0,0,255)),
                (f"Rel Vel: {max_rel_vel:.2f}", (255,255,255)),
                (f"Lane Dev: {lane_departure:.4f}", (255,255,255)),
                (f"Lane Center: {curr_lane_center:.2f}", (255,255,255))
            ]
            
            for m_txt, m_col in metrics:
                cv2.putText(sidebar, m_txt, (40, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.6, m_col, 1)
                y_off += 25

            # 6. Depth Inset (on canvas)
            depth_min = depth_map.min()
            depth_max = depth_map.max()
            depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
            depth_viz = (depth_norm * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_viz, cv2.COLORMAP_MAGMA)
            
            inset_size = (dash_w - 40, 200)
            depth_inset = cv2.resize(depth_color, inset_size)
            sidebar[height - 240:height - 40, 20:dash_w-20] = depth_inset
            cv2.putText(sidebar, "DEPTH ESTIMATION", (20, height - 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if out:
                out.write(canvas)
                
            if show:
                cv2.imshow("Dashcam Debugger", canvas)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_idx += 1
            pbar.update(1)
            
        cap.release()
        if out: out.release()
        if show: cv2.destroyAllWindows()
        pbar.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default="output_debug.mp4")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    
    if args.show and os.environ.get('DISPLAY') is None:
        args.show = False
        
    visualizer = Visualizer()
    visualizer.process_video(args.video, output_path=args.output, show=args.show, limit=args.limit)
    print(f"Saved debug visualization to {args.output}")
