import os
import glob
import pandas as pd
import numpy as np
import joblib
import argparse
from tqdm import tqdm
import sys

# Import VideoProcessor from pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipeline'))
from processor import VideoProcessor

# Import feature extraction logic from training script
sys.path.append(os.path.dirname(__file__))
from train_classifier import extract_video_features

def run_inference(video_folder, model_path, limit=300):
    # 1. Load Model
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train_classifier.py first.")
        return
        
    clf = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # 2. Initialize Processor
    processor = VideoProcessor()
    
    # 3. Find Videos
    extensions = ["*.mp4", "*.mov", "*.avi", "*.MOV"]
    videos = []
    for ext in extensions:
        videos.extend(glob.glob(os.path.join(video_folder, ext)))
        
    if not videos:
        print(f"No videos found in {video_folder}")
        return

    print(f"Found {len(videos)} videos. Starting inference...")
    
    results = []
    
    for video_path in videos:
        video_name = os.path.basename(video_path)
        print(f"\n--- Analyzing: {video_name} ---")
        
        try:
            # Extract frame-level features
            df = processor.process_video(video_path, limit=limit)
            
            if df is not None and not df.empty:
                # Aggregate for the classifier
                feat_dict = extract_video_features(df)
                X = pd.DataFrame([feat_dict])
                
                # Predict main class
                pred_prob = clf.predict_proba(X)[0]
                main_class = "Reckless" if np.argmax(pred_prob) == 1 else "Normal"
                confidence = pred_prob[np.argmax(pred_prob)]
                
                # Identify secondary behaviors (from the heuristic "why" column)
                # We report any behavior that appears in more than 5% of the processed frames
                why_counts = df['why'].value_counts(normalize=True).to_dict()
                secondary = []
                for behavior, fraction in why_counts.items():
                    if behavior != 'normal' and fraction > 0.05:
                        secondary.append(f"{behavior} ({fraction*100:.1f}%)")
                
                results.append({
                    'video': video_name,
                    'class': main_class,
                    'confidence': confidence,
                    'secondary': ", ".join(secondary) if secondary else "None detected"
                })
                
                print(f"RESULT: {main_class} ({confidence*100:.1f}% confidence)")
                print(f"SECONDARY: {', '.join(secondary) if secondary else 'None detected'}")
                
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            
    # 4. Final Summary
    print("\n" + "="*50)
    print(f"{'VIDEO':<30} | {'CLASS':<10} | {'SECONDARY BEHAVIORS'}")
    print("-" * 70)
    for res in results:
        print(f"{res['video']:<30} | {res['class']:<10} | {res['secondary']}")
    print("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a folder of dashcam videos.")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing videos.")
    parser.add_argument("--model", type=str, default="src/models/classifier.joblib", help="Path to the trained model.")
    parser.add_argument("--limit", type=int, default=300, help="Frame limit per video (default 300).")
    
    args = parser.parse_args()
    run_inference(args.folder, args.model, args.limit)
