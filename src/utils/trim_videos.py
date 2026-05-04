import cv2
import os
import argparse
from tqdm import tqdm

def trim_video(video_path, output_path, trim_percentage=10.0):
    """
    Trims the last X% of a video and saves it to the output path.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Calculate target frames
    trim_frames = int(total_frames * (trim_percentage / 100.0))
    target_frames = total_frames - trim_frames

    # Setup VideoWriter
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    with tqdm(total=target_frames, desc=f"Trimming {os.path.basename(video_path)}") as pbar:
        while cap.isOpened() and frame_count < target_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_count += 1
            pbar.update(1)

    cap.release()
    out.release()

def process_directory(input_dir, output_dir, trim_percentage=10.0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    videos = [f for f in os.listdir(input_dir) if f.lower().endswith(video_extensions)]

    print(f"Found {len(videos)} videos in {input_dir}")
    for video in videos:
        input_path = os.path.join(input_dir, video)
        output_path = os.path.join(output_dir, video)
        trim_video(input_path, output_path, trim_percentage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim the last X% of videos in a directory.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing source videos.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save trimmed videos.")
    parser.add_argument("--percentage", type=float, default=10.0, help="Percentage of video to trim from the end (0-100).")

    args = parser.parse_args()
    process_directory(args.input_dir, args.output_dir, args.percentage)
