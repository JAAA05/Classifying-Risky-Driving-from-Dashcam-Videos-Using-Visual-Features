import os
import glob
import sys
import multiprocessing
from functools import partial

# Ensure current directory is in path if run from project root
sys.path.append(os.path.dirname(__file__))
from processor import VideoProcessor

# Global processor instance for each worker process
_processor = None

def init_worker():
    """Initialize the VideoProcessor once per worker process."""
    global _processor
    # We re-import here to ensure it's available in the worker process context if needed, 
    # though it should be available via the module-level import.
    _processor = VideoProcessor()

def process_single_video(task):
    """Worker function to process a single video."""
    video_path, output_path, category = task
    global _processor
    
    # Fallback if init_worker wasn't called or failed
    if _processor is None:
        _processor = VideoProcessor()
        
    video_name = os.path.basename(video_path)
    
    if os.path.exists(output_path):
        return f"Skipping {video_name}, already processed."
        
    print(f"Processing {video_name} ({category}) in process {os.getpid()}...")
    try:
        # Limit to 300 frames (approx 10s at 30fps) for feature extraction
        df = _processor.process_video(video_path, limit=300, show_progress=False)
        if df is not None and not df.empty:
            df['class'] = category
            df.to_csv(output_path, index=False)
            return f"Successfully processed {video_name}."
        else:
            return f"Finished {video_name} but no features were extracted."
    except Exception as e:
        return f"Error processing {video_name}: {e}"

def main():
    base_video_dir = "videos/2026-04-20"
    base_output_dir = "data/processed/2026-04-20"
    
    # Process both 'reckless' and 'normal' subfolders
    categories = ["reckless", "normal"]
    
    all_tasks = []
    
    for category in categories:
        video_dir = os.path.join(base_video_dir, category)
        output_dir = os.path.join(base_output_dir, category)
        os.makedirs(output_dir, exist_ok=True)
        
        extensions = ["*.mp4", "*.mov", "*.avi", "*.MOV"]
        videos = []
        for ext in extensions:
            videos.extend(glob.glob(os.path.join(video_dir, ext)))
        
        print(f"Found {len(videos)} videos in {category} folder.")
        
        for video_path in videos:
            video_name = os.path.basename(video_path)
            output_path = os.path.join(output_dir, f"{video_name}.csv")
            all_tasks.append((video_path, output_path, category))

    if not all_tasks:
        print("No videos found to process.")
        return

    num_cores = multiprocessing.cpu_count()
    print(f"\nStarting parallel processing with {num_cores} cores...")
    
    # Use a Pool to process videos in parallel
    # We use 'spawn' to avoid issues with CUDA if present
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    with multiprocessing.Pool(processes=num_cores, initializer=init_worker) as pool:
        results = pool.map(process_single_video, all_tasks)
        
    for result in results:
        if result:
            print(result)

if __name__ == "__main__":
    main()
