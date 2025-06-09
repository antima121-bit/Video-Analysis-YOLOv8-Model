import cv2
import os
from tqdm import tqdm
from typing import List, Tuple

def extract_frames(video_path: str, output_dir: str, interval: float = 0.5) -> List[str]:
    """
    Extract frames from video at specified interval
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        interval: Time interval between frames in seconds
    
    Returns:
        List of paths to extracted frames
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    
    frame_paths = []
    frame_count = 0
    
    with tqdm(desc="Extracting frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Save frame
                frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
                
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    return frame_paths

def get_video_info(video_path: str) -> Tuple[int, int, float]:
    """
    Get video information
    
    Args:
        video_path: Path to video file
    
    Returns:
        Tuple of (width, height, fps)
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return width, height, fps

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Output directory for frames")
    parser.add_argument("--interval", type=float, default=0.5, help="Time interval between frames in seconds")
    
    args = parser.parse_args()
    
    # Extract frames
    frame_paths = extract_frames(args.video, args.output, args.interval)
    print(f"Extracted {len(frame_paths)} frames to {args.output}") 