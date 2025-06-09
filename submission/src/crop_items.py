import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any

def crop_detection(frame: np.ndarray, bbox: List[float], padding: int = 10) -> np.ndarray:
    """
    Crop detection from frame with padding
    
    Args:
        frame: Input frame
        bbox: Bounding box [x1, y1, x2, y2]
        padding: Padding around detection in pixels
    
    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(frame.shape[1], x2 + padding)
    y2 = min(frame.shape[0], y2 + padding)
    
    return frame[y1:y2, x1:x2]

def process_detections(video_path: str, detections_path: str, output_dir: str) -> List[str]:
    """
    Process detections and crop items
    
    Args:
        video_path: Path to input video
        detections_path: Path to detections JSON file
        output_dir: Directory to save cropped items
    
    Returns:
        List of paths to cropped items
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load detections
    with open(detections_path, 'r') as f:
        detections_data = json.load(f)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    cropped_paths = []
    current_frame = -1
    frame = None
    
    with tqdm(desc="Cropping items", total=len(detections_data['detections'])) as pbar:
        for det in detections_data['detections']:
            # Read frame if needed
            if det['frame'] != current_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, det['frame'])
                ret, frame = cap.read()
                if not ret:
                    break
                current_frame = det['frame']
            
            # Crop detection
            cropped = crop_detection(frame, det['bbox'])
            
            # Save cropped image
            item_path = os.path.join(
                output_dir,
                f"item_{det['frame']:06d}_{len(cropped_paths):03d}.jpg"
            )
            cv2.imwrite(item_path, cropped)
            cropped_paths.append(item_path)
            
            pbar.update(1)
    
    cap.release()
    return cropped_paths

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Crop detected fashion items")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--detections", required=True, help="Path to detections JSON file")
    parser.add_argument("--output", required=True, help="Output directory for cropped items")
    
    args = parser.parse_args()
    
    # Process detections
    cropped_paths = process_detections(args.video, args.detections, args.output)
    print(f"Cropped {len(cropped_paths)} items to {args.output}") 