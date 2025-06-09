import os
import json
import torch
from ultralytics import YOLO
from tqdm import tqdm
from typing import List, Dict, Any
import cv2
import numpy as np

class FashionDetector:
    def __init__(self, model_path: str = None):
        """
        Initialize fashion detector
        
        Args:
            model_path: Path to pretrained YOLO model. If None, uses default YOLOv8n
        """
        self.model = YOLO(model_path if model_path else 'yolov8n.pt')
        
        # Fashion-specific classes
        self.classes = [
            'dress', 'top', 'bottom', 'outerwear', 'shoes',
            'bag', 'accessory', 'hat', 'glasses', 'jewelry'
        ]
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Detect fashion items in frame
        
        Args:
            frame: Input frame
            conf_threshold: Confidence threshold for detections
        
        Returns:
            List of detections with bounding boxes and confidence scores
        """
        # Run inference
        results = self.model(frame, conf=conf_threshold)[0]
        
        detections = []
        for box in results.boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Get class and confidence
            cls = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            
            # Add to detections
            detections.append({
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'class': self.classes[cls],
                'confidence': conf
            })
        
        return detections
    
    def process_video(self, video_path: str, output_dir: str, conf_threshold: float = 0.5) -> str:
        """
        Process video and save detections
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save detections
            conf_threshold: Confidence threshold
        
        Returns:
            Path to detections JSON file
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        detections = []
        frame_count = 0
        
        with tqdm(desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect fashion items
                frame_detections = self.detect(frame, conf_threshold)
                
                # Add frame number
                for det in frame_detections:
                    det['frame'] = frame_count
                
                detections.extend(frame_detections)
                frame_count += 1
                pbar.update(1)
        
        cap.release()
        
        # Save detections
        output_path = os.path.join(output_dir, 'detections.json')
        with open(output_path, 'w') as f:
            json.dump({
                'video_path': video_path,
                'fps': fps,
                'total_frames': frame_count,
                'detections': detections
            }, f, indent=2)
        
        return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Detect fashion items in video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Output directory for detections")
    parser.add_argument("--model", help="Path to pretrained YOLO model")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = FashionDetector(args.model)
    
    # Process video
    output_path = detector.process_video(args.video, args.output, args.conf)
    print(f"Saved detections to {output_path}") 