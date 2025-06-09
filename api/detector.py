import cv2
import numpy as np
from ultralytics import YOLO
import logging
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoDetector:
    def __init__(self):
        """Initialize the detector with YOLO model"""
        try:
            # Check if CUDA is available and use GPU if possible
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Using device: {self.device}")
            
            # Load model and move to appropriate device
            self.model = YOLO('yolov8n.pt')
            self.model.to(self.device)
            
            # Set model parameters for faster inference
            self.model.conf = 0.5  # Confidence threshold
            self.model.iou = 0.45  # NMS IoU threshold
            
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            raise

    def extract_frames(self, video_path):
        """Extract frames from video at 2-second intervals"""
        try:
            frames = []
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Extract frames at 2-second intervals for faster processing
            frame_interval = int(fps * 2)  # Process every 2 seconds
            frame_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % frame_interval == 0:
                    # Resize frame to smaller size for faster processing
                    frame = cv2.resize(frame, (640, 480))
                    frames.append(frame)
                frame_count += 1
            
            cap.release()
            return frames, fps, total_frames
        except Exception as e:
            logger.error(f"Error extracting frames: {str(e)}")
            raise

    def detect_objects(self, frames):
        """Detect objects in frames using YOLO with batch processing"""
        try:
            detections = []
            # Process frames in batches of 4 for better GPU utilization
            batch_size = 4
            for i in range(0, len(frames), batch_size):
                batch_frames = frames[i:i + batch_size]
                # Run inference on batch
                results = self.model(batch_frames, verbose=False)
                
                for j, r in enumerate(results):
                    frame_detections = []
                    boxes = r.boxes
                    
                    for box in boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.model.names[class_id]
                        
                        frame_detections.append({
                            'class': class_name,
                            'confidence': confidence
                        })
                    
                    detections.append({
                        'frame': i + j,
                        'objects': frame_detections
                    })
            
            return detections
        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            raise

    def match_items(self, detections):
        """Match detected objects with catalog items"""
        try:
            matches = []
            # Use a set to track unique objects for faster matching
            seen_objects = set()
            
            for frame_det in detections:
                for obj in frame_det['objects']:
                    obj_key = f"{obj['class']}_{obj['confidence']:.2f}"
                    if obj_key not in seen_objects:
                        seen_objects.add(obj_key)
                        matches.append({
                            'type': obj['class'],
                            'confidence': obj['confidence'],
                            'match_type': 'exact' if obj['confidence'] > 0.8 else 'partial'
                        })
            
            return matches
        except Exception as e:
            logger.error(f"Error matching items: {str(e)}")
            raise

    def process_video(self, video_path):
        """Process video and return analysis results"""
        try:
            logger.info("Starting video processing...")
            
            # Extract frames
            frames, fps, total_frames = self.extract_frames(video_path)
            logger.info(f"Extracted {len(frames)} frames from video")
            
            # Detect objects
            detections = self.detect_objects(frames)
            logger.info(f"Completed object detection on {len(detections)} frames")
            
            # Match items
            matches = self.match_items(detections)
            logger.info(f"Found {len(matches)} unique matches")
            
            # Calculate statistics
            total_objects = sum(len(frame['objects']) for frame in detections)
            avg_confidence = sum(match['confidence'] for match in matches) / len(matches) if matches else 0
            
            result = {
                'frames': {
                    'total': total_frames,
                    'processed': len(frames),
                    'fps': fps
                },
                'detections': {
                    'count': total_objects,
                    'confidence': round(avg_confidence * 100, 1)
                },
                'matches': {
                    'count': len(matches),
                    'accuracy': round(avg_confidence * 100, 1)
                },
                'overall': {
                    'confidence': round(avg_confidence * 100, 1),
                    'success_rate': round((len(matches) / total_objects * 100) if total_objects > 0 else 0, 1)
                }
            }
            
            logger.info("Video processing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            raise 