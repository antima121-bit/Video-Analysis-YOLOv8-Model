import os
import json
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FashionDetector:
    def __init__(self, model_path=None):
        """
        Initialize the fashion detector with YOLOv8 model
        Args:
            model_path: Path to custom trained YOLOv8 model (optional)
        """
        try:
            # Load YOLOv8 model
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading custom model from {model_path}")
                self.model = YOLO(model_path)
            else:
                logger.info("Loading pretrained YOLOv8n model")
                self.model = YOLO('yolov8n.pt')
            
            # Fashion item classes we're interested in
            self.fashion_classes = {
                0: 'person',  # We'll use this to detect people wearing clothes
                27: 'handbag',
                28: 'umbrella',
                31: 'backpack',
                32: 'umbrella',
                33: 'handbag',
                34: 'tie',
                35: 'suitcase',
                41: 'skateboard',
                42: 'surfboard',
                43: 'tennis racket',
                44: 'bottle',
                46: 'wine glass',
                47: 'cup',
                48: 'fork',
                49: 'knife',
                50: 'spoon',
                51: 'bowl',
                52: 'banana',
                53: 'apple',
                54: 'sandwich',
                55: 'orange',
                56: 'broccoli',
                57: 'carrot',
                58: 'hot dog',
                59: 'pizza',
                60: 'donut',
                61: 'cake',
                62: 'chair',
                63: 'couch',
                64: 'potted plant',
                65: 'bed',
                66: 'dining table',
                67: 'toilet',
                68: 'tv',
                69: 'laptop',
                70: 'mouse',
                71: 'remote',
                72: 'keyboard',
                73: 'cell phone',
                74: 'microwave',
                75: 'oven',
                76: 'toaster',
                77: 'sink',
                78: 'refrigerator',
                79: 'book',
                80: 'clock',
                81: 'vase',
                82: 'scissors',
                83: 'teddy bear',
                84: 'hair drier',
                85: 'toothbrush'
            }
            
            # Enhanced confidence thresholds for different classes
            self.confidence_thresholds = {
                'person': 0.6,  # Increased for better accuracy
                'handbag': 0.65,
                'backpack': 0.65,
                'suitcase': 0.65,
                'tie': 0.6,
                'umbrella': 0.6,
                'default': 0.55
            }
            
            # Minimum size thresholds for different classes (in pixels)
            self.size_thresholds = {
                'person': 100,  # Minimum height for person detection
                'handbag': 30,
                'backpack': 30,
                'suitcase': 40,
                'default': 20
            }
            
            # Maximum overlap threshold for non-maximum suppression
            self.nms_threshold = 0.5
            
            logger.info("Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise

    def validate_bbox(self, bbox, image_shape, class_name):
        """
        Enhanced bounding box validation
        Args:
            bbox: Dictionary containing x1, y1, x2, y2 coordinates
            image_shape: Tuple of (height, width) of the image
            class_name: Name of the detected class
        Returns:
            bool: True if bbox is valid, False otherwise
        """
        try:
            height, width = image_shape[:2]
            
            # Check if coordinates are within image bounds
            if (bbox['x1'] < 0 or bbox['y1'] < 0 or 
                bbox['x2'] > width or bbox['y2'] > height):
                logger.debug(f"Bbox out of bounds: {bbox}")
                return False
            
            # Check if bbox has valid dimensions
            if (bbox['x2'] <= bbox['x1'] or bbox['y2'] <= bbox['y1']):
                logger.debug(f"Invalid bbox dimensions: {bbox}")
                return False
            
            # Get minimum size threshold for this class
            min_size = self.size_thresholds.get(class_name, self.size_thresholds['default'])
            
            # Check if bbox is too small
            bbox_width = bbox['x2'] - bbox['x1']
            bbox_height = bbox['y2'] - bbox['y1']
            
            if bbox_width < min_size or bbox_height < min_size:
                logger.debug(f"Bbox too small: {bbox}, class: {class_name}")
                return False
            
            # Check if bbox is too large (more than 80% of image)
            if (bbox_width > width * 0.8 or bbox_height > height * 0.8):
                logger.debug(f"Bbox too large: {bbox}")
                return False
            
            # Check aspect ratio for person detection
            if class_name == 'person':
                aspect_ratio = bbox_height / bbox_width
                if aspect_ratio < 1.5 or aspect_ratio > 3.0:  # Typical human aspect ratio
                    logger.debug(f"Invalid person aspect ratio: {aspect_ratio}")
                    return False
            
            return True
        except Exception as e:
            logger.error(f"Error validating bbox: {str(e)}")
            return False

    def apply_nms(self, detections):
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        """
        if not detections:
            return []
        
        # Sort detections by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # Initialize list of kept detections
        kept_detections = []
        
        while detections:
            # Get detection with highest confidence
            current = detections.pop(0)
            kept_detections.append(current)
            
            # Remove overlapping detections
            detections = [
                det for det in detections
                if self.calculate_iou(current['bbox'], det['bbox']) < self.nms_threshold
            ]
        
        return kept_detections

    def calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union between two bounding boxes
        """
        # Get coordinates
        x1 = max(bbox1['x1'], bbox2['x1'])
        y1 = max(bbox1['y1'], bbox2['y1'])
        x2 = min(bbox1['x2'], bbox2['x2'])
        y2 = min(bbox1['y2'], bbox2['y2'])
        
        # Calculate intersection area
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union area
        bbox1_area = (bbox1['x2'] - bbox1['x1']) * (bbox1['y2'] - bbox1['y1'])
        bbox2_area = (bbox2['x2'] - bbox2['x1']) * (bbox2['y2'] - bbox2['y1'])
        union = bbox1_area + bbox2_area - intersection
        
        return intersection / union if union > 0 else 0.0

    def detect_items(self, image_path, confidence_threshold=0.55):
        """
        Detect fashion items in an image with enhanced validation
        Args:
            image_path: Path to the image
            confidence_threshold: Minimum confidence score for detections
        Returns:
            List of detections with bounding boxes and confidence scores
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Error: Could not read image {image_path}")
                return []

            # Run YOLOv8 inference
            results = self.model(image, conf=confidence_threshold)[0]
            
            detections = []
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = r
                class_id = int(class_id)
                
                # Only include fashion-related items
                if class_id in self.fashion_classes:
                    class_name = self.fashion_classes[class_id]
                    
                    # Get class-specific confidence threshold
                    threshold = self.confidence_thresholds.get(
                        class_name, 
                        self.confidence_thresholds['default']
                    )
                    
                    # Skip if confidence is below threshold
                    if score < threshold:
                        continue
                    
                    bbox = {
                            'x1': int(x1),
                            'y1': int(y1),
                            'x2': int(x2),
                            'y2': int(y2)
                    }
                    
                    # Validate bounding box
                    if not self.validate_bbox(bbox, image.shape, class_name):
                        continue
                    
                    detection = {
                        'class': class_name,
                        'confidence': float(score),
                        'bbox': bbox,
                        'metadata': {
                            'width': bbox['x2'] - bbox['x1'],
                            'height': bbox['y2'] - bbox['y1'],
                            'aspect_ratio': (bbox['y2'] - bbox['y1']) / (bbox['x2'] - bbox['x1'])
                        }
                    }
                    detections.append(detection)
            
            # Apply Non-Maximum Suppression
            detections = self.apply_nms(detections)
            
            logger.debug(f"Found {len(detections)} valid detections in {image_path}")
            return detections
        except Exception as e:
            logger.error(f"Error detecting items in {image_path}: {str(e)}")
            return []

    def process_frames(self, frames_dir, output_dir, confidence_threshold=0.55):
        """
        Process all frames in a directory and save detection results
        Args:
            frames_dir: Directory containing video frames
            output_dir: Directory to save detection results
            confidence_threshold: Minimum confidence score for detections
        """
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
            
            # Get all frame directories
            frame_dirs = [d for d in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, d))]
            logger.info(f"Found {len(frame_dirs)} video directories to process")
            
            for video_dir in tqdm(frame_dirs, desc="Processing videos"):
                video_path = os.path.join(frames_dir, video_dir)
                output_path = os.path.join(output_dir, f"{video_dir}_detections.json")
                
                # Get all frames for this video
                frames = sorted([f for f in os.listdir(video_path) if f.endswith('.jpg')])
                logger.info(f"Processing {len(frames)} frames for video {video_dir}")
                
                video_detections = []
                for frame in tqdm(frames, desc=f"Processing {video_dir}", leave=False):
                    frame_path = os.path.join(video_path, frame)
                    try:
                        frame_number = int(frame.split('_')[1].split('.')[0])
                    except:
                        logger.warning(f"Could not parse frame number from {frame}, skipping")
                        continue
                    
                    # Detect items in frame
                    detections = self.detect_items(frame_path, confidence_threshold)
                    
                    if detections:
                        video_detections.append({
                            'frame_number': frame_number,
                            'frame_path': frame_path,
                            'detections': detections
                        })
                
                # Save detections for this video
                if video_detections:
                    with open(output_path, 'w') as f:
                        json.dump({
                            'video_id': video_dir,
                            'frames': video_detections,
                            'metadata': {
                                'total_frames': len(frames),
                                'frames_with_detections': len(video_detections),
                                'confidence_threshold': confidence_threshold,
                                'detection_stats': {
                                    'total_detections': sum(len(frame['detections']) for frame in video_detections),
                                    'detections_by_class': self._count_detections_by_class(video_detections)
                                }
                            }
                        }, f, indent=2)
                    logger.info(f"Saved {len(video_detections)} frames with detections for {video_dir}")
                else:
                    logger.warning(f"No detections found for video {video_dir}")
        except Exception as e:
            logger.error(f"Error processing frames: {str(e)}")
            raise

    def _count_detections_by_class(self, video_detections):
        """
        Count detections by class for statistics
        """
        class_counts = {}
        for frame in video_detections:
            for detection in frame['detections']:
                class_name = detection['class']
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        return class_counts

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Detect fashion items in video frames')
    parser.add_argument('--frames_dir', type=str, default='data/frames',
                      help='Directory containing video frames')
    parser.add_argument('--output_dir', type=str, default='data/detections',
                      help='Directory to save detection results')
    parser.add_argument('--model_path', type=str, default=None,
                      help='Path to custom trained YOLOv8 model (optional)')
    parser.add_argument('--confidence', type=float, default=0.55,
                      help='Minimum confidence score for detections')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting fashion item detection")
        logger.info(f"Frames directory: {args.frames_dir}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Confidence threshold: {args.confidence}")
        
        # Initialize detector
        detector = FashionDetector(model_path=args.model_path)
        
        # Process frames
        detector.process_frames(args.frames_dir, args.output_dir, args.confidence)
        
        logger.info("Detection completed successfully")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main() 