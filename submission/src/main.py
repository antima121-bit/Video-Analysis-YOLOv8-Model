import os
import json
import cv2
import torch
from ultralytics import YOLO
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from tqdm import tqdm

class FashionVideoAnalyzer:
    def __init__(self, catalog_path, vibes_path, model_dir):
        self.catalog_path = catalog_path
        self.vibes_path = vibes_path
        self.model_dir = model_dir
        
        # Load models
        self.yolo_model = YOLO('yolov8n.pt')  # We'll fine-tune this for fashion
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Load catalog and vibes
        self.catalog = self._load_catalog()
        self.vibes = self._load_vibes()
        
        # Create output directory
        os.makedirs('outputs', exist_ok=True)
    
    def _load_catalog(self):
        """Load and preprocess catalog data"""
        import pandas as pd
        catalog = pd.read_csv(self.catalog_path)
        return catalog
    
    def _load_vibes(self):
        """Load predefined vibes"""
        with open(self.vibes_path, 'r') as f:
            return json.load(f)
    
    def extract_frames(self, video_path, interval=0.5):
        """Extract frames from video at specified interval"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                frames.append(frame)
            frame_count += 1
            
        cap.release()
        return frames
    
    def detect_fashion_items(self, frames):
        """Detect fashion items in frames using YOLO"""
        detections = []
        for frame in frames:
            results = self.yolo_model(frame)
            detections.extend(results)
        return detections
    
    def match_products(self, detections):
        """Match detected items with catalog products using CLIP"""
        matches = []
        for det in detections:
            # Process detection with CLIP
            inputs = self.clip_processor(images=det, return_tensors="pt")
            image_features = self.clip_model.get_image_features(**inputs)
            
            # Compare with catalog
            for _, product in self.catalog.iterrows():
                similarity = self._compute_similarity(image_features, product)
                if similarity > 0.75:  # Threshold as per requirements
                    matches.append({
                        "type": product['category'],
                        "color": product['color'],
                        "matched_product_id": product['product_id'],
                        "match_type": "exact" if similarity > 0.9 else "similar",
                        "confidence": float(similarity)
                    })
        return matches
    
    def classify_vibes(self, matches):
        """Classify video vibes based on matched products"""
        # Simple rule-based approach for now
        # Can be enhanced with ML-based classification
        vibes = []
        # Add vibe classification logic here
        return vibes[:3]  # Return top 3 vibes
    
    def analyze_video(self, video_path):
        """Main pipeline to analyze a video"""
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        
        # Extract frames
        frames = self.extract_frames(video_path)
        
        # Detect fashion items
        detections = self.detect_fashion_items(frames)
        
        # Match with catalog
        matches = self.match_products(detections)
        
        # Classify vibes
        vibes = self.classify_vibes(matches)
        
        # Prepare output
        output = {
            "video_id": video_id,
            "vibes": vibes,
            "products": matches
        }
        
        # Save output
        output_path = os.path.join('outputs', f'{video_id}.json')
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        return output

def main():
    # Initialize analyzer
    analyzer = FashionVideoAnalyzer(
        catalog_path='catalog.csv',
        vibes_path='vibes_list.json',
        model_dir='models'
    )
    
    # Process all videos
    video_dir = 'videos'
    for video_file in os.listdir(video_dir):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_dir, video_file)
            print(f"Processing {video_file}...")
            analyzer.analyze_video(video_path)

if __name__ == "__main__":
    main() 