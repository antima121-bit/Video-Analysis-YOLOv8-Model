import os
import cv2
import numpy as np
from PIL import Image
import torch
from typing import List, Dict, Any
import json

def download_video(url: str, save_path: str) -> str:
    """Download video from URL"""
    import requests
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    return save_path

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Preprocess frame for model input"""
    # Resize to model input size
    frame = cv2.resize(frame, (640, 640))
    # Convert to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def crop_detection(frame: np.ndarray, bbox: List[float]) -> np.ndarray:
    """Crop detection from frame using bounding box"""
    x1, y1, x2, y2 = map(int, bbox)
    return frame[y1:y2, x1:x2]

def compute_similarity(features1: torch.Tensor, features2: torch.Tensor) -> float:
    """Compute cosine similarity between feature vectors"""
    similarity = torch.nn.functional.cosine_similarity(features1, features2)
    return float(similarity)

def get_vibe_rules() -> Dict[str, List[str]]:
    """Get rules for vibe classification"""
    return {
        "Coquette": ["pink", "lace", "floral", "romantic"],
        "Clean Girl": ["white", "minimal", "neutral"],
        "Cottagecore": ["floral", "vintage", "natural"],
        "Streetcore": ["urban", "edgy", "bold"],
        "Y2K": ["retro", "vibrant", "playful"],
        "Boho": ["ethnic", "flowy", "natural"],
        "Party Glam": ["sparkle", "bold", "elegant"]
    }

def classify_vibe_by_rules(matches: List[Dict[str, Any]], rules: Dict[str, List[str]]) -> List[str]:
    """Classify vibe using rule-based approach"""
    vibe_scores = {vibe: 0 for vibe in rules.keys()}
    
    for match in matches:
        product_type = match["type"].lower()
        color = match["color"].lower()
        
        for vibe, keywords in rules.items():
            if any(keyword in product_type or keyword in color for keyword in keywords):
                vibe_scores[vibe] += match["confidence"]
    
    # Get top 3 vibes
    sorted_vibes = sorted(vibe_scores.items(), key=lambda x: x[1], reverse=True)
    return [vibe for vibe, score in sorted_vibes[:3] if score > 0]

def save_output(output: Dict[str, Any], output_path: str):
    """Save output to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2) 