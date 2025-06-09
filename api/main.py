from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, Any
import json
import os
from datetime import datetime
import shutil
from pathlib import Path
import sys
import logging

# Add src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detect_items import FashionDetector
from match_items import ItemMatcher
from classify_vibe import VibeClassifier
from extract_frames import extract_frames

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models
detector = FashionDetector()
matcher = ItemMatcher("data/catalog.json")
vibe_classifier = VibeClassifier()

app = FastAPI(
    title="Flickd Smart Tagging API",
    description="API for processing videos and detecting fashion items and vibes",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
TEMP_DIR = Path("data/temp")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/process-video")
async def process_video(video: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Process a video file and return detected products and vibes
    
    Args:
        video: The video file to process
        
    Returns:
        Dict containing video_id, vibes, and products in the specified format
    """
    try:
        # Generate unique video ID
        video_id = f"vid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create temporary directory for this video
        video_dir = TEMP_DIR / video_id
        video_dir.mkdir(exist_ok=True)
        
        # Save uploaded video
        video_path = video_dir / "input.mp4"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)
        
        # Extract frames
        frames_dir = video_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        extract_frames(str(video_path), str(frames_dir))
        
        # Run object detection
        detections_dir = video_dir / "detections"
        detections_dir.mkdir(exist_ok=True)
        detector.process_frames(str(frames_dir), str(detections_dir))
        
        # Match products (limit to 30 images)
        matches_dir = video_dir / "matches"
        matches_dir.mkdir(exist_ok=True)
        detected_items = [{"image_path": str(f), "class": "unknown"} 
                         for f in frames_dir.glob("*.jpg")][:30]  # Limit to 30 images
        matched_products = matcher.match_items(detected_items, threshold=0.75)
        
        # Generate a caption from the video (placeholder - should be implemented with a video captioning model)
        caption = "This video shows fashion items being displayed."
        
        # Classify vibe
        vibes = vibe_classifier.classify_vibe(caption)
        
        # Format vibes as simple list
        vibe_list = [v["vibe"] if isinstance(v, dict) else v for v in vibes]
        
        # Format products in the required structure
        formatted_products = []
        for product in matched_products[:30]:  # Take top 30 products
            formatted_product = {
                "type": product.get("type", "unknown"),
                "color": product.get("color", "unknown"),
                "match_type": product.get("match_type", "similar"),
                "matched_product_id": product.get("matched_product_id", "unknown"),
                "confidence": float(product.get("confidence", 0.0))
            }
            formatted_products.append(formatted_product)
        
        # Prepare response in the exact required format
        response = {
            "video_id": video_id,
            "vibes": vibe_list,
            "products": formatted_products
        }
        
        # Clean up temporary files
        shutil.rmtree(video_dir)
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 