import json
import os
from collections import defaultdict
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.7  # Threshold for exact match

def load_matches(matches_file):
    """Load matches from JSON file"""
    with open(matches_file, 'r') as f:
        return json.load(f)

def clean_matches(matches):
    """Clean matches by removing duplicates and organizing by frame"""
    frame_matches = defaultdict(lambda: defaultdict(list))
    for match in matches:
        frame_key = match.get('frame_number', 0)
        type_key = match.get('type', 'unknown')
        # Set match_type based on confidence
        confidence = match.get('confidence', 0)
        if confidence >= CONFIDENCE_THRESHOLD:
            match['match_type'] = 'Exact Match'
        else:
            match['match_type'] = 'No Match'
        # Only add if not already present
        if match not in frame_matches[frame_key][type_key]:
            frame_matches[frame_key][type_key].append(match)
    cleaned_matches = []
    for frame_key in sorted(frame_matches.keys(), key=int):
        for type_key in frame_matches[frame_key]:
            cleaned_matches.extend(frame_matches[frame_key][type_key])
    return cleaned_matches

def save_matches(matches, output_dir):
    """Save cleaned matches to a new JSON file"""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S_UTC")
    output_file = os.path.join(output_dir, f"{timestamp}_cleaned_matches.json")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(matches, f, indent=4)
    
    logger.info(f"Saved cleaned matches to {output_file}")

def main():
    # Get the most recent matches file
    matches_dir = "data/matches"
    matches_files = [f for f in os.listdir(matches_dir) if f.endswith("_matches.json") and not f.endswith("_cleaned_matches.json")]
    if not matches_files:
        logger.error("No matches files found")
        return
    
    latest_matches = max(matches_files)
    matches_file = os.path.join(matches_dir, latest_matches)
    
    logger.info(f"Processing matches from {matches_file}")
    
    # Load and clean matches
    matches = load_matches(matches_file)
    cleaned_matches = clean_matches(matches)
    
    # Save cleaned matches
    save_matches(cleaned_matches, matches_dir)
    
    logger.info(f"Original matches: {len(matches)}")
    logger.info(f"Cleaned matches: {len(cleaned_matches)}")

if __name__ == "__main__":
    main() 