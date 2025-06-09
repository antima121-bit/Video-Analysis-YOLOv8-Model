# Fashion Video Analysis System

A comprehensive system for analyzing fashion videos, detecting items, matching with catalog products, and classifying vibes.

## Features

- Video frame extraction
- Fashion item detection using YOLOv8
- Item cropping and preprocessing
- Product catalog matching using CLIP
- Vibe classification
- RESTful API interface

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your videos in the `videos/` directory
4. Place your catalog.csv in the root directory
5. Run the analysis:
```bash
python src/main.py
```

## Project Structure

```
submission/
├── videos/          # Input videos
├── outputs/         # Analysis results
├── models/          # Trained models
├── src/            # Source code
├── catalog.csv     # Product catalog
├── vibes_list.json # Predefined vibes
├── requirements.txt
└── README.md
```

## Models Used

- YOLOv8n for fashion item detection
- CLIP ViT-B/32 for product matching
- Custom vibe classification model

## Output Format

For each video, a JSON file is generated in the `outputs/` directory containing:
- Video ID
- Detected vibes
- Matched products with confidence scores

## Demo

A demo video is available showing the pipeline in action.

## License

[Your License] 