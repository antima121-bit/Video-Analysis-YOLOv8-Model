# Video Frame Analysis and Item Matching System

A comprehensive system for analyzing video frames, detecting objects, and matching them with catalog items.

## Features

- Video frame extraction and processing
- Object detection using YOLOv8
- Vibe classification for frames
- Catalog item matching
- RESTful API endpoints
- Web interface for video upload and analysis

## Project Structure

```
├── api/                    # API endpoints
│   ├── web_demo.py        # Flask server implementation
│   └── main.py            # API endpoints and business logic
├── frames/                # Frame processing
│   ├── extract_frames.py  # Frame extraction from videos
│   └── detector.py        # Object detection implementation
├── models/                # ML models and weights
│   ├── yolov8n.pt        # YOLOv8 model weights
│   ├── classify_vibe.py  # Vibe classification model
│   └── match_items.py    # Item matching implementation
├── data/                  # Data management
│   ├── download_data.py  # Data download utilities
│   ├── update_catalog.py # Catalog update scripts
│   ├── convert_catalog.py # Catalog format conversion
│   └── download_catalog_images.py # Catalog image downloader
├── src/                   # Source code
│   ├── static/           # Static assets (CSS, JS, images)
│   └── templates/        # HTML templates
├── README.md             # Project documentation
└── requirements.txt      # Project dependencies
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-name>
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download required models:
   ```bash
   # YOLOv8 model will be downloaded automatically on first run
   ```

## Usage

1. Start the Flask server:
   ```bash
   python api/web_demo.py
   ```

2. Access the web interface at `http://localhost:5000`

3. Upload a video file through the web interface

4. View the analysis results:
   - Detected objects
   - Frame vibes
   - Matched catalog items

## API Endpoints

- `POST /api/upload`: Upload video file
- `GET /api/frames/<video_id>`: Get extracted frames
- `GET /api/analysis/<video_id>`: Get analysis results
- `GET /api/matches/<video_id>`: Get matched items

## Data Management

The system uses several data management scripts:

- `download_data.py`: Downloads required datasets
- `update_catalog.py`: Updates the item catalog
- `convert_catalog.py`: Converts catalog formats
- `download_catalog_images.py`: Downloads catalog images

## Dependencies

- Flask 3.0.2
- OpenCV 4.9.0
- YOLOv8
- NumPy 1.26.4
- Other dependencies listed in requirements.txt

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Flask for the web framework
- OpenCV for video processing
- Font Awesome for icons
- Inter font family for typography 