from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
import traceback
import threading
import queue
import sys
import webbrowser
import time
import cv2
import base64
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

app = Flask(__name__,
            static_folder=os.path.join(parent_dir, 'src', 'static'),
            template_folder=os.path.join(parent_dir, 'src', 'templates'))

# Configuration
UPLOAD_FOLDER = os.path.join(parent_dir, 'src', 'uploads')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Store processing status
processing_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_id, video_path):
    """Process video in background thread"""
    try:
        logger.info(f"Starting video processing for {video_id}")
        processing_status[video_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting video processing...'
        }
        
        # Simulate processing steps
        steps = [
            'Analyzing video frames...',
            'Detecting objects...',
            'Matching with database...',
            'Calculating confidence scores...',
            'Generating results...'
        ]
        
        for i, step in enumerate(steps):
            time.sleep(2)  # Simulate processing time
            progress = (i + 1) * 20
            processing_status[video_id]['progress'] = progress
            processing_status[video_id]['message'] = f'Step {i + 1}/5: {step}'
        
        # Create sample results
        results = {
            'matches': [
                {
                    'type': 'Bottle',
                    'color': 'Blue',
                    'confidence': 0.95,
                    'match_type': 'exact',
                    'matched_product_id': 'PROD001'
                },
                {
                    'type': 'Can',
                    'color': 'Red',
                    'confidence': 0.88,
                    'match_type': 'exact',
                    'matched_product_id': 'PROD002'
                },
                {
                    'type': 'Box',
                    'color': 'Green',
                    'confidence': 0.92,
                    'match_type': 'partial',
                    'matched_product_id': 'PROD003'
                }
            ]
        }
        
        # Save results
        output_dir = os.path.join(app.static_folder, 'results', video_id)
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, f'{video_id}_matches.json'), 'w') as f:
            json.dump(results, f)
        
        # Update status
        processing_status[video_id]['status'] = 'completed'
        processing_status[video_id]['progress'] = 100
        processing_status[video_id]['message'] = 'Processing completed successfully'
        logger.info(f"Video processing completed for {video_id}")
        
    except Exception as e:
        logger.error(f"Error processing video {video_id}: {str(e)}")
        logger.error(traceback.format_exc())
        processing_status[video_id] = {
            'status': 'error',
            'progress': 0,
            'message': f'Error processing video: {str(e)}'
        }

@app.route('/')
def index():
    logger.info(f"Template folder: {app.template_folder}")
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Extract frames
            frames, fps, total_frames = extract_frames(filepath)
            
            # Detect objects
            detections = detect_objects(frames)
            
            # Match items
            matches = match_items(detections)
            
            # Calculate statistics
            total_objects = sum(len(frame['objects']) for frame in detections)
            avg_confidence = sum(match['confidence'] for match in matches) / len(matches) if matches else 0
            
            return jsonify({
                'status': 'success',
                'message': 'Video processed successfully',
                'results': {
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
            })
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

def extract_frames(video_path):
    """Extract frames from video"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Extract frames at 1-second intervals
    frame_interval = int(fps)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % frame_interval == 0:
            frames.append(frame)
        frame_count += 1
    
    cap.release()
    return frames, fps, total_frames

def detect_objects(frames):
    """Detect objects in frames using YOLO"""
    # Initialize YOLO model
    model = YOLO('yolov8n.pt')
    
    detections = []
    for i, frame in enumerate(frames):
        results = model(frame)
        frame_detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                
                if confidence > 0.5:  # Confidence threshold
                    frame_detections.append({
                        'class': class_name,
                        'confidence': confidence
                    })
        
        detections.append({
            'frame': i,
            'objects': frame_detections
        })
    
    return detections

def match_items(detections):
    """Match detected objects with catalog items"""
    matches = []
    for frame_det in detections:
        for obj in frame_det['objects']:
            # Simple matching logic - can be enhanced with more sophisticated matching
            match_confidence = obj['confidence']
            matches.append({
                'type': obj['class'],
                'confidence': match_confidence,
                'match_type': 'exact' if match_confidence > 0.8 else 'partial'
            })
    
    return matches

@app.route('/status/<video_id>')
def get_status(video_id):
    """Get processing status"""
    if video_id not in processing_status:
        return jsonify({'error': 'Video ID not found'}), 404
    return jsonify(processing_status[video_id])

@app.route('/results/<video_id>')
def get_results(video_id):
    """Get processing results"""
    try:
        results_file = os.path.join(app.static_folder, 'results', video_id, f'{video_id}_matches.json')
        if not os.path.exists(results_file):
            return jsonify({'error': 'Results not found'}), 404
            
        with open(results_file, 'r') as f:
            results = json.load(f)
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error getting results: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    try:
        return send_from_directory(app.static_folder, filename)
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {str(e)}")
        return str(e), 404

@app.errorhandler(404)
def not_found(error):
    logger.error(f"404 error: {str(error)}")
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"500 error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    try:
        logger.info("Starting Flask server...")
        logger.info(f"Current directory: {current_dir}")
        logger.info(f"Template folder: {app.template_folder}")
        logger.info(f"Static folder: {app.static_folder}")
        logger.info(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)  # Wait for server to start
            try:
                webbrowser.open('http://127.0.0.1:5000')
                logger.info("Browser opened successfully")
            except Exception as e:
                logger.error(f"Failed to open browser: {str(e)}")
        
        # Start browser in a separate thread
        threading.Thread(target=open_browser, daemon=True).start()
        
        # Run the Flask app
        app.run(
            debug=True,
            host='127.0.0.1',
            port=5000,
            use_reloader=False,
            threaded=True
        )
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1) 