<h1>🎥 Flickd - AI-Powered Fashion Detection and Matching</h1>

<p>
An AI-powered fashion detection and matching system developed for the 
<b>Flickd Hackathon</b>. This project enhances fashion discovery by analyzing videos, 
detecting fashion items, and recommending similar products using Computer Vision 
and Vision-Language Models.
</p>

<hr>

<h2>📋 Overview</h2>
<ul>
  <li>Automatically detects fashion items in user-uploaded videos</li>
  <li>Matches detected items with similar products from the Flickd catalog</li>
  <li>Analyzes vibes/styles for personalized suggestions</li>
  <li>Provides seamless product discovery & shopping experience</li>
</ul>

<hr>

<h2>✨ Key Features</h2>

<table>
<tr><th>Component</th><th>Description</th></tr>
<tr><td>Video Processing</td><td>Smart frame extraction and batch video analysis</td></tr>
<tr><td>Fashion Detection</td><td>YOLOv8-based high-accuracy object detection</td></tr>
<tr><td>Product Matching</td><td>CLIP embeddings + FAISS similarity search</td></tr>
<tr><td>Style Analysis</td><td>Vibe classification + color/style profiling</td></tr>
<tr><td>API Support</td><td>REST endpoints for integration</td></tr>
</table>

<hr>

<h2>🛠️ Technologies Used</h2>
<ul>
  <li><b>Python 3.8+</b></li>
  <li><b>YOLOv8 (Ultralytics)</b> — Object detection</li>
  <li><b>OpenAI CLIP</b> — Semantic image-text mapping</li>
  <li><b>FAISS</b> — Fast vector similarity search</li>
  <li><b>PyTorch</b> — Deep learning framework</li>
  <li><b>OpenCV</b> — Video processing</li>
  <li><b>FastAPI / Flask</b> — Web & API layer</li>
  <li><b>Pandas, NumPy, scikit-learn</b></li>
</ul>

<hr>

<h2>📂 Project Structure</h2>

<pre>
├── api/
│   ├── web_demo.py
│   └── main.py
├── frames/
│   ├── extract_frames.py
│   └── detector.py
├── models/
│   ├── yolov8n.pt
│   ├── classify_vibe.py
│   └── match_items.py
├── data/
│   ├── download_data.py
│   ├── update_catalog.py
│   └── convert_catalog.py
├── src/
│   ├── static/
│   └── templates/
├── requirements.txt
└── README.md
</pre>

<hr>

<h2>🚀 Getting Started</h2>

<h3>Prerequisites</h3>
<ul>
  <li>Python 3.8+</li>
  <li>CUDA GPU (Recommended)</li>
  <li>Git</li>
</ul>

<h3>Installation</h3>

<pre>
git clone https://github.com/antima121-bit/Video-Analysis-YOLOv8-Model.git
cd Flickd-AI-Hackathon

python -m venv venv
source venv/bin/activate   (Windows: venv\Scripts\activate)

pip install -r requirements.txt
</pre>

<h3>Download YOLO Weights</h3>

<pre>
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
</pre>

<hr>

<h2>🌐 API Endpoints</h2>

<table>
<tr><th>Method</th><th>Endpoint</th><th>Description</th></tr>
<tr><td>POST</td><td>/api/upload</td><td>Upload video file</td></tr>
<tr><td>GET</td><td>/api/frames/&lt;video_id&gt;</td><td>View extracted frames</td></tr>
<tr><td>GET</td><td>/api/analysis/&lt;video_id&gt;</td><td>Get detection & vibe analysis</td></tr>
<tr><td>GET</td><td>/api/matches/&lt;video_id&gt;</td><td>Get recommended catalog items</td></tr>
</table>

<hr>

<h2>📊 Example Output (JSON)</h2>

<pre>
{
  "detections": [
    {
      "item": "dress",
      "confidence": 0.95,
      "matches": ["product_1", "product_2"],
      "vibe": "casual"
    }
  ]
}
</pre>

<hr>

<h2>⚡ Performance</h2>
<ul>
  <li>Fast frame processing</li>
  <li>High detection accuracy</li>
  <li>Optimized FAISS-based matching</li>
  <li>Speed-boost with cached embeddings</li>
</ul>

<hr>

<h2>🤝 Contributing</h2>

<pre>
git checkout -b feature/NewFeature
git commit -m "Add NewFeature"
git push origin feature/NewFeature
</pre>

Open a Pull Request 🎉

<hr>

<h2>📝 License</h2>
<p>This project is licensed under the <b>MIT License</b>.</p>

<hr>

<p align="center">Made with 💛 for the Flickd Hackathon</p>
