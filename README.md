

# Real-Time Multi-Modal System for Person Detection, Tracking, and Emotion Recognition

An integrated, real-time computer vision pipeline that performs person detection, instance segmentation, multi-object tracking (MOT), face cropping and emotion recognition — built to support applications in public safety, retail analytics, smart surveillance, healthcare, and human-computer interaction.


Project overview
---------------
This repository implements a Multi-Modal Person Analysis System that integrates:
- Person detection (bounding boxes)
- Instance segmentation (pixel-level masks)
- Person counting
- Multi-object tracking (consistent IDs across frames)
- Face detection/cropping per person
- Facial emotion recognition (categorical emotions)

### Problem Statement 
The project aims to develop an integrated computer vision system that can simultaneously perform multiple person-centric tasks: detecting individuals in images and videos, segmenting person instances, tracking their movement across video frames, counting the total number of people present, and recognizing facial emotions. Current systems often handle these tasks in isolation, requiring multiple separate models and pipelines, which leads to inefficiency and increased computational overhead.

**Why does the problem need to be solved?**

This integrated system addresses several real-world applications:
Public Safety & Crowd Management: Security personnel need to monitor crowd density, track individuals of interest, and assess emotional states in public spaces (airports, stadiums, shopping malls)
Retail Analytics: Businesses require customer counting, movement tracking, and emotion analysis to understand customer behavior and improve service
Smart Surveillance: Modern surveillance systems need comprehensive person analysis beyond simple detection
Healthcare & Eldercare: Monitoring facilities can benefit from tracking patients and detecting emotional distress
Human-Computer Interaction: Interactive systems need to understand both presence and emotional state of users
The integration of multiple capabilities into a single pipeline reduces deployment complexity, computational resources, and provides correlated insights that isolated systems cannot achieve.

### What aspect of the problem will a computer vision algorithm solve?

The computer vision algorithms will solve:
1. **Person Detection:** Identify and locate all persons in an image/video frame using object detection (YOLO, Faster R-CNN, or similar)
2. **Instance Segmentation:** Precisely delineate pixel-level boundaries of each person using Mask R-CNN or similar architectures
3. **Person Counting:** Aggregate detection results to provide accurate headcount in crowded scenarios
4. **Multi-Object Tracking (MOT):** Maintain consistent identities of individuals across video frames using algorithms like DeepSORT, ByteTrack, or SORT
5. **Face Detection & Cropping:** Isolate facial regions from detected persons using face detection models (MTCNN, RetinaFace, or YOLO-Face)
6. **Emotion Recognition:** Classify facial expressions into emotion categories (happy, sad, angry, neutral, surprise, fear, disgust) using CNN-based emotion classifiers

### Pipeline Flow:
![Pipeline Flow Diagram](Images/App_PipeLine_Flow.png)

The pipeline targets near real-time performance by combining fast detection models (e.g., YOLO-family) with optimized trackers (ByteTrack / DeepSORT) and lightweight emotion classifiers (MobileNet/ResNet variants tuned on FER2013/AffectNet).

Why this matters
----------------
A single integrated pipeline reduces deployment complexity and resource consumption compared to standalone systems, and enables cross-modal correlations (e.g., tracking + emotion over time) useful for safety, analytics, and assisted-care scenarios.

Features
--------
- Real-time person detection and instance segmentation
- Robust multi-object tracking with per-ID history
- Face cropping and per-face emotion classification
- Person counting and overlayed visualizations (bounding boxes, masks, IDs, emotion labels)
- Support for video files, webcams, and RTSP streams
- Modular model switches (swap detection / tracker / emotion model)

System architecture & pipeline
-----------------------------
High level steps:
1. Input frame (video / webcam / image)
2. Person detection (YOLOv8/YOLOv9 / Faster R-CNN)
3. Instance segmentation (Mask R-CNN or SAM-based refinement)
4. Tracking (ByteTrack / DeepSORT) → maintain persistent IDs
5. Face detection per person crop (MTCNN / RetinaFace)
6. Face preprocessing and emotion classification (CNN)
7. Visualization and logging (bbox, mask, ID, emotion, counts)
8. Metrics & evaluation output (mAP, MOT metrics, emotion accuracy)

A pipeline diagram is included at Images/App_PipeLine_Flow.png (open-source pipeline diagram).

Quick start
-----------

Prerequisites
- Linux / macOS / Windows (WSL recommended for GPU)
- Python 3.8+ (3.9/3.10 recommended)
- CUDA + cuDNN (if running on NVIDIA GPU). CPU-only mode supported (slower).
- At least 8GB RAM, recommended GPU with 6GB+ VRAM for real-time experiments.

Recommended Python workflow
- Create virtual environment (venv / conda) and activate it.

Installation
------------
1. Clone the repository
   git clone https://github.com/usd-aai-500-in1-group02/Real-Time-Multi-Modal-System-for-Person-Detection-Tracking-and-Emotion-Recognition.git
   cd Real-Time-Multi-Modal-System-for-Person-Detection-Tracking-and-Emotion-Recognition

2. (Optional) Create virtual environment
   python -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   .venv\Scripts\activate      # Windows

3. Install requirements
   pip install -r requirements.txt

Note: If GPU-enabled PyTorch is required, install the correct torch + torchvision build for your CUDA version:
https://pytorch.org/get-started/locally/

Running the app
---------------
The main demo script is app.py (examples assume it follows common CLI pattern; adapt flags if your app.py differs).

Run on a video file:
python app.py --source path/to/video.mp4 --weights models/detector.pt --tracker bytetrack --emotion weights/emotion.pt --output outputs/video_out.mp4

Run on webcam (device 0):
python app.py --source 0 --weights models/detector.pt --tracker deepsort --emotion weights/emotion.pt

Run on an RTSP stream:
python app.py --source "rtsp://user:pass@camera_ip/stream" --weights models/detector.pt --tracker bytetrack

Common flags (adjust based on app.py):
- --source: input source (file path, camera index, or RTSP URL)
- --weights: path to detection model weights
- --seg-weights: path to segmentation model weights (optional)
- --tracker: tracker selection (bytetrack / deepsort / sort)
- --emotion: path to emotion classifier weights
- --conf: detection confidence threshold (default 0.25)
- --output: path to save annotated output video
- --device: cpu / cuda:0

If your app.py uses a different CLI, replace the example flags accordingly. If you want, I can open your app.py and tailor these commands precisely.

Configuration & models
----------------------
- Detection: YOLOv8/YOLOv9 recommended for speed; Mask R-CNN for higher quality masks.
- Tracking: ByteTrack recommended for crowded scenes; DeepSORT if you prefer Re-ID features.
- Face detection: MTCNN / RetinaFace work well; RetinaFace is faster/more accurate for small faces.
- Emotion classifier: Lightweight MobileNet or ResNet-18 pre-trained and fine-tuned on FER2013 / AffectNet.

Place model weight files in a /models directory and update paths in config or CLI.

Suggested file layout for models:
- models/detector.pt
- models/segmentation.pt
- models/bytetrack.pth (if required by tracker)
- models/emotion_classifier.pth

Datasets & recommended weights
------------------------------
- Person detection/segmentation: COCO (http://cocodataset.org), CrowdHuman
- Tracking: MOTChallenge (MOT17, MOT20) for tracker evaluation
- Emotion recognition: FER2013 (Kaggle), AffectNet, RAF-DB

Download common pretrained weights:
- YOLO family: Ultralytics/releases or official vendor repository
- Mask R-CNN: torchvision or detectron2 weights
- Face detection: RetinaFace / MTCNN checkpoints
- Emotion classifier: trained on FER2013 or AffectNet (we recommend providing your own fine-tuned model)

Evaluation & metrics
--------------------
- Detection / segmentation: mAP (COCO-style), precision, recall
- Tracking: MOT metrics (MOTA, MOTP, IDF1, ID switches)
- Emotion recognition: accuracy, precision/recall per class, confusion matrix

Typical evaluation commands (adapt to tooling you include):
- Detection using COCO eval tools (pycocotools)
- Tracking: use py-motmetrics or MOTChallenge evaluation scripts
- Emotion: evaluate on FER2013 validation set using your classifier evaluation script

Directory structure (suggested)
-------------------------------
- app.py                     # main demo / application script
- requirements.txt
- models/                    # model weight files (not tracked)
- datasets/                  # dataset download scripts / links
- notebooks/                 # experiments and visualizations
- src/                       # core pipeline modules (detector, tracker, emotion)
- configs/                   # config files for experiments / models
- outputs/                   # annotated outputs, logs, saved videos
- Images/                    # pipeline diagrams and images
- README.md

Results / Expected outputs
-------------------------
After running the app on a demonstration video you should see:
- Annotated frames with bounding boxes, segmentation masks, tracking IDs, and emotion labels
- Console logs (frame rate, person count per frame)
- A saved annotated video (if --output is provided)
- Optional CSV / JSON logs containing per-frame detections and tracking metadata

Privacy & ethics note
---------------------
Emotion recognition is a sensitive technology. Use responsibly and ethically, ensuring compliance with local privacy laws and consent where required. Avoid deployment that may cause harm, discrimination, or violate user rights.

Contributing
------------
We welcome contributions. Suggested workflow:
1. Fork the repo
2. Create a feature branch: git checkout -b feat/your-feature
3. Commit changes and open a PR describing your changes
4. Add tests / notebooks where applicable

Please follow the repository code style and include a short description of experimental settings for reproducibility.

License
-------
Specify your license here (e.g., MIT, Apache-2.0). If none yet, add LICENSE file to the repo.

Authors & contact
-----------------
Team: USD AAI 500 In1 Group 02  
Primary repo contact: repo maintainers (see AUTHORS or CONTRIBUTORS file)

Acknowledgements
----------------
- COCO, MOTChallenge, FER2013, AffectNet datasets
- Ultralytics (YOLO) and other open-source model authors
- Research papers and community implementations used as references






