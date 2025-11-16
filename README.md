# RealTime Multi-Modal System for Person Detection, Tracking, and Emotion Recognition using Computer Vision

## Multi-Modal Person Detection & Segmentation, Tracking, and Emotion Recognition System

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

### Proposed Technical Approach
**Dataset Suggestions:**
Person Detection/Segmentation: COCO Dataset, CrowdHuman, or custom dataset
Emotion Recognition: FER2013, AffectNet, or RAF-DB
Tracking: MOT Challenge datasets (MOT17, MOT20)

**Model Architecture Options:**
Detection & Segmentation:
YOLOv8/YOLOv9 for real-time detection
Mask R-CNN for instance segmentation
SAM (Segment Anything Model) for advanced segmentation

**Tracking:**
DeepSORT (combines detection with Re-ID features)
ByteTrack (state-of-the-art tracker)
StrongSORT

**Emotion Recognition:**
ResNet-based emotion classifier
MobileNet for efficiency
Custom CNN architecture
Pre-trained models fine-tuned on emotion datasets



### Pipeline Flow:
![Pipeline Flow Diagram](Images/App_PipeLine_Flow.png)

### Expected Deliverables:
Person detection with bounding boxes
Instance segmentation masks for each person
Person count display
Emotion labels on detected faces
Tracking IDs are maintained across video frames
Performance metrics (mAP, precision, recall, tracking accuracy, emotion classification accuracy)


