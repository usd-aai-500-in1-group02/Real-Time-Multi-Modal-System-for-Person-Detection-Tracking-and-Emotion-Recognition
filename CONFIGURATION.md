# Configuration Guide
**Multi-Modal Person Detection & Analysis System**

---

## 📋 Feature Configuration Reference

### Analysis Tasks

| Feature | What It Does | Requirements | When to Use | Output |
|---------|-------------|--------------|-------------|---------|
| **Person Detection** | Detects people in images/videos with bounding boxes | None (base feature) | **Always enable** - Required for all other features | Green boxes around persons, confidence scores |
| **Instance Segmentation** | Creates pixel-level masks of each person | Person Detection | Use for: Precise person boundaries, background removal, occupancy maps | Colored overlay masks on persons |
| **Multi-Object Tracking** | Assigns unique IDs to track persons across frames | Person Detection | Use for: Video analysis, counting unique people, trajectory analysis | Track IDs displayed, maintains identity across frames |
| **Person Counting** | Counts total persons in frame | Person Detection | Use for: Occupancy monitoring, foot traffic analysis | Count displayed on image |
| **Emotion Recognition** | Detects facial emotions (happy, sad, angry, neutral, surprise, fear, disgust) | Person Detection | Use for: Sentiment analysis, customer satisfaction, security monitoring | Emotion labels with confidence scores |

### Advanced Features

| Feature | What It Does | Requirements | When to Use | Output |
|---------|-------------|--------------|-------------|---------|
| **🔥 Show Heatmap** | Visualizes crowd density as heat map | Person Detection | Use for: Identifying congestion zones, popular areas in retail/events | Red (high density) to blue (low density) overlay |
| **📈 Show Trajectories** | Draws movement paths of tracked persons | Person Detection + Multi-Object Tracking | Use for: Understanding movement patterns, flow analysis, security | Colored lines showing person paths |
| **👥 Social Distancing** | Measures distance between people, flags violations | Person Detection | Use for: COVID compliance, safety monitoring, crowd management | Red lines between people too close, violation count |
| **🌊 Crowd Flow** | Analyzes dominant movement directions | Person Detection + Multi-Object Tracking | Use for: Traffic flow analysis, evacuation planning, facility design | Direction arrows, flow statistics (N/S/E/W) |
| **🚨 Enable Alerts** | Triggers alerts for crowding, violations, loitering | Person Detection (+ specific features for each alert type) | Use for: Real-time monitoring, security alerts, automated notifications | Alert notifications with severity levels |

---

## 🎯 Recommended Configurations by Use Case

### 1. Basic Person Counting (Simplest)
**Use for:** Simple foot traffic counting, occupancy monitoring

**Configuration:**
- ✅ Person Detection
- ✅ Person Counting
- ❌ All others OFF

**Best for:** Retail stores, libraries, small venues

---

### 2. Video Analysis with Tracking (Most Common)
**Use for:** Security footage, retail analytics, traffic monitoring

**Configuration:**
- ✅ Person Detection
- ✅ Multi-Object Tracking
- ✅ Person Counting
- ✅ Enable Alerts
- ❌ Others OFF

**Best for:** Security cameras, parking lots, building entrances

---

### 3. Emotion & Sentiment Analysis
**Use for:** Customer satisfaction, user experience research, marketing

**Configuration:**
- ✅ Person Detection
- ✅ Emotion Recognition
- ✅ Enable Alerts
- ❌ Others OFF

**Best for:** Customer service, focus groups, user testing

---

### 4. COVID/Safety Compliance
**Use for:** Social distancing monitoring, safety compliance, crowd management

**Configuration:**
- ✅ Person Detection
- ✅ Multi-Object Tracking
- ✅ Social Distancing
- ✅ Enable Alerts
- ✅ Show Heatmap

**Alert Settings:**
- Crowding Threshold: 10 people
- Min Distance: 150 pixels

**Best for:** Offices, schools, public spaces during pandemic

---

### 5. Advanced Crowd Analytics (Full Features)
**Use for:** Facility planning, event management, comprehensive analytics

**Configuration:**
- ✅ Person Detection
- ✅ Instance Segmentation
- ✅ Multi-Object Tracking
- ✅ Person Counting
- ✅ Show Heatmap
- ✅ Show Trajectories
- ✅ Crowd Flow
- ✅ Enable Alerts

**Best for:** Large events, shopping malls, transportation hubs

---

### 6. Image Analysis Only (Single Frame)
**Use for:** Single photo analysis, profile pictures, static scenes

**Configuration:**
- ✅ Person Detection
- ✅ Emotion Recognition (optional)
- ✅ Show Heatmap (optional)
- ❌ Multi-Object Tracking (not useful for single images)
- ❌ Show Trajectories (not useful for single images)
- ❌ Crowd Flow (not useful for single images)

**Best for:** Photo analysis, group photos, event snapshots

---

## ⚙️ Model Parameters

### Detection Confidence (Default: 0.50)
- **Range:** 0.0 - 1.0
- **Lower values (0.3-0.4):** More detections, may include false positives (people that aren't really there)
- **Medium values (0.5-0.6):** Balanced accuracy and recall
- **Higher values (0.7-0.9):** Fewer but more accurate detections, may miss some people

**Recommended:**
- Crowded scenes: 0.4-0.5
- Clear scenes: 0.6-0.7
- High precision needed: 0.7+

### Emotion Confidence (Default: 0.60)
- **Range:** 0.0 - 1.0
- **Lower values:** Accept more emotion predictions (may be less accurate)
- **Higher values:** Only accept very confident predictions

**Recommended:**
- Good lighting, clear faces: 0.6-0.7
- Poor lighting, distant faces: 0.4-0.5

### IoU Threshold (Default: 0.45)
- **Range:** 0.0 - 1.0
- **Purpose:** Removes duplicate detections of the same person
- **Lower values:** More aggressive duplicate removal
- **Higher values:** Keep more detections, may have duplicates

**Recommended:** Keep at 0.45 unless you see duplicate boxes on same person

---

## 🚨 Alert Settings

### Crowding Threshold (Default: 10)
- **Range:** 1-50 people
- **Triggers:** Alert when person count exceeds this number
- **Use case specific:**
  - Small room: 5-10
  - Medium space: 10-20
  - Large venue: 20-50

### Social Distancing Settings
- **Min Distance:** 50-300 pixels (Default: 150)
  - Depends on camera height and field of view
  - Test and adjust based on your camera setup
  - Closer camera: 50-100 pixels
  - Farther camera: 150-300 pixels

---

## 📊 Feature Compatibility Matrix

| Feature | Works on Images | Works on Videos | Requires Tracking |
|---------|----------------|-----------------|-------------------|
| Person Detection | ✅ | ✅ | ❌ |
| Instance Segmentation | ✅ | ✅ | ❌ |
| Multi-Object Tracking | ❌ | ✅ | N/A |
| Person Counting | ✅ | ✅ | ❌ |
| Emotion Recognition | ✅ | ✅ | ❌ |
| Show Heatmap | ✅ | ✅ | ❌ |
| Show Trajectories | ❌ | ✅ | ✅ |
| Social Distancing | ✅ | ✅ | ❌ |
| Crowd Flow | ❌ | ✅ | ✅ |
| Enable Alerts | ✅ | ✅ | ❌ |

---

## 💡 Performance Tips

### For Faster Processing:
1. **Disable unused features** - Each feature adds processing time
2. **Lower detection confidence** slightly (0.4-0.5) for faster inference
3. **Skip frames** in video processing (process every 2-3 frames)
4. **Use smaller video resolution** if possible
5. **Disable Instance Segmentation** - it's the slowest feature

### For Better Accuracy:
1. **Good lighting** - Ensure scenes are well-lit for emotion recognition
2. **Higher confidence thresholds** (0.6-0.7) for fewer false positives
3. **Clear, front-facing views** work best for emotion detection
4. **Stable camera** produces better tracking results
5. **Consistent frame rate** improves tracking performance

---

## 🎥 Video Processing Settings

### Process Every N Frames
- **1 frame:** Process every frame (slowest, most accurate)
- **2-3 frames:** Good balance (recommended)
- **5+ frames:** Fast but may miss events

### Max Frames to Process
- **Small videos (< 1 min):** Process all frames
- **Medium videos (1-5 min):** 100-300 frames
- **Long videos (> 5 min):** 100-500 frames for testing, then full run

---

## 🔍 Troubleshooting

### No Detections Found
- ✅ Ensure **Person Detection** is enabled
- ✅ Lower **Detection Confidence** threshold (try 0.3-0.4)
- ✅ Check image quality and lighting
- ✅ Ensure people are clearly visible in frame

### Emotions Not Detected
- ✅ **Person Detection** must be enabled (auto-enabled)
- ✅ Faces must be visible and relatively large
- ✅ Lower **Emotion Confidence** threshold (try 0.4-0.5)
- ✅ Ensure good lighting on faces

### Tracking IDs Jumping/Switching
- ✅ Use consistent frame rate video
- ✅ Avoid sudden camera movements
- ✅ Ensure good lighting
- ✅ Lower **Detection Confidence** to catch all people

### Slow Processing
- ✅ Disable **Instance Segmentation**
- ✅ Disable **Emotion Recognition** if not needed
- ✅ Process fewer frames (increase skip_frames)
- ✅ Use smaller resolution video

---

## 📝 Examples

### Example 1: Retail Store Analytics
**Goal:** Count customers and analyze shopping patterns

**Configuration:**
```
Person Detection: ✅ (0.5 confidence)
Multi-Object Tracking: ✅
Person Counting: ✅
Show Heatmap: ✅
Show Trajectories: ✅
Enable Alerts: ✅ (threshold: 15 people)
```

### Example 2: Office Safety Monitoring
**Goal:** Monitor social distancing compliance

**Configuration:**
```
Person Detection: ✅ (0.6 confidence)
Multi-Object Tracking: ✅
Social Distancing: ✅ (min distance: 150px)
Show Heatmap: ✅
Enable Alerts: ✅ (crowding: 8 people)
```

### Example 3: Customer Satisfaction Survey
**Goal:** Analyze customer emotions in service area

**Configuration:**
```
Person Detection: ✅ (0.6 confidence)
Emotion Recognition: ✅ (0.6 confidence)
Person Counting: ✅
Enable Alerts: ❌
```

---

## 📚 Additional Resources

- **Model Information:** Uses YOLOv11s for detection, DeepSort for tracking, DeepFace for emotions
- **Supported Formats:**
  - Images: JPG, PNG, BMP
  - Videos: MP4, AVI, MOV, MKV
- **Recommended Resolution:** 720p-1080p for best balance of speed and accuracy

---

**Need Help?** Check the Analytics tab for performance metrics and Reports tab for detailed analysis exports.
