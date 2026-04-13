# 🚗 Driver Fatigue Detection with Gesture-Based Activation

> Computer Vision group project — a real-time driver fatigue detection system with a gesture-based activation mechanism. Built with MediaPipe, OpenCV, and scikit-learn.

---

## 📌 Overview

This system has two main components:

1. **Gesture-Based Activation** — the system remains inactive until the driver performs a specific hand gesture sequence (open hand → thumbs up)
2. **Fatigue Detection** — once activated, the system monitors the driver for signs of drowsiness using eye closure, yawning, and head pose analysis

> ⚠️ Video files are not included in the repo due to size. Download them from [Google Drive](https://drive.google.com/drive/folders/12Bj_WIQJwLvqsWceDXsqbualINhLOvw0?usp=sharing) and place them in the corresponding `data/` subfolders.

---

## ⚙️ Setup

```bash
# 1. Clone the repo
git clone https://github.com/mjcolmenarez/driver_fatigue_detection.git
cd driver_fatigue_detection

# 2. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download videos from Google Drive and place in data/ folders

# 5. Test that everything works
python test_video.py
```

---

## ✅ What's Done

### Gesture Activation
The system uses **MediaPipe Hand Landmarker** to detect and classify hand gestures in real time. The activation sequence requires the driver to show an **open hand** followed by a **thumbs up**, each held for 0.5 seconds, completed within a 5-second window. If the sequence is incorrect or times out, the system remains inactive.

```bash
# Test it with your webcam
python gesture_activation/gesture_detector.py
# Press 'q' to quit, 'r' to reset
```

### Classical Fatigue Detection
The classical pipeline extracts **13 features per frame** using MediaPipe's 478 facial landmarks:

| Feature | Description |
|---|---|
| EAR | Eye Aspect Ratio (eye openness) |
| MAR | Mouth Aspect Ratio (yawning) |
| Head Ratio | Head pose estimation (nodding) |
| Rolling Mean/Std | Smoothed values over 15-frame window |
| EAR Velocity | Speed of eye closure |
| PERCLOS | % of eye closure over 30 frames |
| Blink Sum | Blink count in rolling window |

**Results on combined dataset (75,205 frames from 119 videos):**

| Model | Per-Frame Accuracy | Per-Video Accuracy |
|---|---|---|
| **SVM (RBF)** | **84.3%** | **91.7%** |
| Random Forest | 83.7% | 90.8% |
| KNN (k=7) | 82.5% | 89.9% |

Top features by importance: `ear_rolling_mean` (26%), `ear` (24%), `head_ratio_rolling_mean` (10%)

---

## 🔲 What's Left

### Task 1 — Deep Learning Fatigue Detection
Build a CNN or CNN-LSTM model in `fatigue_detection/modern/` that classifies awake vs sleepy. Use the same video data and compare results against the classical approach.

**Steps:**
1. Extract face crops or use landmark sequences from the existing videos
2. Build and train a CNN or CNN-LSTM model (PyTorch or TensorFlow)
3. Evaluate with the same video-based train/test split
4. Save the trained model as a `.pth` file

### Task 2 — Integration and Demo
Build `main.py` that runs the full pipeline:
1. System starts inactive
2. Gesture activation waits for open hand → thumbs up
3. Once activated, fatigue detection runs in real time
4. Display alerts when fatigue is detected
5. Record demo video showing the full flow

### Task 3 — Technical Report
Write the report covering: system description, methodology, experimental setup, results comparing classical vs modern, and discussion.

---
## 🛠️ Dependencies
opencv-python
mediapipe
numpy
scipy
scikit-learn
pandas