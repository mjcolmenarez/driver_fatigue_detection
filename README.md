# 🚗 Driver Fatigue Detection with Gesture-Based Activation

> Computer Vision group project, a real-time driver fatigue detection system with a gesture-based activation mechanism. Built with MediaPipe, OpenCV, PyTorch, and scikit-learn.

---

## 📌 Overview

This system implements a **hybrid driver fatigue detection pipeline** with two main components:

1. **Gesture-Based Activation**: the system stays inactive until the driver performs a specific hand gesture sequence (**open hand → thumbs up**), each held for 0.5 s within a 5-second window.
2. **Fatigue Detection**: once activated, the system monitors the driver for signs of drowsiness using two parallel pipelines:
   - **Classical pipeline** : handcrafted features (EAR, MAR, head ratio, PERCLOS, etc.) + SVM classifier
   - **Modern pipeline** : ResNet-34 CNN encoder + 2-layer LSTM head (CNN-LSTM)


---

## 📁 Project Structure

```
driver_fatigue_detection/
│
├── main.py                          # ← Main entry point (full real-time pipeline)
├── test_video.py                    # Quick smoke-test with a video file
├── requirements.txt
│
├── face_landmarker.task             # MediaPipe face model (bundled)
├── hand_landmarker.task             # MediaPipe hand model (bundled)
│
├── gesture_activation/
│   └── gesture_detector.py         # GestureDetector + GestureSequenceValidator
│
├── fatigue_detection/
│   ├── classical/
│   │   ├── extract_features.py     # Extract EAR/MAR/head features from videos
│   │   ├── extract_features_v2.py  # Extended feature set
│   │   ├── extract_all_features.py # Combined dataset extraction
│   │   ├── train_classifier.py     # Train SVM / RF / KNN (per-video features)
│   │   ├── train_classifier_v2.py  # Train on extended feature set
│   │   ├── train_classifier_all.py # Train on full combined dataset
│   │   ├── test_detection.py       # Offline evaluation script
│   │   ├── best_model_all.pkl      # Trained SVM model ← used by main.py
│   │   ├── scaler_all.pkl          # Feature scaler   ← used by main.py
│   │   └── features_all.csv
│   │
│   └── modern/
│       ├── extract_face_crops.py   # Step 1: extract face crops from videos
│       ├── extract_cnn_features.py # Step 2: encode crops with ResNet-34
│       ├── dataset.py              # PyTorch Dataset for sequence windows
│       ├── model.py                # CNNEncoder (ResNet-34) + FatigueLSTMHead
│       ├── train.py                # Train the LSTM head
│       ├── evaluate.py             # Offline evaluation
│       └── cnn_lstm_model.pth      # ✅ Trained CNN-LSTM checkpoint ← used by main.py
│
└── data/
    └── crops/
        ├── features/               # Pre-computed .npy CNN feature files
        ├── features_manifest.csv
        └── manifest.csv
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/mjcolmenarez/driver_fatigue_detection.git
cd driver_fatigue_detection
```

### 2. Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on PyTorch:** `requirements.txt` pulls the default CPU build. If you have a CUDA GPU, install PyTorch first from [pytorch.org](https://pytorch.org/get-started/locally/) before running the command above.

### 4. (Optional) Download training videos

Only needed if you want to **retrain** the models. For running the pre-trained system, skip this — all `.pkl` and `.pth` checkpoints are already included in the repo.

Download from [Google Drive](https://drive.google.com/drive/folders/12Bj_WIQJwLvqsWceDXsqbualINhLOvw0?usp=drive_link) and place the folders inside `fatigue_detection/modern/` 

---

## 🚀 Running the System

### Run the full real-time pipeline

```bash
python main.py
```

Launches the **hybrid** system (SVM + CNN-LSTM) using your default webcam. The system starts **inactive**, perform the gesture sequence to activate fatigue detection.

#### Command-line options

| Flag | Default | Description |
|---|---|---|
| `--camera ID` | `0` | Webcam device index |
| `--video PATH` | — | Use a pre-recorded video file instead of webcam |
| `--output PATH` | — | Save the annotated output to a video file |
| `--mode MODE` | `hybrid` | `classical`, `modern`, or `hybrid` |

```bash
# Classical SVM only
python main.py --mode classical

# CNN-LSTM only
python main.py --mode modern

# Hybrid (both side-by-side) — default
python main.py --mode hybrid

# Use a different camera (e.g. index 1 for external webcam)
python main.py --camera 1

# Run on a pre-recorded video
python main.py --video path/to/video.mp4

# Save annotated output to a file
python main.py --output demo_output.mp4

# Full example: hybrid on webcam 0, saving output
python main.py --mode hybrid --camera 0 --output demo_output.mp4
```

> **Windows users:** open `main.py` and swap the two `cv2.VideoCapture` lines in `DemoApp.__init__` — comment out `CAP_AVFOUNDATION` and uncomment `CAP_DSHOW`.

#### Keyboard controls

| Key | Action |
|---|---|
| `q` or `ESC` | Quit |
| `r` | Reset the gesture sequence |
| `s` | Toggle output video recording on/off |

---

### Gesture activation sequence

The system is **inactive by default**. To activate it:

1. Show an **open hand** (all 5 fingers extended) to the camera, hold for **0.5 seconds**.
2. Then show a **thumbs up**, hold for **0.5 seconds**.
3. Both gestures must be performed within **5 seconds** of each other.
4. The HUD banner changes from `SYSTEM: INACTIVE` (red) to `SYSTEM: ACTIVE` (green).

If the sequence is wrong or times out, the system stays inactive. Press `r` to reset and try again.

---

### Test gesture detection in isolation

```bash
python gesture_activation/gesture_detector.py
```

Opens your webcam, prints the detected gesture label in real time, and shows the activation progress. Press `q` to quit or `r` to reset.

---

### Quick smoke-test

```bash
python test_video.py
```

Runs the classical pipeline on a bundled sample clip to verify the installation is working correctly.

---

## 🔁 Retraining the Models (Optional)

### Classical pipeline

```bash
# Step 1 — extract per-frame features from all videos
python fatigue_detection/classical/extract_all_features.py

# Step 2 — train and evaluate classifiers
python fatigue_detection/classical/train_classifier_all.py
```

Outputs: `best_model_all.pkl` and `scaler_all.pkl` inside `fatigue_detection/classical/`

### Modern pipeline (CNN-LSTM)

```bash
# Step 1 — extract face crops from every video frame
python fatigue_detection/modern/extract_face_crops.py

# Step 2 — encode crops with the frozen ResNet-34 backbone
python fatigue_detection/modern/extract_cnn_features.py

# Step 3 — train the LSTM head on the pre-computed features
python fatigue_detection/modern/train.py

# Step 4 — evaluate on the held-out test split
python fatigue_detection/modern/evaluate.py
```

Output: `fatigue_detection/modern/cnn_lstm_model.pth`

---

## 🛠️ Dependencies

```
opencv-python
mediapipe
numpy
scipy
scikit-learn
pandas
torch
torchvision
```
