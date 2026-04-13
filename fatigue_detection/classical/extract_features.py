import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import csv
import os
import math

# --- Landmark indices for EAR and MAR ---
# MediaPipe FaceMesh 478 landmarks
# Left eye
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133

# Right eye
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263

# Mouth
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_LEFT = 78
MOUTH_RIGHT = 308


def compute_ear(landmarks, w, h):
    """Compute Eye Aspect Ratio for both eyes"""
    def eye_ratio(top, bottom, left, right):
        top_pt = np.array([landmarks[top].x * w, landmarks[top].y * h])
        bottom_pt = np.array([landmarks[bottom].x * w, landmarks[bottom].y * h])
        left_pt = np.array([landmarks[left].x * w, landmarks[left].y * h])
        right_pt = np.array([landmarks[right].x * w, landmarks[right].y * h])

        vertical = np.linalg.norm(top_pt - bottom_pt)
        horizontal = np.linalg.norm(left_pt - right_pt)
        return vertical / (horizontal + 1e-6)

    left_ear = eye_ratio(LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
    right_ear = eye_ratio(RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
    return (left_ear + right_ear) / 2.0


def compute_mar(landmarks, w, h):
    """Compute Mouth Aspect Ratio"""
    top = np.array([landmarks[MOUTH_TOP].x * w, landmarks[MOUTH_TOP].y * h])
    bottom = np.array([landmarks[MOUTH_BOTTOM].x * w, landmarks[MOUTH_BOTTOM].y * h])
    left = np.array([landmarks[MOUTH_LEFT].x * w, landmarks[MOUTH_LEFT].y * h])
    right = np.array([landmarks[MOUTH_RIGHT].x * w, landmarks[MOUTH_RIGHT].y * h])

    vertical = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)
    return vertical / (horizontal + 1e-6)


def compute_head_pose(landmarks, w, h):
    """Estimate head tilt using nose tip vs forehead vertical distance"""
    nose = np.array([landmarks[1].x * w, landmarks[1].y * h])
    forehead = np.array([landmarks[10].x * w, landmarks[10].y * h])
    chin = np.array([landmarks[152].x * w, landmarks[152].y * h])

    face_height = np.linalg.norm(forehead - chin)
    nose_to_forehead = np.linalg.norm(nose - forehead)

    # Ratio indicates head tilt (lower = head dropping)
    return nose_to_forehead / (face_height + 1e-6)


def process_video(video_path, label, detector):
    """Extract features from every frame of a video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Could not open {video_path}")
        return []

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    features = []
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            ear = compute_ear(lm, w, h)
            mar = compute_mar(lm, w, h)
            head_ratio = compute_head_pose(lm, w, h)

            features.append({
                "video": os.path.basename(video_path),
                "frame": frame_num,
                "ear": round(ear, 5),
                "mar": round(mar, 5),
                "head_ratio": round(head_ratio, 5),
                "label": label
            })

        frame_num += 1

    cap.release()
    return features


def main():
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        print("ERROR: face_landmarker.task not found. Run test_video.py first to download it.")
        return

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    all_features = []

    # Process awake videos
    awake_dir = "data/sofia/awake"
    awake_videos = sorted([f for f in os.listdir(awake_dir) if f.endswith((".MOV", ".mp4", ".mov"))])
    print(f"Processing {len(awake_videos)} AWAKE videos...")
    for i, vid in enumerate(awake_videos):
        print(f"  [{i+1}/{len(awake_videos)}] {vid}")
        feats = process_video(os.path.join(awake_dir, vid), "awake", detector)
        all_features.extend(feats)

    # Process sleepy videos
    sleepy_dir = "data/sofia/sleepy"
    sleepy_videos = sorted([f for f in os.listdir(sleepy_dir) if f.endswith((".MOV", ".mp4", ".mov"))])
    print(f"Processing {len(sleepy_videos)} SLEEPY videos...")
    for i, vid in enumerate(sleepy_videos):
        print(f"  [{i+1}/{len(sleepy_videos)}] {vid}")
        feats = process_video(os.path.join(sleepy_dir, vid), "sleepy", detector)
        all_features.extend(feats)

    # Save to CSV
    output_path = "fatigue_detection/classical/features.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "frame", "ear", "mar", "head_ratio", "label"])
        writer.writeheader()
        writer.writerows(all_features)

    print(f"\nDone! Extracted {len(all_features)} frames")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()