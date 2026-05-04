"""
Extract face crops from labeled video files for ResNet-34 training.

Reads videos from data/sofia/{awake,sleepy} and data/matteo/{awake,sleepy},
detects the face in each sampled frame using MediaPipe, crops and resizes it
to 224x224, and saves the result as a JPEG.

Output:
    data/crops/awake/        - face crop JPEGs labeled awake
    data/crops/sleepy/       - face crop JPEGs labeled sleepy
    data/crops/manifest.csv  - metadata (path, video, person, label, frame)

Usage (from project root):
    python fatigue_detection/modern/extract_face_crops.py
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os
import pandas as pd
from pathlib import Path

SAMPLE_EVERY_N_FRAMES = 5    # ~6 fps from a 30fps video (richer temporal resolution)
CROP_SIZE = 224
PADDING = 0.25               # fractional padding around the face bbox
FACE_MODEL = "face_landmarker.task"


def get_face_bbox(landmarks, w, h, padding=PADDING):
    """Compute a padded bounding box from all 478 MediaPipe face landmarks."""
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bw = x_max - x_min
    bh = y_max - y_min
    pad_x = bw * padding
    pad_y = bh * padding
    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(w, int(x_max + pad_x))
    y2 = min(h, int(y_max + pad_y))
    return x1, y1, x2, y2


def process_video(video_path, label, person, detector, out_dir):
    """Extract and save face crops from a single video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"    ERROR: Could not open {video_path}")
        return []

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    records = []
    frame_idx = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % SAMPLE_EVERY_N_FRAMES == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)

            if result.face_landmarks:
                lm = result.face_landmarks[0]
                x1, y1, x2, y2 = get_face_bbox(lm, w, h)

                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
                    stem = Path(video_path).stem
                    filename = f"{person}_{stem}_f{frame_idx:06d}.jpg"
                    crop_path = os.path.join(out_dir, filename)
                    cv2.imwrite(crop_path, crop_resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    records.append({
                        "crop_path": crop_path,
                        "video": Path(video_path).name,
                        "person": person,
                        "label": label,
                        "frame": frame_idx,
                    })
                    saved += 1

        frame_idx += 1

    cap.release()
    print(f"    {saved} crops saved from {frame_idx} frames")
    return records


def main():
    if not os.path.exists(FACE_MODEL):
        print(f"ERROR: {FACE_MODEL} not found. Run from the project root.")
        return

    base_options = python.BaseOptions(model_asset_path=FACE_MODEL)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    os.makedirs("data/crops/awake", exist_ok=True)
    os.makedirs("data/crops/sleepy", exist_ok=True)

    # Map (person, label) → actual folder on disk
    video_folders = {
        ("sofia",  "awake"):  "fatigue_detection/modern/Sofia- awake",
        ("sofia",  "sleepy"): "fatigue_detection/modern/Sofia- sleepy",
        ("matteo", "awake"):  "fatigue_detection/modern/Matteo - awake",
        ("matteo", "sleepy"): "fatigue_detection/modern/Matteo - Sleepy",
    }
    all_records = []

    for (person, label), folder in video_folders.items():
        if not os.path.exists(folder):
            print(f"Skipping {folder} (not found)")
            continue
        out_dir = f"data/crops/{label}"
        videos = sorted(
            f for f in os.listdir(folder)
            if f.lower().endswith((".mov", ".mp4"))
        )
        print(f"\nProcessing {len(videos)} {label.upper()} videos from {person} ({folder})...")
        for i, vid in enumerate(videos):
            print(f"  [{i+1}/{len(videos)}] {vid}")
            records = process_video(
                os.path.join(folder, vid), label, person, detector, out_dir
            )
            all_records.extend(records)

    if not all_records:
        print("\nNo crops were extracted. Check that your video folders exist and")
        print("contain .mov or .mp4 files, then re-run this script.")
        return

    manifest = pd.DataFrame(all_records)
    manifest.to_csv("data/crops/manifest.csv", index=False)

    print(f"\nDone!")
    print(f"  Total crops : {len(manifest)}")
    print(f"  Awake       : {len(manifest[manifest['label'] == 'awake'])}")
    print(f"  Sleepy      : {len(manifest[manifest['label'] == 'sleepy'])}")
    print(f"  Manifest    : data/crops/manifest.csv")


if __name__ == "__main__":
    main()
