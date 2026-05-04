import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import os

# --- Landmark indices ---
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
LEFT_EYE_LEFT, LEFT_EYE_RIGHT = 33, 133
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374
RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT = 362, 263
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 78, 308


def compute_ear(lm, w, h):
    def eye_ratio(t, b, l, r):
        tv = np.array([lm[t].x * w, lm[t].y * h])
        bv = np.array([lm[b].x * w, lm[b].y * h])
        lv = np.array([lm[l].x * w, lm[l].y * h])
        rv = np.array([lm[r].x * w, lm[r].y * h])
        return np.linalg.norm(tv - bv) / (np.linalg.norm(lv - rv) + 1e-6)
    left = eye_ratio(LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
    right = eye_ratio(RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
    return (left + right) / 2.0


def compute_mar(lm, w, h):
    top = np.array([lm[MOUTH_TOP].x * w, lm[MOUTH_TOP].y * h])
    bot = np.array([lm[MOUTH_BOTTOM].x * w, lm[MOUTH_BOTTOM].y * h])
    left = np.array([lm[MOUTH_LEFT].x * w, lm[MOUTH_LEFT].y * h])
    right = np.array([lm[MOUTH_RIGHT].x * w, lm[MOUTH_RIGHT].y * h])
    return np.linalg.norm(top - bot) / (np.linalg.norm(left - right) + 1e-6)


def compute_head_ratio(lm, w, h):
    nose = np.array([lm[1].x * w, lm[1].y * h])
    forehead = np.array([lm[10].x * w, lm[10].y * h])
    chin = np.array([lm[152].x * w, lm[152].y * h])
    face_h = np.linalg.norm(forehead - chin)
    return np.linalg.norm(nose - forehead) / (face_h + 1e-6)


def process_video(video_path, label, person, detector, window=15):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ERROR: Could not open {video_path}")
        return pd.DataFrame()

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if result.face_landmarks:
            lm = result.face_landmarks[0]
            raw_frames.append({
                "ear": compute_ear(lm, w, h),
                "mar": compute_mar(lm, w, h),
                "head_ratio": compute_head_ratio(lm, w, h),
            })
        else:
            raw_frames.append({"ear": np.nan, "mar": np.nan, "head_ratio": np.nan})

    cap.release()

    if not raw_frames:
        return pd.DataFrame()

    df = pd.DataFrame(raw_frames)
    df = df.interpolate().bfill().ffill()

    for col in ["ear", "mar", "head_ratio"]:
        df[f"{col}_rolling_mean"] = df[col].rolling(window, min_periods=1).mean()
        df[f"{col}_rolling_std"] = df[col].rolling(window, min_periods=1).std().fillna(0)

    df["ear_velocity"] = df["ear"].diff().fillna(0)

    ear_threshold = df["ear"].quantile(0.25)
    df["eye_closed"] = (df["ear"] < ear_threshold).astype(int)
    df["blink_rolling_sum"] = df["eye_closed"].rolling(window, min_periods=1).sum()
    df["perclos"] = df["eye_closed"].rolling(window * 2, min_periods=1).mean()

    df["video"] = os.path.basename(video_path)
    df["person"] = person
    df["frame"] = range(len(df))
    df["label"] = label

    return df


def main():
    model_path = "face_landmarker.task"
    if not os.path.exists(model_path):
        print("ERROR: face_landmarker.task not found")
        return

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    all_data = []

    # Process both people
    video_folders = {
        ("sofia",  "awake"):  "fatigue_detection/modern/Sofia- awake",
        ("sofia",  "sleepy"): "fatigue_detection/modern/Sofia- sleepy",
        ("matteo", "awake"):  "fatigue_detection/modern/Matteo - awake",
        ("matteo", "sleepy"): "fatigue_detection/modern/Matteo - Sleepy",
    }

    for (person, label), folder in video_folders.items():
            if not os.path.exists(folder):
                print(f"Skipping {folder} (not found)")
                continue
            videos = sorted([f for f in os.listdir(folder) if f.lower().endswith((".mov", ".mp4"))])
            print(f"Processing {len(videos)} {label.upper()} videos from {person}...")
            for i, vid in enumerate(videos):
                print(f"  [{i+1}/{len(videos)}] {vid}")
                df = process_video(os.path.join(folder, vid), label, person, detector)
                all_data.append(df)

    result = pd.concat(all_data, ignore_index=True)
    output = "fatigue_detection/classical/features_all.csv"
    result.to_csv(output, index=False)

    print(f"\nDone! {len(result)} total frames")
    print(f"Sofia:  {len(result[result['person'] == 'sofia'])} frames")
    print(f"Matteo: {len(result[result['person'] == 'matteo'])} frames")
    print(f"Awake:  {len(result[result['label'] == 'awake'])}")
    print(f"Sleepy: {len(result[result['label'] == 'sleepy'])}")
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()