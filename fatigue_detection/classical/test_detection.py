import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import joblib
import collections

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


def main():
    # Load trained model and scaler
    model = joblib.load("fatigue_detection/classical/best_model_all.pkl")
    scaler = joblib.load("fatigue_detection/classical/scaler_all.pkl")

    # Load face detector
    base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    # Test on one awake and one sleepy video
    test_videos = [
        ("data/sofia/awake/IMG_9498.MOV", "SHOULD BE: AWAKE"),
        ("data/sofia/sleepy/IMG_9529.MOV", "SHOULD BE: SLEEPY"),
    ]

    window = 15

    for video_path, expected in test_videos:
        print(f"\n{'='*50}")
        print(f"Testing: {video_path}")
        print(f"{expected}")
        print(f"{'='*50}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Could not open {video_path}")
            continue

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Buffers for rolling features
        ear_buffer = collections.deque(maxlen=window * 2)
        mar_buffer = collections.deque(maxlen=window * 2)
        head_buffer = collections.deque(maxlen=window * 2)

        frame_num = 0
        predictions = {"awake": 0, "sleepy": 0}

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
                head = compute_head_ratio(lm, w, h)

                ear_buffer.append(ear)
                mar_buffer.append(mar)
                head_buffer.append(head)

                if len(ear_buffer) >= window:
                    ear_arr = np.array(list(ear_buffer))
                    mar_arr = np.array(list(mar_buffer))
                    head_arr = np.array(list(head_buffer))

                    # Compute all 13 features
                    ear_rolling_mean = np.mean(ear_arr[-window:])
                    ear_rolling_std = np.std(ear_arr[-window:])
                    mar_rolling_mean = np.mean(mar_arr[-window:])
                    mar_rolling_std = np.std(mar_arr[-window:])
                    head_rolling_mean = np.mean(head_arr[-window:])
                    head_rolling_std = np.std(head_arr[-window:])

                    ear_velocity = ear - ear_arr[-2] if len(ear_arr) >= 2 else 0

                    ear_threshold = np.quantile(ear_arr, 0.25)
                    eye_closed = 1 if ear < ear_threshold else 0

                    recent_closed = [1 if e < ear_threshold else 0 for e in ear_arr[-window:]]
                    blink_rolling_sum = sum(recent_closed)

                    perclos_window = ear_arr[-(window * 2):]
                    perclos_closed = [1 if e < ear_threshold else 0 for e in perclos_window]
                    perclos = np.mean(perclos_closed)

                    features = np.array([[
                        ear, mar, head,
                        ear_rolling_mean, ear_rolling_std,
                        mar_rolling_mean, mar_rolling_std,
                        head_rolling_mean, head_rolling_std,
                        ear_velocity, eye_closed, blink_rolling_sum, perclos
                    ]])

                    features_scaled = scaler.transform(features)
                    prediction = model.predict(features_scaled)[0]
                    label = "awake" if prediction == 0 else "sleepy"
                    predictions[label] += 1

                    # Show on resized frame
                    display = cv2.resize(frame, (540, 960))
                    color = (0, 255, 0) if label == "awake" else (0, 0, 255)
                    cv2.putText(display, f"State: {label.upper()}", (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    cv2.putText(display, f"EAR: {ear:.3f}", (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display, f"MAR: {mar:.3f}", (10, 110),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display, f"PERCLOS: {perclos:.2f}", (10, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    cv2.imshow("Fatigue Detection Test", display)
                    # Slow down to roughly real-time
                    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
                        break

            frame_num += 1

        cap.release()

        total = predictions["awake"] + predictions["sleepy"]
        if total > 0:
            print(f"Results: {predictions['awake']} awake frames ({predictions['awake']/total*100:.1f}%)")
            print(f"         {predictions['sleepy']} sleepy frames ({predictions['sleepy']/total*100:.1f}%)")
            majority = "AWAKE" if predictions["awake"] > predictions["sleepy"] else "SLEEPY"
            print(f"Overall prediction: {majority}")

    cv2.destroyAllWindows()
    print("\nDone! Press 'q' during playback to skip to next video.")


if __name__ == "__main__":
    main()