import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# Download face landmarker model if not present
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading face landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, model_path)
    print("Download complete!")

# Open one of Sofia's videos
video_path = "data/sofia/awake/IMG_9498.MOV"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video")
    exit()

# Video info
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps if fps > 0 else 0

print(f"Resolution: {width}x{height}")
print(f"FPS: {fps}")
print(f"Total frames: {total_frames}")
print(f"Duration: {duration:.1f} seconds")

# Test face detection on first frame
ret, frame = cap.read()
if ret:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    result = detector.detect(mp_image)

    if result.face_landmarks:
        print(f"Face detected! {len(result.face_landmarks[0])} landmarks found")
    else:
        print("No face detected in first frame")

cap.release()
print("Test complete!")