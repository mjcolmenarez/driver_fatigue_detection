"""
Driver Fatigue Detection System - Main Integration Pipeline

Real-time system combining:
1. Gesture-based activation (open hand → thumbs up)
2. Classical fatigue detection (eye closure, yawning, head pose)

Usage:
    python main.py [--camera CAMERA_ID] [--video VIDEO_PATH] [--output OUTPUT_PATH]

Controls:
    'q' or ESC - Quit
    'r' - Reset gesture sequence
    's' - Start/stop recording output video
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import joblib
import argparse
import sys
import time
from pathlib import Path
from collections import deque
from datetime import datetime

# Import gesture detector
sys.path.insert(0, str(Path(__file__).parent))
from gesture_activation.gesture_detector import GestureDetector, GestureSequenceValidator


# ========================================================================
# FACIAL FEATURE EXTRACTION (Real-time)
# ========================================================================

class FaceFeatureExtractor:
    """Extracts facial features (EAR, MAR, head ratio, etc.) in real-time"""
    
    # Landmark indices
    LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
    LEFT_EYE_LEFT, LEFT_EYE_RIGHT = 33, 133
    RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374
    RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT = 362, 263
    MOUTH_TOP, MOUTH_BOTTOM = 13, 14
    MOUTH_LEFT, MOUTH_RIGHT = 78, 308
    NOSE_TIP, NOSE_BRIDGE = 1, 10
    CHIN = 152
    
    def __init__(self, face_model_path="face_landmarker.task"):
        """Initialize MediaPipe FaceLandmarker"""
        if not Path(face_model_path).exists():
            self._download_model(face_model_path)
        
        base_options = python.BaseOptions(model_asset_path=face_model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)
    
    @staticmethod
    def _download_model(model_path):
        """Download face landmarker model if not present"""
        import urllib.request
        print(f"Downloading face landmarker model to {model_path}...")
        url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print("Download complete!")
    
    def compute_ear(self, landmarks, w, h):
        """Compute Eye Aspect Ratio (one eye)"""
        def eye_ratio(t, b, l, r):
            tv = np.array([landmarks[t].x * w, landmarks[t].y * h])
            bv = np.array([landmarks[b].x * w, landmarks[b].y * h])
            lv = np.array([landmarks[l].x * w, landmarks[l].y * h])
            rv = np.array([landmarks[r].x * w, landmarks[r].y * h])
            return np.linalg.norm(tv - bv) / (np.linalg.norm(lv - rv) + 1e-6)
        
        left = eye_ratio(self.LEFT_EYE_TOP, self.LEFT_EYE_BOTTOM, 
                        self.LEFT_EYE_LEFT, self.LEFT_EYE_RIGHT)
        right = eye_ratio(self.RIGHT_EYE_TOP, self.RIGHT_EYE_BOTTOM,
                         self.RIGHT_EYE_LEFT, self.RIGHT_EYE_RIGHT)
        return (left + right) / 2.0
    
    def compute_mar(self, landmarks, w, h):
        """Compute Mouth Aspect Ratio (yawning indicator)"""
        mouth_top = np.array([landmarks[self.MOUTH_TOP].x * w, landmarks[self.MOUTH_TOP].y * h])
        mouth_bottom = np.array([landmarks[self.MOUTH_BOTTOM].x * w, landmarks[self.MOUTH_BOTTOM].y * h])
        mouth_left = np.array([landmarks[self.MOUTH_LEFT].x * w, landmarks[self.MOUTH_LEFT].y * h])
        mouth_right = np.array([landmarks[self.MOUTH_RIGHT].x * w, landmarks[self.MOUTH_RIGHT].y * h])
        
        vertical = np.linalg.norm(mouth_top - mouth_bottom)
        horizontal = np.linalg.norm(mouth_left - mouth_right)
        return vertical / (horizontal + 1e-6)
    
    def compute_head_ratio(self, landmarks, w, h):
        """Compute head pose ratio (nodding indicator)"""
        nose_tip = np.array([landmarks[self.NOSE_TIP].x * w, landmarks[self.NOSE_TIP].y * h])
        nose_bridge = np.array([landmarks[self.NOSE_BRIDGE].x * w, landmarks[self.NOSE_BRIDGE].y * h])
        chin = np.array([landmarks[self.CHIN].x * w, landmarks[self.CHIN].y * h])
        
        forehead_dist = np.linalg.norm(nose_tip - nose_bridge)
        chin_dist = np.linalg.norm(nose_tip - chin)
        return forehead_dist / (chin_dist + 1e-6)
    
    def extract_frame_features(self, frame):
        """Extract raw features from a single frame"""
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)
        
        if not result.face_landmarks:
            return None
        
        landmarks = result.face_landmarks[0]
        
        # Extract base features
        ear = self.compute_ear(landmarks, w, h)
        mar = self.compute_mar(landmarks, w, h)
        head_ratio = self.compute_head_ratio(landmarks, w, h)
        
        return {
            "ear": ear,
            "mar": mar,
            "head_ratio": head_ratio,
        }


# ========================================================================
# REAL-TIME FATIGUE DETECTION PIPELINE
# ========================================================================

class FatigueDetectionPipeline:
    """Main real-time pipeline combining gesture activation and fatigue detection"""
    
    FEATURE_COLS = [
        "ear", "mar", "head_ratio",
        "ear_rolling_mean", "ear_rolling_std",
        "mar_rolling_mean", "mar_rolling_std",
        "head_ratio_rolling_mean", "head_ratio_rolling_std",
        "ear_velocity", "eye_closed", "blink_rolling_sum", "perclos"
    ]
    
    EAR_THRESHOLD = 0.2  # Eye closure threshold
    ROLLING_WINDOW = 15  # Frames for rolling statistics
    PERCLOS_WINDOW = 30  # Frames for PERCLOS calculation
    
    def __init__(self, model_path="fatigue_detection/classical/best_model_all.pkl",
                 scaler_path="fatigue_detection/classical/scaler_all.pkl",
                 face_model_path="face_landmarker.task",
                 hand_model_path="hand_landmarker.task"):
        """Initialize the complete pipeline"""
        
        # Load models
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Initialize feature extractors
        self.face_extractor = FaceFeatureExtractor(face_model_path)
        self.gesture_detector = GestureDetector(hand_model_path)
        
        # Initialize gesture validator (open_hand → thumbs_up)
        self.gesture_validator = GestureSequenceValidator(
            required_sequence=["open_hand", "thumbs_up"],
            time_window=5.0,
            hold_time=0.5
        )
        
        # Data buffers for temporal features
        self.feature_history = deque(maxlen=self.ROLLING_WINDOW)
        self.eye_closure_history = deque(maxlen=self.PERCLOS_WINDOW)
        self.last_ear = None
        self.blink_count = 0
        
        # State
        self.system_active = False
        self.last_prediction = "UNKNOWN"
        self.last_confidence = 0.0
        self.frame_count = 0
    
    def compute_temporal_features(self):
        """Compute temporal features from historical data"""
        if len(self.feature_history) < self.ROLLING_WINDOW:
            return None  # Not enough history yet
        
        features = {}
        history_list = list(self.feature_history)
        ears = np.array([f["ear"] for f in history_list])
        mars = np.array([f["mar"] for f in history_list])
        head_ratios = np.array([f["head_ratio"] for f in history_list])
        
        # Rolling statistics
        features["ear_rolling_mean"] = np.mean(ears)
        features["ear_rolling_std"] = np.std(ears)
        features["mar_rolling_mean"] = np.mean(mars)
        features["mar_rolling_std"] = np.std(mars)
        features["head_ratio_rolling_mean"] = np.mean(head_ratios)
        features["head_ratio_rolling_std"] = np.std(head_ratios)
        
        # EAR velocity (frame-to-frame change)
        features["ear_velocity"] = abs(ears[-1] - ears[-2]) if len(ears) >= 2 else 0.0
        
        # Eye closure flag
        features["eye_closed"] = 1.0 if ears[-1] < self.EAR_THRESHOLD else 0.0
        
        # Blink sum (count of closed eyes in window)
        features["blink_rolling_sum"] = sum([1 for e in ears if e < self.EAR_THRESHOLD])
        
        # PERCLOS (percentage of closure in last 30 frames)
        closure_percent = len(self.eye_closure_history) / max(len(self.eye_closure_history), 1)
        features["perclos"] = closure_percent
        
        return features
    
    def process_frame(self, frame):
        """Process a single frame and return detection results"""
        self.frame_count += 1
        h, w = frame.shape[:2]
        
        # 1. Detect hand gesture
        gesture, _ = self.gesture_detector.detect(frame)
        
        # 2. Update gesture activation state
        activated, status_message = self.gesture_validator.update(gesture)
        if activated:
            self.system_active = True
        
        # 3. Extract face features
        face_features = self.face_extractor.extract_frame_features(frame)
        
        results = {
            "frame_count": self.frame_count,
            "gesture": gesture,
            "gesture_status": status_message,
            "system_active": self.system_active,
            "prediction": "UNKNOWN",
            "confidence": 0.0,
            "face_features": face_features,
        }
        
        # If face detected, add to history
        if face_features:
            self.feature_history.append(face_features)
            eye_closed = face_features["ear"] < self.EAR_THRESHOLD
            self.eye_closure_history.append(eye_closed)
            self.last_ear = face_features["ear"]
        
        # 4. Classify fatigue (if system active and enough history)
        if self.system_active and len(self.feature_history) >= self.ROLLING_WINDOW:
            temporal_features = self.compute_temporal_features()
            if temporal_features:
                # Build feature vector in correct order
                feature_vector = np.array([
                    self.feature_history[-1]["ear"],
                    self.feature_history[-1]["mar"],
                    self.feature_history[-1]["head_ratio"],
                    temporal_features["ear_rolling_mean"],
                    temporal_features["ear_rolling_std"],
                    temporal_features["mar_rolling_mean"],
                    temporal_features["mar_rolling_std"],
                    temporal_features["head_ratio_rolling_mean"],
                    temporal_features["head_ratio_rolling_std"],
                    temporal_features["ear_velocity"],
                    temporal_features["eye_closed"],
                    temporal_features["blink_rolling_sum"],
                    temporal_features["perclos"],
                ]).reshape(1, -1)
                
                # Scale and predict
                feature_vector_scaled = self.scaler.transform(feature_vector)
                prediction = self.model.predict(feature_vector_scaled)[0]
                confidence = max(self.model.decision_function(feature_vector_scaled)[0])
                
                self.last_prediction = "SLEEPY" if prediction == 1 else "AWAKE"
                self.last_confidence = abs(confidence)
                
                results["prediction"] = self.last_prediction
                results["confidence"] = self.last_confidence
                results["temporal_features"] = temporal_features
        
        return results
    
    def reset(self):
        """Reset the gesture sequence validator"""
        self.gesture_validator.reset()
        self.system_active = False
        self.feature_history.clear()
        self.eye_closure_history.clear()


# ========================================================================
# MAIN APPLICATION
# ========================================================================

class DemoApp:
    """Real-time demo application with visualization"""
    
    def __init__(self, source=0, output_path=None):
        """
        Args:
            source: Camera device ID (0 for default) or video file path
            output_path: Path to save output video (optional)
        """
        self.pipeline = FatigueDetectionPipeline()
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        
        self.source = source
        self.output_path = output_path
        self.recording = False
        self.writer = None
        self.init_writer(output_path)
    
    def init_writer(self, output_path):
        """Initialize video writer if output path specified"""
        if output_path:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            self.recording = True
            print(f"Recording to: {output_path}")
    
    def draw_ui(self, frame, results):
        """Draw UI elements on frame"""
        h, w = frame.shape[:2]
        
        # Top section: System status
        bg_color = (50, 50, 50)
        cv2.rectangle(frame, (0, 0), (w, 120), bg_color, -1)
        
        # Gesture status
        gesture_text = f"Gesture: {results['gesture'] or 'None'}"
        cv2.putText(frame, gesture_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Activation status
        status_color = (0, 255, 0) if results['system_active'] else (0, 0, 255)
        status_text = "SYSTEM: ACTIVE" if results['system_active'] else "SYSTEM: INACTIVE"
        cv2.putText(frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 2)
        
        # Gesture sequence hint
        hint_text = results['gesture_status']
        cv2.putText(frame, hint_text, (w - 400, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Bottom section: Fatigue prediction (if active)
        if results['system_active']:
            cv2.rectangle(frame, (0, h - 100), (w, h), bg_color, -1)
            
            pred_color = (0, 0, 255) if results['prediction'] == "SLEEPY" else (0, 255, 0)
            pred_text = f"Prediction: {results['prediction']}"
            cv2.putText(frame, pred_text, (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, pred_color, 3)
            
            conf_text = f"Confidence: {results['confidence']:.2f}"
            cv2.putText(frame, conf_text, (20, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Recording indicator
        if self.recording:
            cv2.circle(frame, (w - 30, 30), 8, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return frame
    
    def run(self):
        """Main application loop"""
        print("\n" + "=" * 60)
        print("Driver Fatigue Detection System - Real-time Demo")
        print("=" * 60)
        print("\nControls:")
        print("  'q' or ESC  - Quit")
        print("  'r'         - Reset gesture sequence")
        print("  's'         - Start/stop recording output video")
        print("=" * 60 + "\n")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of video or camera disconnected")
                break
            
            # Process frame
            results = self.pipeline.process_frame(frame)
            
            # Draw visualization
            frame = self.draw_ui(frame, results)
            
            # Write to output video if recording
            if self.recording and self.writer:
                self.writer.write(frame)
            
            # Display
            cv2.imshow("Driver Fatigue Detection", frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 = ESC
                print("Quitting...")
                break
            elif key == ord('r'):
                self.pipeline.reset()
                print("Gesture sequence reset")
            elif key == ord('s'):
                if self.output_path:
                    self.recording = not self.recording
                    print(f"Recording: {'ON' if self.recording else 'OFF'}")
                else:
                    print("No output path specified, cannot record")
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.cap.release()
        if self.writer:
            self.writer.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")


# ========================================================================
# ENTRY POINT
# ========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Driver Fatigue Detection System - Real-time Pipeline"
    )
    parser.add_argument(
        "--camera", type=int, default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Video file path (if not using camera)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output video file path for recording"
    )
    
    args = parser.parse_args()
    
    # Determine video source
    source = args.video if args.video else args.camera
    
    # Create and run app
    try:
        app = DemoApp(source=source, output_path=args.output)
        app.run()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
