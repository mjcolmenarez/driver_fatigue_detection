import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time


class GestureDetector:
    """Detects hand gestures: open_hand, thumbs_up, fist, peace, unknown"""

    def __init__(self, model_path="hand_landmarker.task"):
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def _is_finger_extended(self, landmarks, finger_tip, finger_pip):
        """Check if a finger is extended (tip is above pip joint)"""
        # Using y-coordinate: lower y = higher in image
        return landmarks[finger_tip].y < landmarks[finger_pip].y

    def _is_thumb_extended(self, landmarks):
        """Check if thumb is extended (using x-distance from palm)"""
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        wrist = landmarks[0]

        # Determine hand orientation (left vs right) using wrist and middle finger mcp
        palm_dir = landmarks[5].x - wrist.x

        if palm_dir > 0:  # right-ish hand
            return thumb_tip.x > thumb_ip.x
        else:  # left-ish hand
            return thumb_tip.x < thumb_ip.x

    def classify_gesture(self, landmarks):
        """Classify the current hand gesture from landmarks"""
        thumb = self._is_thumb_extended(landmarks)
        index = self._is_finger_extended(landmarks, 8, 6)
        middle = self._is_finger_extended(landmarks, 12, 10)
        ring = self._is_finger_extended(landmarks, 16, 14)
        pinky = self._is_finger_extended(landmarks, 20, 18)

        fingers = [thumb, index, middle, ring, pinky]
        extended_count = sum(fingers)

        # Open hand: all 5 fingers extended
        if extended_count >= 4 and index and middle and ring:
            return "open_hand"

        # Thumbs up: only thumb extended
        if thumb and not index and not middle and not ring and not pinky:
            return "thumbs_up"

        # Peace sign: index and middle extended, others closed
        if index and middle and not ring and not pinky:
            return "peace"

        # Fist: no fingers extended
        if extended_count <= 1 and not index and not middle and not ring and not pinky:
            return "fist"

        return "unknown"

    def detect(self, frame):
        """Detect hand and classify gesture from a BGR frame"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_image)

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]
            gesture = self.classify_gesture(landmarks)
            return gesture, landmarks
        return None, None


class GestureSequenceValidator:
    """Validates a sequence of gestures performed in order within a time window"""

    def __init__(self, required_sequence, time_window=5.0, hold_time=0.5):
        """
        required_sequence: list of gesture names e.g. ["open_hand", "thumbs_up"]
        time_window: max seconds to complete the full sequence
        hold_time: seconds a gesture must be held to count
        """
        self.required_sequence = required_sequence
        self.time_window = time_window
        self.hold_time = hold_time
        self.reset()

    def reset(self):
        self.current_step = 0
        self.sequence_start_time = None
        self.gesture_start_time = None
        self.current_gesture_held = None
        self.activated = False

    def update(self, detected_gesture):
        """
        Call each frame with the detected gesture.
        Returns: (activated, status_message)
        """
        if self.activated:
            return True, "SYSTEM ACTIVE"

        now = time.time()
        expected = self.required_sequence[self.current_step]

        # Check time window
        if self.sequence_start_time and (now - self.sequence_start_time > self.time_window):
            self.reset()
            return False, "TIMEOUT - Sequence reset"

        # Is the detected gesture the one we expect?
        if detected_gesture == expected:
            # Start tracking hold time
            if self.current_gesture_held != detected_gesture:
                self.current_gesture_held = detected_gesture
                self.gesture_start_time = now

            # Check if held long enough
            if now - self.gesture_start_time >= self.hold_time:
                # Gesture confirmed
                if self.current_step == 0:
                    self.sequence_start_time = now

                self.current_step += 1
                self.current_gesture_held = None
                self.gesture_start_time = None

                if self.current_step >= len(self.required_sequence):
                    self.activated = True
                    return True, "ACTIVATION COMPLETE!"

                next_expected = self.required_sequence[self.current_step]
                return False, f"Step {self.current_step}/{len(self.required_sequence)} done! Now show: {next_expected}"

            remaining = self.hold_time - (now - self.gesture_start_time)
            return False, f"Hold {expected}... ({remaining:.1f}s)"
        else:
            # Wrong gesture or no gesture
            self.current_gesture_held = None
            self.gesture_start_time = None

            if self.current_step == 0:
                return False, f"Show gesture: {expected}"
            else:
                return False, f"Step {self.current_step}/{len(self.required_sequence)} - Now show: {expected}"


def main():
    """Test the gesture activation with webcam"""
    print("=" * 50)
    print("GESTURE ACTIVATION TEST")
    print("Sequence: OPEN HAND -> THUMBS UP")
    print("Hold each gesture for 0.5 seconds")
    print("Complete within 5 seconds")
    print("Press 'q' to quit, 'r' to reset")
    print("=" * 50)

    detector = GestureDetector()
    validator = GestureSequenceValidator(
        required_sequence=["open_hand", "thumbs_up"],
        time_window=5.0,
        hold_time=0.5
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror
        gesture, landmarks = detector.detect(frame)

        # Update validator
        activated, status = validator.update(gesture)

        # Draw hand landmarks
        if landmarks:
            h, w = frame.shape[:2]
            for lm in landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)

        # Display info
        color = (0, 255, 0) if activated else (0, 165, 255)
        gesture_text = gesture if gesture else "No hand"

        cv2.putText(frame, f"Gesture: {gesture_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, status, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if activated:
            cv2.putText(frame, "SYSTEM ACTIVATED", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Gesture Activation", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            validator.reset()
            print("Sequence reset!")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()