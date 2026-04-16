#!/usr/bin/env python3
"""
Debug tool to test gesture detection in real-time
Shows what the system sees vs what you're showing
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import sys
from gesture_activation.gesture_detector import GestureDetector

def main():
    print("=" * 70)
    print("GESTURE DETECTION DEBUG - Real-time Diagnostic")
    print("=" * 70)
    print("\nTesting hand gesture recognition")
    print("Show different gestures to the camera:")
    print("  - Open hand (all fingers extended)")
    print("  - Thumbs up (only thumb extended)")
    print("  - Peace sign (index + middle)")
    print("  - Fist (no fingers extended)")
    print("\nPress 'q' to quit\n")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not access camera")
        print("Ensure you've granted camera permissions in System Preferences")
        return False
    
    detector = GestureDetector("hand_landmarker.task")
    
    frame_count = 0
    gesture_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        h, w = frame.shape[:2]
        
        # Detect gesture
        gesture, landmarks = detector.detect(frame)
        
        if gesture:
            gesture_history.append(gesture)
        
        # Draw UI
        cv2.rectangle(frame, (0, 0), (w, 80), (50, 50, 50), -1)
        
        if gesture:
            color = (0, 255, 0)
            cv2.putText(frame, f"Gesture Detected: {gesture.upper()}", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
            cv2.putText(frame, f"Confidence: HAND VISIBLE", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        else:
            cv2.putText(frame, "Gesture Detected: NONE", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, "No hand detected or gesture unrecognized", (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        
        # Show recent gestures
        bottom_y = h - 120
        cv2.rectangle(frame, (0, h - 120), (w, h), (50, 50, 50), -1)
        
        recent = gesture_history[-10:] if gesture_history else []
        text = "Recent: " + " → ".join(recent[-5:]) if recent else "Recent: (none)"
        cv2.putText(frame, text, (20, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.putText(frame, f"Frame: {frame_count} | Gestures detected: {len(set(gesture_history))}", 
                   (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, "Press 'q' to quit", (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Gesture Detection Debug", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total frames processed: {frame_count}")
    print(f"Unique gestures detected: {set(gesture_history)}")
    print(f"Gesture frequency:")
    for gesture in set(gesture_history):
        count = gesture_history.count(gesture)
        print(f"  {gesture}: {count} times")
    
    if not gesture_history:
        print("\n⚠️ NO GESTURES DETECTED")
        print("\nTroubleshooting:")
        print("1. Ensure your hand is clearly visible in the camera")
        print("2. Make sure the background is not too dark")
        print("3. Try moving closer to the camera")
        print("4. Try different lighting")
        print("\nIf still not working:")
        print("- The hand detection model may need adjustment")
        print("- Check MediaPipe hand landmarks are working")
    else:
        print("\n✓ Gesture detection working!")
        print("Try using these gestures with the main pipeline:")
        print("  1. Start with: open_hand")
        print("  2. Then show: thumbs_up")
    
    print("=" * 70)
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
