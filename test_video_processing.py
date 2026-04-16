#!/usr/bin/env python3
"""
Test the pipeline with synthetic video frames (no camera needed)
Simulates realistic face/gesture scenarios
"""

import numpy as np
import cv2
import sys
from main import FatigueDetectionPipeline
import time

def create_test_video(output_path="test_output.mp4", num_frames=100):
    """Generate a test video with synthetic frames"""
    print(f"Generating {num_frames} synthetic frames for testing...")
    
    w, h = 640, 480
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # Generate frames with gradient changes to simulate motion
    for i in range(num_frames):
        # Create a frame with varying background
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Add some variation to simulate motion
        intensity = int(50 + 100 * abs(np.sin(i / 10)))
        frame[:, :] = intensity
        
        # Add text showing frame number
        cv2.putText(frame, f"Frame {i+1}/{num_frames}", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.putText(frame, "(No face detection - synthetic test)", (50, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 1)
        
        writer.write(frame)
    
    writer.release()
    print(f"✓ Test video saved to {output_path}")
    return output_path

def main():
    """Test pipeline with synthetic video"""
    print("=" * 70)
    print("FATIGUE DETECTION PIPELINE - SYNTHETIC VIDEO TEST")
    print("=" * 70)
    
    # Generate test video
    test_video = create_test_video("test_synthetic.mp4", num_frames=60)
    
    print("\n2. Testing pipeline with video file...")
    print(f"   Opening: {test_video}")
    
    # Initialize pipeline
    try:
        pipeline = FatigueDetectionPipeline()
        print("   ✓ Pipeline initialized")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Open video
    cap = cv2.VideoCapture(test_video)
    if not cap.isOpened():
        print(f"   ✗ Failed to open video")
        return False
    
    print("   ✓ Video opened")
    
    # Process frames
    frame_count = 0
    print("\n3. Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        try:
            results = pipeline.process_frame(frame)
            
            if frame_count % 10 == 0:
                print(f"   Frame {frame_count}: ", end="")
                print(f"Gesture={results['gesture']}, Active={results['system_active']}, Pred={results['prediction']}")
        except Exception as e:
            print(f"   ✗ Error processing frame {frame_count}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    cap.release()
    
    print(f"\n   ✓ Processed {frame_count} frames successfully")
    
    print("\n" + "=" * 70)
    print("✓✓✓ VIDEO PROCESSING TEST PASSED!")
    print("=" * 70)
    print("\nNotes:")
    print("- Synthetic frames don't contain faces, so no predictions are made")
    print("- This validates that the pipeline can handle video I/O and frame processing")
    print("- For real testing with faces, use:")
    print("  1. Camera (requires camera permissions on macOS):")
    print("     python main.py --camera 0")
    print("  2. Real video file:")
    print("     python main.py --video <path_to_video>")
    print("\nTo grant camera permissions on macOS:")
    print("  System Preferences > Security & Privacy > Camera")
    print("  Make sure Terminal (or your Python app) is allowed")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
