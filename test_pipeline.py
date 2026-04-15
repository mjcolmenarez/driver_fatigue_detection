#!/usr/bin/env python3
"""
Test script to validate the FatigueDetectionPipeline logic
Creates synthetic frame data to test the pipeline without requiring video files
"""

import numpy as np
import cv2
from main import FatigueDetectionPipeline

def create_synthetic_frame(w=640, h=480):
    """Create a synthetic frame (black background)"""
    return np.zeros((h, w, 3), dtype=np.uint8)

def test_pipeline_logic():
    """Test pipeline initialization and frame processing"""
    print("=" * 70)
    print("FATIGUE DETECTION PIPELINE - LOGIC TEST")
    print("=" * 70)
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    try:
        pipeline = FatigueDetectionPipeline()
        print("   ✓ Pipeline initialized")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test synthetic frame processing
    print("\n2. Testing frame processing (no face detection)...")
    frame = create_synthetic_frame()
    try:
        results = pipeline.process_frame(frame)
        print("   ✓ Frame processed successfully")
        print(f"     - Frame count: {results['frame_count']}")
        print(f"     - Gesture: {results['gesture']}")
        print(f"     - System active: {results['system_active']}")
        print(f"     - Prediction: {results['prediction']}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test gesture state tracking
    print("\n3. Testing gesture validator state...")
    try:
        # Process multiple frames
        for i in range(5):
            frame = create_synthetic_frame()
            results = pipeline.process_frame(frame)
        
        print(f"   ✓ Processed 5 frames successfully")
        print(f"     - Gesture status: {results['gesture_status']}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test reset functionality
    print("\n4. Testing reset functionality...")
    try:
        pipeline.reset()
        results = pipeline.process_frame(create_synthetic_frame())
        print("   ✓ Reset successful")
        print(f"     - System active: {results['system_active']}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    print("\n" + "=" * 70)
    print("✓✓✓ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Download test videos from Google Drive to data/ folder")
    print("2. Run: python main.py --video <path_to_video>")
    print("3. Or use webcam: python main.py --camera 0")
    print("\nControls:")
    print("  'q' or ESC - Quit")
    print("  'r'        - Reset gesture sequence")
    print("  's'        - Start/stop recording")
    print("=" * 70)
    
    return True

if __name__ == "__main__":
    import sys
    success = test_pipeline_logic()
    sys.exit(0 if success else 1)
