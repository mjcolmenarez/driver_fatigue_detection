#!/bin/bash
# Script to record a demo video of the Driver Fatigue Detection system

echo "======================================================================"
echo "DRIVER FATIGUE DETECTION - DEMO RECORDING GUIDE"
echo "======================================================================"
echo ""
echo "This will record a demo video showing:"
echo "  1. System starting inactive"
echo "  2. You performing the activation gestures (open hand → thumbs up)"
echo "  3. System becoming active (SYSTEM: ACTIVE in green)"
echo "  4. You showing signs of fatigue (close eyes, yawn, nod head)"
echo "  5. System detecting and displaying SLEEPY/AWAKE predictions"
echo ""
echo "======================================================================"
echo ""
echo "Starting demo recording..."
echo "Output will be saved to: demo_output.mp4"
echo ""
echo "CONTROLS during recording:"
echo "  'r' - Reset gesture sequence if needed"
echo "  'q' or ESC - Stop recording and save"
echo ""
echo "======================================================================"
echo ""

cd /Users/sofiaclaudiabonoan/Desktop/driver_fatigue_detection
.venv/bin/python main.py --camera 0 --output demo_output.mp4

echo ""
echo "======================================================================"
echo "Demo recording complete!"
echo "Video saved to: demo_output.mp4"
echo "======================================================================"
