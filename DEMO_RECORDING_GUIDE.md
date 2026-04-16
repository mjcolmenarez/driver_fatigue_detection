# Demo Recording Guide
**Driver Fatigue Detection System - Real-Time Demonstration**

---

## 📹 Quick Start

**Total Time:** ~10 minutes (5 min demo content + setup)

### One Command to Start Recording:
```bash
cd /Users/sofiaclaudiabonoan/Desktop/driver_fatigue_detection
.venv/bin/python main.py --camera 0 --output demo_output.mp4
```

Output will be saved to: **`demo_output.mp4`**

---

## 🎬 What to Demonstrate (In Order)

### Phase 1: System Startup (5 seconds) ⏱️
1. Start the app with the command above
2. Show yourself on camera
3. Display should show:
   - `Gesture: None`
   - `SYSTEM: INACTIVE` (red text)
   - Your face clearly visible in the center

**What's happening:** System detects no hand gestures, remains inactive

---

### Phase 2: Gesture Activation (15-20 seconds) ⏱️
1. **Show Open Hand Gesture**
   - Hold up your open hand to camera
   - All 5 fingers visible/extended
   - Palm facing camera
   - Keep steady for ~1 second
   - Display should show: `Step 1/2 done! Now show: thumbs_up`

2. **Show Thumbs Up Gesture**
   - Immediately show thumbs up
   - Only thumb extended, other fingers in fist
   - Keep steady for ~1 second  
   - Display should change to:
     - `Gesture: thumbs_up`
     - `SYSTEM: ACTIVE` (green text)

**What's happening:** System validates gesture sequence, becomes ACTIVE

---

### Phase 3: Fatigue Detection & Predictions (2-3 minutes) ⏱️

Once activated (green "SYSTEM: ACTIVE"), demonstrate the system detecting fatigue by:

#### **A. Show Alertness (Awake State)**
- Look at camera normally
- Keep eyes wide open
- Normal head position
- **Expected display:** `Prediction: AWAKE` (green)

#### **B. Simulate Fatigue Signs**
Show these signs one at a time (hold each for 3-5 seconds):

1. **Eye Closure (strongest indicator)**
   - Gently close eyes partway/fully
   - Hold closed for 3-5 seconds
   - **Expected:** `Prediction: SLEEPY` (red)

2. **Yawning**
   - Open mouth wide as if yawning
   - Hold for 2-3 seconds
   - **Expected:** `Prediction: SLEEPY`

3. **Head Nodding**
   - Move head forward/back slowly (simulates nodding/drowsiness)
   - Hold position for 2-3 seconds
   - **Expected:** Changes between AWAKE/SLEEPY based on eye state

4. **Repeat Pattern**
   - Cycle 2-3 times through fatigue signs
   - This shows the system is responsive and accurate

#### **C. Return to Alertness**
- Open eyes wide
- Look normal
- **Expected:** Back to `Prediction: AWAKE`

---

## 🎥 Recording Tips

### Camera Setup
- ✅ **Face must be clearly visible** (requirements state this!)
- ✅ **Good lighting** — Face should be well-lit
- ✅ **Steady camera/mount** — Avoid shaking
- ✅ **Centered position** — Face in middle of frame
- ✅ **Show full sequence** — From inactive → active → predictions

### What NOT to Do
- ❌ Don't exaggerate facial expressions (stay realistic)
- ❌ Don't cover face with hands
- ❌ Don't move around too much
- ❌ Don't press any keys during demo (might disrupt recording)

### Keyboard Controls During Recording
- `q` or `ESC` — Stop recording and save video
- `r` — Reset gesture sequence (if needed)
- `s` — Toggle recording on/off (not recommended during demo)

---

## 📊 Expected UI Display

### When System is INACTIVE
```
Top Section:
  Gesture: [None or detected gesture]
  SYSTEM: INACTIVE  ← Red text

Middle Section:
  [Your face from camera]

Bottom Section:
  (empty - no predictions yet)
```

### When System is ACTIVE
```
Top Section:
  Gesture: [detected gesture]
  SYSTEM: ACTIVE  ← Green text

Middle Section:
  [Your face from camera]

Bottom Section:
  Prediction: AWAKE or SLEEPY  ← Changes based on eyes/expression
  Confidence: 0.XX
```

---

## ✅ Demo Checklist

Before recording, ensure:
- [ ] Camera is working and has good lighting
- [ ] Your face is centered and clearly visible
- [ ] You're in a quiet location (audio not critical but good to have)
- [ ] You've tested the gestures work (run `debug_gestures.py` if unsure)
- [ ] You know what key to press to stop (`q` or `ESC`)
- [ ] You have 10 minutes of uninterrupted time

Before publishing the video, check:
- [ ] Video shows full demo scenario (all phases)
- [ ] Your face is visible throughout
- [ ] System transitions from INACTIVE → ACTIVE clearly
- [ ] Multiple fatigue predictions shown (AWAKE/SLEEPY changes)
- [ ] File `demo_output.mp4` was created successfully

---

## 🛠️ Troubleshooting

### "Camera failed to initialize" error
1. Check System Preferences → Security & Privacy → Camera
2. Ensure Terminal (or IDE) has camera permission
3. Close camera app if open
4. Try again

### "No gestures detected"
1. Run `python debug_gestures.py` to test gesture recognition
2. Make sure hand is in frame and well-lit
3. Try different hand positions/angles
4. Ensure gestures are clear (not blurry)

### "Predictions don't change"
1. Wait for system to build up temporal features (~0.5 sec after activation)
2. Make facial expression changes more pronounced
3. Close eyes more completely (EAR is most important signal)
4. Check that face is properly detected in video

### "Video file not created"
1. Check if `demo_output.mp4` exists in project directory
2. Verify write permissions in the directory
3. Try specifying a different output path
4. Ensure you pressed `q` to properly close video writer

---

## 📹 Example Timing

```
0:00 - Start app, show system INACTIVE
0:10 - Show open hand gesture
0:15 - Show thumbs up gesture  
0:20 - System becomes ACTIVE
0:25 - Show alert state (AWAKE prediction)
0:35 - Close eyes (SLEEPY prediction)
0:45 - Show awake state (AWAKE prediction)
0:55 - Yawn (SLEEPY prediction)
1:05 - Show awake state (AWAKE prediction)
1:15 - Nod head slowly (SLEEPY prediction)
1:30 - Return to awake
1:40 - Final state check
1:50 - Press 'q' to stop recording
```

Total: ~3-4 minutes of actual content

---

## 🎁 What the Instructor Will See

The demo video should show:

✅ **Student clearly visible** — Check  
✅ **System initially inactive** — Check  
✅ **Correct gesture sequence executed** — Check  
✅ **System activation (UI change)** — Check  
✅ **Live fatigue detection working** — Check  
✅ **Predictions correct** — Drowsy signs → SLEEPY, Alert → AWAKE  
✅ **Real-time performance** — System responds instantly  
✅ **Full end-to-end pipeline** — All components working together  

---

## 💡 Pro Tips

1. **Practice gestures first** — Run `debug_gestures.py` to get comfortable
2. **Do a test run** — Record 30 seconds to check lighting/positioning
3. **Extra footage is fine** — More examples of predictions = stronger demo
4. **Narration optional** — Audio is not required, visual demo is sufficient
5. **Reuse existing faces** — This is Sofia's data, system is trained on her patterns

---

## 📝 After Recording

1. **Save the video file** — `demo_output.mp4` should be in project root
2. **Verify file size** — Should be several MB (video data)
3. **Test playback** — Open in QuickTime or VLC to confirm
4. **Commit to Git:**
   ```bash
   cd /Users/sofiaclaudiabonoan/Desktop/driver_fatigue_detection
   git add demo_output.mp4
   git commit -m "demo: add recorded demonstration video

   - Shows system initialization (INACTIVE state)
   - Shows gesture activation sequence (open_hand → thumbs_up)
   - Shows system becoming ACTIVE (green UI)
   - Shows fatigue detection in action
   - Shows AWAKE vs SLEEPY predictions
   - Real-time performance on live camera feed"
   git push origin main
   ```

---

## 🎬 Recording Format

- **Codec:** H.264 (MP4)
- **Resolution:** Matches camera (typically 640x480)
- **FPS:** 30 FPS (real-time)
- **Audio:** Not captured (not needed)

---

## ❓ Questions?

If something doesn't work:
1. Check that all dependencies installed: `pip list | grep -E "opencv|mediapipe|sklearn"`
2. Test individual components: `python debug_gestures.py`
3. Verify models loaded: Run `python test_pipeline.py`
4. Check camera permissions in System Preferences

---

**You're ready to record! 🎬**

Good luck with the demo! The system is fully working and tested — just show what it can do.

