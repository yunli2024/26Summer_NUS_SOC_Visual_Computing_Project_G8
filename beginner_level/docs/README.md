# Part II Beginner Level: Face Landmarks

## 1. Goal

Part II opens the webcam, detects faces with a Haar cascade, and detects 68 facial landmarks with OpenCV's LBF facemark model.

The program draws:

- a rectangle around each detected face
- small circles on facial landmarks
- real-time FPS
- current preprocessing mode and status text

## 2. Folder Structure

```text
beginner_level/
├─ src/
├─ tests/
├─ docs/
└─ outputs/
```

`src` stores working code. `tests` stores safe checks that do not open the webcam. `docs` stores notes and experiment records. `outputs` stores screenshots, robustness results, and logs.

## 3. Python Files

- `src/config.py`: project paths, camera index, detection parameters, and CLAHE settings.
- `src/face_detector.py`: loads the Haar cascade and detects face rectangles.
- `src/landmark_detector.py`: loads the LBF model and detects 68 facial landmarks.
- `src/visualization.py`: draws face boxes, landmark points, FPS, and status text.
- `src/run_face_landmarks.py`: main webcam program.
- `tests/check_part2_setup.py`: checks imports and model loading without opening the webcam.

## 4. Run The Check Script

Use the project environment:

```powershell
conda activate vc_sws3026
python beginner_level\tests\check_part2_setup.py
```

This script should not open the camera.

## 5. Run The Main Program

Original grayscale mode:

```powershell
.\run_beginner_level.ps1
```

CLAHE preprocessing mode:

```powershell
.\run_beginner_level.ps1 --preprocess clahe
```

YuNet alternative detector:

```powershell
.\run_beginner_level.ps1 --detector yunet
```

Haar remains the course baseline. YuNet uses the versioned OpenCV Zoo ONNX
model and can be selected for the required alternative-detector comparison.
The HUD identifies the active detector so matched robustness runs can be
recorded without ambiguity.

## 6. Exit

Press `q` in the OpenCV window to exit. The program releases the camera and closes the window.

## 7. Haar Face Detection Flow

The webcam frame is converted from BGR to grayscale. The Haar cascade scans the grayscale image and returns face rectangles in `(x, y, w, h)` format.

## 8. LBF 68-Point Detection Flow

The LBF facemark model receives the grayscale frame and face rectangles. If fitting succeeds, it returns 68 landmark points for each face.

## 9. Toggle CLAHE

Use `--clahe` to enable CLAHE preprocessing. This can help in uneven lighting, but may also increase noise.

## 10. Robustness Testing

Use `docs/robustness_test.md` to record matched Haar and YuNet results under
different lighting, head poses, occlusions, distances, and multiple people.

## 11. Output Locations

- Screenshots: `beginner_level/outputs/screenshots/`
- Robustness results: `beginner_level/outputs/robustness_results/`
- Logs: `beginner_level/outputs/logs/`

## 12. Current Limitations

- Haar detection may fail with large head rotation or strong occlusion.
- LBF landmarks depend on successful face detection.
- Fast motion can cause jitter.
- Very dark or overexposed lighting may reduce accuracy.
- Multiple faces may reduce FPS.
