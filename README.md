# SWS3026 Visual Computing Group 8 Project

This repository contains the Group 8 implementation for the NUS SOC SWS3026 Visual Computing project, **Real-time Video Analysis and Rendering**.

The project focuses on local real-time webcam demos. It does not depend on Google Colab or notebooks because the application must access the laptop camera.

## Project Structure

| Path | Description |
|---|---|
| `Visual_Computing_Project.pdf` | Original project brief |
| `beginner/` | Starter webcam and facial landmark assets |
| `expert/` | FER-style expression dataset zip |
| `bonus/` | Starter dance/pose GUI assets |
| `demo_0710/` | Main implementation, reports, trained expression model, and run manual |

## Implemented Levels

### Beginner

- Opens the local webcam with OpenCV.
- Detects faces using Haar Cascade.
- Fits 68-point facial landmarks using OpenCV LBF.
- Draws face boxes, landmarks, FPS, and timing information in real time.
- Includes robustness notes for lighting, pose, and occlusion cases.

Run:

```powershell
cd demo_0710
python beginner_demo.py --mirror
```

### Expert

- Reads the FER-style dataset from `expert/facial_expression_dataset.zip`.
- Extracts facial landmarks for every train/test image.
- Builds landmark-derived features instead of using full face images.
- Trains an RBF SVM expression classifier.
- Reports accuracy, macro F1, confusion matrix, and failure cases.
- Runs a real-time webcam expression classifier with visual effects.

Current full train/test evaluation:

| Metric | Value |
|---|---:|
| Train samples | 28,709 |
| Test samples | 7,178 |
| Landmark extraction failures | 0 |
| Accuracy | 0.4536 |
| Macro F1 | 0.4198 |
| Average classifier prediction time | 3.2867 ms |

Run:

```powershell
cd demo_0710
python expert_demo.py --mirror
```

Benchmark:

```powershell
python expert_demo.py --mirror --benchmark-frames 30
```

## Environment

Use Conda with Python 3.10:

```powershell
conda create -n sws3026-beginner python=3.10 -y
conda activate sws3026-beginner
cd demo_0710
python -m pip install -r requirements.txt
```

`opencv-contrib-python` is required because the LBF landmark detector uses `cv2.face.createFacemarkLBF()`.

## Notes

- The detailed run manual is in `demo_0710/manu.md`.
- The Expert report is in `demo_0710/expert_report.md`.
- The large optional TikTok raw video dataset is not tracked in Git because several files exceed GitHub's normal file-size limits.
