# Real-Time Facial and Body Keypoint Analysis

> A CPU-first visual-computing system that turns webcam and video keypoints into expression effects, reaction-aware dance scoring, and gesture-controlled gameplay.

NUS School of Computing Summer Workshop 2026 · SWS3026 Visual Computing · Group 8

<p align="center">
  <a href="poster/SWS3026_08_final.pdf">
    <img src="poster/SWS3026_08_final.png" alt="Project poster for Real-Time Facial and Body Keypoint Analysis" width="100%">
  </a>
</p>

<p align="center">
  <a href="poster/SWS3026_08_final.pdf">View poster PDF</a> ·
  <a href="poster/SWS3026_08_final.pptx">Download editable poster</a> ·
  <a href="docs/course_materials/Visual_Computing_Project.pdf">Course brief</a>
</p>

## Project Highlights

- Built four local, camera-ready demos around sparse facial and body keypoints rather than raw-frame classification.
- Engineered a keypoint-only, seven-class expression model from 136 normalised LBF coordinates and 38 geometric descriptors.
- Achieved **46.33% Macro-F1** and **18.08 ms classifier-only latency** on the 7,178-image FER-style test split.
- Designed scale-, mirror-, motion-, and reaction-lag-aware dance scoring with `PERFECT / GREAT / GOOD / MISS`, score, and combo feedback.
- Validated YOLOv8n Pose on a complete **2,680-frame CPU run**: **91.16% primary-dancer detection**, **31.40 ms pose inference**, and **23.09 FPS end-to-end offline processing** at `imgsz=320`.
- Reused the pose stream for a camera-controlled platform game with temporal gesture confirmation and keyboard fallback.

## System Overview

<p align="center">
  <img src="docs/project-framework.png" alt="System architecture from video input through keypoints, analysis, and interactive output" width="100%">
</p>

The project follows one reusable flow:

1. Capture a webcam frame or reference-video frame.
2. Detect a face or person and extract sparse keypoints.
3. Normalise geometry to reduce translation, scale, rotation, and body-size bias.
4. Classify facial expression or compare pose and motion over time.
5. Render immediate visual, scoring, or gameplay feedback.

## Interactive Demos

| Module | What it demonstrates | Entry point |
|---|---|---|
| Facial landmarks | Haar/YuNet face detection, 68-point LBF landmarks, CLAHE, ROI tracking, smoothing, and FPS | `.\run_beginner_level.ps1` |
| Expression effects | Keypoint-only seven-class inference, probability smoothing, labels, confidence, and seven animated effects | `.\run_expert_level.ps1` |
| Just Dance-style scoring | Dual-panel reference/webcam pose tracking, spatial normalisation, mirror matching, reaction-lag search, score, and combo | `.\run_bonus_level.ps1` |
| Gesture-controlled game | Upper-body pose gestures mapped to run, jump, and crouch in a complete platform level | `.\run_mario.ps1` |

All demos run locally because webcam access and OpenCV/Tkinter windows are core to the project. Google Colab and notebook-only workflows are intentionally out of scope.

## Measured Results

### Keypoint-Only Expression Recognition

The classifier never receives the original face pixels. FER-style images are used only to fit 68 LBF landmarks; the model input is the resulting keypoint geometry.

| Metric | Final result |
|---|---:|
| Training samples | 28,709 |
| Test samples | 7,178 |
| Input representation | 136 normalised coordinates + 38 geometry features |
| Classifier | Class-balanced RBF-SVM |
| Test accuracy | 47.31% |
| Test Macro-F1 | **46.33%** |
| Test weighted F1 | 47.35% |
| Single-sample classifier latency | **18.08 ms** |
| `< 30 ms` classifier target | Passed |

Model comparison and parameter selection were restricted to a stratified validation subset of the official training split; the official test split was used for the reported final evaluation. PCA was retained as a controlled comparison and reduced latency at a substantial Macro-F1 cost. A full Stratified K-fold confirmation remains a documented next step rather than a completed claim.

Evidence:

- [Metrics](expert_level/artifacts_svm_geometry/metrics.json)
- [Confusion matrix](expert_level/artifacts_svm_geometry/confusion_matrix.png)
- [Failure cases](expert_level/artifacts_svm_geometry/failure_cases.png)
- [Detailed training report](expert_level/LEVEL2_TRAINING_DETAILED_REPORT.md)

### Body Pose Benchmark

The final reference-video benchmark processes every frame of the included 2,680-frame clip on CPU.

| Metric | Final result |
|---|---:|
| Frames processed | 2,680 |
| Primary-dancer detection | **91.16%** |
| Visible keypoints | 16.70 / 17 average |
| Average pose confidence | 0.925 |
| YOLOv8n pose inference | **31.40 ms / frame** |
| End-to-end offline throughput | **23.09 FPS** |
| Input size | 320 |

The difference between inference latency and total throughput reflects decoding, tracking, smoothing, drawing, and result collection around the model call.

Evidence:

- [Full-run summary](bonus_level/task2_results/summary.json)
- [Reference contact sheet](bonus_level/task2_results/dance_example_1/contact_sheet.jpg)
- [Scoring design report](bonus_level/LEVEL3_DANCE_SCORING_DETAILED_REPORT.md)

## Technical Design

### Face Pipeline

- Haar cascade for the required baseline, with YuNet available as an alternative detector.
- OpenCV Facemark LBF for 68 facial landmarks.
- CLAHE/gamma preprocessing, periodic full-frame re-detection, ROI tracking, and exponential smoothing.
- Models are loaded once before the frame loop; each app releases the camera and closes windows on exit.

### Expression Pipeline

- Eye-centred translation, rotation, and scale normalisation.
- Expression-focused distances, aspect ratios, angles, curvatures, and symmetry descriptors.
- Class-balanced RBF-SVM selected with Macro-F1 as the primary metric.
- Per-face landmark and probability smoothing for steadier real-time output.

### Dance Scoring

- Primary-dancer selection combines body size, centre proximity, confidence, and temporal continuity.
- Torso-centred spatial normalisation reduces position and body-size bias.
- Similarity combines pose geometry and motion:

  `active_score = 0.55 × pose_similarity + 0.45 × motion_similarity`

- A short temporal window searches for user reaction lag.
- Mirrored left/right joints are evaluated when mirror matching is enabled.
- `HOLD` / `MOVE!` hysteresis prevents static poses from earning repeated points.

## Getting Started

### Requirements

- Python 3.11
- Windows PowerShell for the provided root launchers
- A laptop webcam for interactive demos
- CPU execution is supported; CUDA is optional

### Install

```powershell
git clone https://github.com/yunli2024/26Summer_NUS_SOC_Visual_Computing_Project_G8.git
cd 26Summer_NUS_SOC_Visual_Computing_Project_G8

conda env create -f environment_setup\environment.yml
conda activate vc_sws3026
python -m pip install -r environment_setup\requirements.txt
```

Use `opencv-contrib-python`, not the standard `opencv-python` wheel, because LBF requires `cv2.face`.

### Restore Course-Provided Assets

Large course assets are intentionally excluded from Git. Place them at:

```text
resources/face_models/lbfmodel.yaml
resources/expression_data/facial_expression_dataset.zip
```

- `lbfmodel.yaml` is required by the facial-landmark and expression webcam demos.
- The FER-style archive is needed only to reproduce feature extraction and training; the final expression classifier is already versioned.
- Haar, YuNet, YOLOv8n Pose, reference videos, and the final classifier are included.

See [resources/README.md](resources/README.md) for the complete asset policy.

## Run

Run all commands from the repository root after activating `vc_sws3026`.

### 1. Facial Landmarks

```powershell
.\run_beginner_level.ps1
.\run_beginner_level.ps1 --preprocess clahe
```

Press `Q` to quit. Runtime keys `1`-`4` switch preprocessing modes; `V` toggles enhanced display preprocessing.

### 2. Expression Effects

```powershell
.\run_expert_level.ps1
```

Controls:

- `E`: effects
- `L`: landmarks
- `S`: temporal smoothing
- `C`: CLAHE
- `Q` / `Esc`: quit

Generate a camera-free effects preview:

```powershell
.\run_expert_level.ps1 --preview expert_level\artifacts\effects_preview.png
```

### 3. Just Dance-Style App

```powershell
.\run_bonus_level.ps1 --check
.\run_bonus_level.ps1
```

If CPU inference is slow:

```powershell
.\run_bonus_level.ps1 --image-size 320
```

### 4. Gesture-Controlled Platform Game

```powershell
.\run_mario.ps1 --check
.\run_mario.ps1
```

Use `--camera 1` for a non-default webcam or `--image-size 256` to reduce CPU load.

## Verification

Camera-free checks:

```powershell
python expert_level\test_expression_features.py
python bonus_level\test_dance_scoring.py
python -m unittest discover -s bonus_level_mario -p "test_*.py" -v
.\run_bonus_level.ps1 --check
.\run_mario.ps1 --check
```

After restoring `lbfmodel.yaml`:

```powershell
python beginner_level\tests\check_part2_setup.py
```

Webcam and GUI behaviour must still be validated on the target laptop: confirm camera index, lighting, working distance, FPS, controls, and clean exit.

## Repository Structure

```text
beginner_level/       Facial detection, LBF landmarks, preprocessing, and robustness checks
expert_level/         Keypoint features, model training/evaluation, real-time effects, and evidence
bonus_level/          Dance analysis, two-panel GUI, temporal pose scoring, and benchmark results
bonus_level_mario/    Pose-controlled platform-game extension and tests
resources/            Versioned detectors, pose model, videos, and local asset instructions
environment_setup/    Conda and pip environment definitions
docs/                 Course brief, reports, framework diagram, and presentation material
poster/               Final editable poster, PDF, and README preview
```

## Known Limitations

- Haar + LBF remains sensitive to large head rotation, heavy occlusion, motion blur, and extreme lighting.
- Landmark geometry omits appearance cues such as wrinkles and skin texture, limiting recognition of subtle or ambiguous expressions.
- The expression latency above measures classifier prediction, not the complete face-detection and landmark pipeline.
- CPU-only pose throughput varies by device and can remain below the reference-video frame rate.
- YOLOv8 Pose exposes body joints, not finger landmarks.
- Live results depend on camera quality, background complexity, subject distance, and hardware.

## Documentation

- [Project documentation index](docs/README.md)
- [Beginner implementation guide](beginner_level/docs/README.md)
- [Beginner robustness observations](beginner_level/docs/robustness_test.md)
- [Expert Task 1 report](expert_level/TASK1_REPORT.md)
- [Expert Task 2 report](expert_level/TASK2_REPORT.md)
- [Bonus Task 1 report](bonus_level/TASK1_REPORT.md)
- [Bonus Task 2 report](bonus_level/TASK2_REPORT.md)
- [Mario extension guide](bonus_level_mario/README.md)

## Team

- Zhang Zonghao
- Wang Xiaorui
- Li Yunzang
- Zhang Yunxiang

Developed for the NUS School of Computing Summer Workshop SWS3026 Visual Computing group project.
