# Real-Time Facial and Body Keypoint Analysis

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-1f6feb.svg" alt="License: AGPL-3.0"></a>
  <a href="https://github.com/yunli2024/NUS_SOC_Visual_Computing_Project_26Summer/actions/workflows/core-checks.yml"><img src="https://github.com/yunli2024/NUS_SOC_Visual_Computing_Project_26Summer/actions/workflows/core-checks.yml/badge.svg" alt="Core checks"></a>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/runtime-CPU--first-0A7F5A.svg" alt="CPU-first runtime">
  <img src="https://img.shields.io/badge/input-keypoints%20only-7B2CBF.svg" alt="Keypoint-only expression input">
  <img src="https://img.shields.io/badge/demos-4%20interactive-E76F51.svg" alt="Four interactive demos">
</p>

> **Project summary:** An end-to-end, CPU-first visual-computing suite that
> converts webcam and video keypoints into facial-expression effects,
> reaction-lag-aware dance scoring, and gesture-controlled gameplay.

**NUS School of Computing Summer Workshop 2026 · SWS3026 Visual Computing · Group 8**

The project goes beyond isolated model scripts: it delivers four local,
webcam-ready applications through one reusable face/body keypoint pipeline,
with measured accuracy, latency, robustness, and interaction behaviour.

**Core stack:** Python · OpenCV · NumPy · scikit-learn · Ultralytics YOLO ·
Tkinter · Pillow

**Review this project quickly:** [engineering case study](PORTFOLIO.md) ·
[measured evidence](#measured-results) · [run locally](#run) ·
[limitations](#known-limitations)

## Demo Gallery

All images below come from the implemented applications or their recorded
evaluation outputs.

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/readme/beginner-landmarks.png" alt="Real-time webcam face box and 68 facial landmarks" width="100%"><br>
      <strong>Facial landmark tracking.</strong> Haar/YuNet detection, 68-point
      LBF landmarks, preprocessing, ROI tracking, smoothing, and live FPS.
    </td>
    <td width="50%" valign="top">
      <img src="expert_level/artifacts/effects_preview.png" alt="Seven expression-driven visual effects" width="100%"><br>
      <strong>Keypoint-only expression effects.</strong> Seven-class
      RBF-SVM inference drives distinct overlays with confidence and temporal
      smoothing.
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/readme/dance-live-scoring.gif" alt="Live dual-panel pose tracking and dance scoring demo" width="100%"><br>
      <strong>Live pose tracking and dance scoring.</strong> Primary-dancer
      selection, spatial normalisation, motion comparison, mirror matching,
      and reaction-lag alignment.
    </td>
    <td width="50%" valign="top">
      <img src="docs/readme/mario-pose-control.gif" alt="Gesture-controlled platform game with live pose input" width="100%"><br>
      <strong>Gesture-controlled platform game.</strong> The same pose stream
      maps upper-body gestures to run, jump, and crouch with keyboard fallback.
    </td>
  </tr>
</table>

## Project Highlights

- **End-to-end delivery:** four local applications sharing reusable detection,
  feature, tracking, scoring, and rendering modules.
- **Constraint-driven ML:** expression classification uses only 174
  keypoint-derived features—never the original face pixels.
- **Measured performance:** **46.33% Macro-F1** and **18.08 ms**
  classifier-only latency on the 7,178-image FER-style test split.
- **Temporal interaction design:** dance scoring handles body scale, mirrored
  movement, motion, static-pose exploitation, and short user reaction delays.
- **Full-video validation:** a 2,680-frame CPU run achieved **91.16%
  primary-dancer detection**, **31.40 ms pose inference**, and **23.09 FPS**
  end-to-end offline processing at `imgsz=320`.
- **Engineering quality:** clear root launchers, camera-free preflight checks,
  modular code, persisted evidence, **52 passing local unit tests**, and a
  lightweight CI gate for platform-independent scoring/geometry logic.

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

## Project Poster

<details>
  <summary><strong>Open the full project poster</strong></summary>
  <p align="center">
    <a href="poster/SWS3026_08_final.pdf">
      <img src="poster/SWS3026_08_final.png" alt="Project poster for Real-Time Facial and Body Keypoint Analysis" width="900">
    </a>
  </p>
</details>

<p align="center">
  <a href="poster/SWS3026_08_final.pdf">View poster PDF</a> ·
  <a href="poster/SWS3026_08_final.pptx">Download editable poster</a> ·
  <a href="docs/course_materials/Visual_Computing_Project.pdf">Course brief</a>
</p>

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
git clone https://github.com/yunli2024/NUS_SOC_Visual_Computing_Project_26Summer.git
cd NUS_SOC_Visual_Computing_Project_26Summer

conda env create -f environment_setup\environment.yml
conda activate vc_sws3026
.\environment_setup\install_runtime.ps1
```

Use `opencv-contrib-python`, not the standard `opencv-python` wheel, because LBF requires `cv2.face`.
The installer deliberately installs Ultralytics with `--no-deps` after its
runtime dependencies so that pip does not add a conflicting OpenCV wheel.
Verify an existing environment without installing anything:

```powershell
.\environment_setup\install_runtime.ps1 -CheckOnly
```

If it reports `opencv-python` or `opencv-python-headless`, remove that
conflicting wheel and reinstall `opencv-contrib-python` before running the
demos.

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
.\check_merge.ps1
```

This runs 52 unit tests, generates the expression-effects preview, and validates
the Bonus and Mario model/data paths without opening a camera. If
`lbfmodel.yaml` has not been restored, only the LBF-specific Beginner check is
skipped with a warning.

To run the Beginner check explicitly after restoring `lbfmodel.yaml`:

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

`bonus_level/mario_demo/` is a frozen historical snapshot retained for
traceability. `bonus_level_mario/` is the only maintained Mario implementation
and the target used by the root launcher and CI.

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

## Open-Source Licence

The project is distributed under the
[GNU Affero General Public License v3.0](LICENSE). AGPL-3.0 was selected
instead of MIT because the Bonus and Mario applications use Ultralytics YOLO
software and pretrained weights that are released under AGPL-3.0.

Original Group 8 code and assets are copyright © 2026 Zhang Zonghao,
Wang Xiaorui, Li Yunzang, and Zhang Yunxiang. OpenCV models, course materials,
datasets, pretrained weights, and other third-party components retain their
original licences and copyright; see
[Third-Party Notices](THIRD_PARTY_NOTICES.md).

## Team

- Zhang Zonghao
- Wang Xiaorui
- Li Yunzang
- Zhang Yunxiang

Developed for the NUS School of Computing Summer Workshop SWS3026 Visual Computing group project.
