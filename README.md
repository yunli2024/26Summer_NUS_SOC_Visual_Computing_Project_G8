# Real-Time Facial and Body Keypoint Analysis

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0-1f6feb.svg" alt="License: AGPL-3.0"></a>
  <a href="https://github.com/yunli2024/NUS_SOC_Visual_Computing_Project_26Summer/actions/workflows/core-checks.yml"><img src="https://github.com/yunli2024/NUS_SOC_Visual_Computing_Project_26Summer/actions/workflows/core-checks.yml/badge.svg" alt="Core checks"></a>
  <img src="https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/runtime-CPU--first-0A7F5A.svg" alt="CPU-first runtime">
  <img src="https://img.shields.io/badge/input-keypoints%20only-7B2CBF.svg" alt="Keypoint-only expression input">
</p>

<p align="center">
  <strong>NUS School of Computing Summer Workshop 2026 | SWS3026 Visual Computing | Group 8</strong>
</p>

<p align="center">
  <img src="docs/readme/cover.png" alt="Group 8 presenting the real-time keypoint analysis project at the NUS SOC showcase" width="560">
</p>

An end-to-end, CPU-first visual-computing suite that turns webcam and video
keypoints into facial-expression effects, reaction-lag-aware dance scoring,
and gesture-controlled gameplay.

**Stack:** Python | OpenCV | NumPy | scikit-learn | Ultralytics YOLO |
Tkinter | Pillow

**Quick links:** [Run the demos](#run) | [Measured results](#measured-results) |
[Project poster](#project-poster) | [Engineering case study](PORTFOLIO.md)

## Demo Gallery

<table>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/readme/beginner-landmarks.png" alt="Real-time face box and 68 facial landmarks" width="100%"><br>
      <strong>Facial landmarks.</strong> Haar/YuNet detection with 68-point
      LBF tracking and live FPS.
    </td>
    <td width="50%" valign="top">
      <img src="expert_level/artifacts/effects_preview.png" alt="Seven expression-driven visual effects" width="100%"><br>
      <strong>Expression effects.</strong> Keypoint-only classification drives
      seven real-time overlays.
    </td>
  </tr>
  <tr>
    <td width="50%" valign="top">
      <img src="docs/readme/dance-live-scoring.gif" alt="Dual-panel pose tracking and dance scoring" width="100%"><br>
      <strong>Dance scoring.</strong> Mirrored pose matching with spatial and
      reaction-lag alignment.
    </td>
    <td width="50%" valign="top">
      <img src="docs/readme/mario-pose-control.gif" alt="Gesture-controlled platform game" width="100%"><br>
      <strong>Pose-controlled game.</strong> Upper-body gestures map to run,
      jump, and crouch.
    </td>
  </tr>
</table>

## Demo Day Engagement

The project was built to be played, not just watched. At the NUS SOC showcase,
visitors tried the gesture-controlled game, explored pose interactions, and
discussed the real-time vision pipeline with the team.

<p align="center">
  <img src="docs/readme/demo-day/demo-day-engagement.jpg" alt="Visitors trying gesture-controlled and pose-based demos during the NUS SOC showcase" width="100%">
</p>

<p align="center">
  <em>Hands-on interaction around the booth: gesture control, live pose input,
  and camera-driven gameplay.</em>
</p>

<details>
  <summary><strong>View all 14 showcase photos</strong></summary>
  <br>
  <table>
    <tr>
      <td width="50%" align="center" valign="top">
        <img src="docs/readme/demo-day/01-game-display.jpg" alt="Camera-driven platform game on the showcase display" width="100%"><br>
        <sub>Showcase-ready game</sub>
      </td>
      <td width="50%" align="center" valign="top">
        <img src="docs/readme/demo-day/02-project-walkthrough.jpg" alt="Team member presenting the project" width="100%"><br>
        <sub>Project walkthrough</sub>
      </td>
    </tr>
    <tr>
      <td width="50%" align="center" valign="top">
        <img src="docs/readme/demo-day/03-gesture-interaction.jpg" alt="Visitor trying gesture control" width="100%"><br>
        <sub>Gesture interaction</sub>
      </td>
      <td width="50%" align="center" valign="top">
        <img src="docs/readme/demo-day/04-poster-demo.jpg" alt="Poster and live application presentation" width="100%"><br>
        <sub>Poster and live demo</sub>
      </td>
    </tr>
    <tr>
      <td width="50%" align="center" valign="top">
        <img src="docs/readme/demo-day/05-pose-control-test.jpg" alt="Participant testing pose control" width="100%"><br>
        <sub>Pose-control test</sub>
      </td>
      <td width="50%" align="center" valign="top">
        <img src="docs/readme/demo-day/06-pose-interaction.jpg" alt="Visitors exploring pose interaction" width="100%"><br>
        <sub>Trying it together</sub>
      </td>
    </tr>
    <tr>
      <td width="50%" align="center" valign="top">
        <img src="docs/readme/demo-day/07-live-gesture-demo.jpg" alt="Live gesture-control demonstration" width="100%"><br>
        <sub>Live gesture demo</sub>
      </td>
      <td width="50%" align="center" valign="top">
        <img src="docs/readme/demo-day/08-game-in-action.jpg" alt="Gesture-controlled game with live pose input" width="100%"><br>
        <sub>Game in action</sub>
      </td>
    </tr>
    <tr>
      <td width="50%" align="center" valign="top">
        <img src="docs/readme/demo-day/09-team-showcase.jpg" alt="Team member beside the project poster and demo" width="100%"><br>
        <sub>Team showcase</sub>
      </td>
      <td width="50%" align="center" valign="top">
        <img src="docs/readme/demo-day/10-booth-visitors.jpg" alt="Visitors around the interactive booths" width="100%"><br>
        <sub>Active exhibition floor</sub>
      </td>
    </tr>
    <tr>
      <td width="50%" align="center" valign="top">
        <img src="docs/readme/demo-day/11-level-complete.jpg" alt="Completed level and live pose-control view" width="100%"><br>
        <sub>Level complete</sub>
      </td>
      <td width="50%" align="center" valign="top">
        <img src="docs/readme/demo-day/12-hands-on-session.jpg" alt="Visitor using the camera-driven game" width="100%"><br>
        <sub>Hands-on session</sub>
      </td>
    </tr>
    <tr>
      <td width="50%" align="center" valign="top">
        <img src="docs/readme/demo-day/13-full-body-gesture.jpg" alt="Participant using a full-body gesture" width="100%"><br>
        <sub>Full-body gesture</sub>
      </td>
      <td width="50%" align="center" valign="top">
        <img src="docs/readme/demo-day/14-project-discussion.jpg" alt="Project discussion beside the running game" width="100%"><br>
        <sub>Project discussion</sub>
      </td>
    </tr>
  </table>
</details>

## Project Highlights

- Four local, webcam-ready applications built on reusable face and body
  keypoint pipelines.
- Expression recognition uses only 174 keypoint-derived features, never the
  original face pixels.
- Dance scoring handles body scale, mirrored motion, static poses, and short
  user reaction delays.
- CPU-first implementation with persisted metrics, camera-free checks, and 47
  platform-independent unit tests.

## System Overview

<p align="center">
  <img src="docs/project-framework.png" alt="System architecture from video input through keypoints to interactive output" width="100%">
</p>

Webcam or video frames are converted to sparse keypoints, normalised for
geometry, classified or compared over time, and rendered as immediate visual,
scoring, or gameplay feedback.

## Project Poster

<details open>
  <summary><strong>Project poster</strong></summary>
  <p align="center">
    <a href="poster/SWS3026_08_final.pdf">
      <img src="poster/SWS3026_08_final.png" alt="Project poster for Real-Time Facial and Body Keypoint Analysis" width="900">
    </a>
  </p>
</details>

<p align="center">
  <a href="poster/SWS3026_08_final.pdf">View PDF</a> |
  <a href="poster/SWS3026_08_final.pptx">Download editable poster</a> |
  <a href="docs/course_materials/Visual_Computing_Project.pdf">Course brief</a>
</p>

## Measured Results

| Pipeline | Key results |
|---|---|
| Keypoint-only expression recognition | **46.33% Macro-F1**, 47.31% accuracy, **18.08 ms** classifier latency on 7,178 test images |
| Full-video pose benchmark | **91.16%** primary-dancer detection, **31.40 ms** pose inference, **23.09 FPS** over 2,680 CPU-processed frames |

Expression model selection used a stratified validation split; the official
test split was reserved for final evaluation. Supporting evidence:
[expression metrics](expert_level/artifacts_svm_geometry/metrics.json),
[confusion matrix](expert_level/artifacts_svm_geometry/confusion_matrix.png),
[failure cases](expert_level/artifacts_svm_geometry/failure_cases.png), and
[pose benchmark](bonus_level/task2_results/summary.json).

## Technical Design

- **Face:** Haar baseline or YuNet detection, OpenCV LBF 68-point landmarks,
  preprocessing, ROI tracking, and temporal smoothing.
- **Expression:** eye-centred geometric normalisation and a class-balanced
  RBF-SVM selected by Macro-F1 and real-time latency.
- **Pose:** primary-dancer tracking, torso-centred scale normalisation,
  mirrored pose/motion similarity, and reaction-lag search.

## Interactive Demos

| Demo | Entry point |
|---|---|
| Facial landmarks | `.\run_beginner_level.ps1` |
| Expression effects | `.\run_expert_level.ps1` |
| Just Dance-style scoring | `.\run_bonus_level.ps1` |
| Gesture-controlled platform game | `.\run_mario.ps1` |

All demos run locally because webcam access and OpenCV/Tkinter windows are core
to the project.

## Getting Started

**Requirements:** Python 3.11, Windows PowerShell, and a webcam for interactive
demos. CPU execution is supported; CUDA is optional.

```powershell
git clone https://github.com/yunli2024/NUS_SOC_Visual_Computing_Project_26Summer.git
cd NUS_SOC_Visual_Computing_Project_26Summer

conda env create -f environment_setup\environment.yml
conda activate vc_sws3026
.\environment_setup\install_runtime.ps1
```

Use `opencv-contrib-python`, not `opencv-python`, because LBF requires
`cv2.face`. The course-provided `lbfmodel.yaml` must be restored to
`resources/face_models/lbfmodel.yaml`; the FER archive is only needed to
reproduce training. See [resources/README.md](resources/README.md).

## Run

Run from the repository root after activating `vc_sws3026`:

```powershell
.\run_beginner_level.ps1
.\run_expert_level.ps1
.\run_bonus_level.ps1
.\run_mario.ps1
```

Use `Q` or `Esc` to leave the OpenCV demos. For slower CPUs, reduce pose input
with `.\run_bonus_level.ps1 --image-size 320`.

## Repository Structure

```text
beginner_level/       Face detection, LBF landmarks, and robustness work
expert_level/         Keypoint features, expression model, effects, and evidence
bonus_level/          Pose analysis, dual-panel dance app, scoring, and benchmark
bonus_level_mario/    Pose-controlled platform-game extension
resources/            Detectors, models, videos, and asset instructions
environment_setup/    Conda and pip environment definitions
docs/                 Course brief, reports, diagrams, and showcase assets
poster/               Final poster in PNG, PDF, and editable PPTX formats
```

## Documentation

[Documentation index](docs/README.md) |
[Beginner guide](beginner_level/docs/README.md) |
[Expression report](expert_level/TASK1_REPORT.md) |
[Effects report](expert_level/TASK2_REPORT.md) |
[Dance analysis](bonus_level/TASK1_REPORT.md) |
[Dance scoring](bonus_level/TASK2_REPORT.md) |
[Mario extension](bonus_level_mario/README.md)

## Licence and Team

Released under the [GNU Affero General Public License v3.0](LICENSE). See
[Third-Party Notices](THIRD_PARTY_NOTICES.md) for model, dataset, and dependency
licences.

**Group 8:** Zhang Zonghao, Wang Xiaorui, Li Yunzang, and Zhang Yunxiang.

Developed for the NUS School of Computing Summer Workshop SWS3026 Visual
Computing group project.
