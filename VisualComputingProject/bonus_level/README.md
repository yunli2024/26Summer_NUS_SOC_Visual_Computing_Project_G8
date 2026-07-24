# Bonus Level: Pose Dance Scoring

This folder contains the final pose-detection and dance-scoring application.
The earlier single-file prototype and its duplicate model have been removed.

## Features

- YOLOv8 Pose detection with 17 COCO keypoints
- Main-dancer selection and temporal tracking
- Confidence filtering and pose smoothing
- Torso-centered, multi-cue body-scale normalization
- Mirrored webcam handling and left/right keypoint correction
- One-way 0-0.8 s reaction-delay search (current player vs recent reference)
- 0.4 s motion-window comparison and static-player anti-cheat factor
- Position (35%) + joint angle (65%) pose score
- Final pose (55%) + motion (45%) score with visibility coverage penalty
- Debounced `Hold`, warm-up `Sync`, and `Move!` feedback states
- Realtime feedback, combo, total score, and dance summary
- Average, median, and P90 matched-delay statistics

The merged baseline also retains:

- Liyunzang and Zhangyx Task 1/2 reports under `docs/branch_evidence/`;
- Zhangyx's pose-controlled Mario extension in `../bonus_level_mario/`;
- Sherry's 3D three-lane runner in `../bonus_level_action_game/`.

## Directory layout

```text
bonus_level/
|-- data/input/reference_videos/tiktok/  # 26 retained TikTok references
|-- outputs/runtime/                     # runtime output location
|-- src/                                 # final modular implementation
|-- main.py
`-- README.md
```

The application uses these shared project resources:

```text
VisualComputingProject/resources/pose_models/yolov8n-pose.pt
VisualComputingProject/resources/videos/dance_example_1.mp4
```

The TikTok reference directory contains one folder per dance sequence. Each
folder retains its video plus the available dance-name and source-link metadata.

## Run

```powershell
.\run_bonus_level.ps1
```

If PowerShell blocks scripts on your machine, run this once in the project root:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_bonus_level.ps1
```

Use the left panel to play a reference video and the right panel to start the
webcam. Start both streams to receive aligned pose scores and feedback.

## Grade construction

1. Normalize both 17-point COCO skeletons; score the 12 body joints.
2. Compute weighted joint-position similarity and eight joint-angle similarities.
3. Apply `0.60 + 0.40 * coverage` so partial detections remain usable but do not
   receive full confidence.
4. Once 0.4 s history exists, combine pose and motion as `0.55 / 0.45`.
5. If the reference is moving while the player is nearly static, apply an
   anti-static multiplier from 0.45 to 1.0.
6. Map similarity to `Perfect >= 85`, `Super >= 70`, `Good >= 55`, otherwise
   `Miss`. Genuine holds and motion warm-up frames are displayed but not counted.

The matcher only searches reference frames at or before the current player
timestamp. A reported `+0.35 s` therefore means the player's best match was
0.35 seconds behind the reference, not that the system used a future pose.
