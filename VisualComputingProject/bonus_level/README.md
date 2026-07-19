# Bonus Level: Pose Dance Scoring

This folder contains the final pose-detection and dance-scoring application.
The earlier single-file prototype and its duplicate model have been removed.

## Features

- YOLOv8 Pose detection with 17 COCO keypoints
- Main-dancer selection and temporal tracking
- Confidence filtering and pose smoothing
- Scale- and translation-normalized pose comparison
- Mirrored webcam handling and left/right keypoint correction
- Sliding-window temporal alignment
- Position, joint-angle, bone-vector, and coarse-pose scoring
- Realtime feedback, combo, total score, and dance summary

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
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\bonus_level\main.py
```

Use the left panel to play a reference video and the right panel to start the
webcam. Start both streams to receive aligned pose scores and feedback.
