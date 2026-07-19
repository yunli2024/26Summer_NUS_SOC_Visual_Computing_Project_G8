# Bonus Level Action Game: Ursina 3D Three-Lane Runner

This folder is a parallel version of `bonus_level` that reuses the existing
YOLOv8 Pose detector, main-person tracker, keypoint smoothing, mirrored webcam
handling, and upper-body gesture recognizer. The current main program renders a
third-person 3D runner using Ursina.

## Features

- YOLOv8 Pose detection with 17 COCO keypoints
- Main-player selection and keypoint smoothing
- Upper-body gesture state machine with calibration
- Ursina 3D third-person camera behind and above the player
- Three perspective lanes, smooth lane switching, jump arc, slide pose, looping road
- Moving obstacles, coins, collision, score, game over, and restart
- Keyboard fallback controls

## Directory layout

```text
bonus_level_action_game/
|-- data/input/reference_videos/tiktok/  # copied parallel data location
|-- outputs/runtime/                     # runtime output location
|-- src/                                 # pose pipeline plus Ursina runner prototype
|-- main.py
`-- README.md
```

The application uses these shared project resources:

```text
VisualComputingProject\resources\pose_models\yolov8n-pose.pt
```

## Run

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\bonus_level_action_game\main.py
```

The environment needs `ursina` installed. It has been installed in
`D:\miniconda3\envs\vc_sws3026` for this prototype.

Pose controls:

- Move your upper body/person left: move left one lane
- Move your upper body/person right: move right one lane
- Raise both hands quickly above shoulders: jump
- Cross both arms near the chest: slide

Keyboard fallback:

- `A` / Left arrow: move left
- `D` / Right arrow: move right
- `W` / Up arrow: jump
- `S` / Down arrow: slide
- `C`: recalibrate pose
- `R`: restart after game over
- `Esc`: quit

Action thresholds, calibration frame count, camera index, optional video input
path, and 3D runner settings are in:

```text
VisualComputingProject\bonus_level_action_game\src\config.py
```

Use webcam by default:

```python
RUNNER_CAMERA_INDEX = 0
RUNNER_VIDEO_PATH = ""
```

Use a video file by setting:

```python
RUNNER_VIDEO_PATH = r"path\to\video.mp4"
```
