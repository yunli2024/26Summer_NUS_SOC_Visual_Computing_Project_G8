# Part Three: Dance Pose Analysis and Just Dance Game

This folder contains both bonus tasks:

- **Bonus Task 1** — analyze dance videos and robustly track the primary dancer.
- **Bonus Task 2** — compare a webcam dancer with a reference video in real time and give game-style feedback.

Neither task trains a model. Both use the supplied pretrained `yolov8n-pose.pt` model.

## Bonus Task 1

`pose_analyzer.py` uses the supplied YOLOv8 pose model to analyze a few selected dance videos. It does not train a new model or scan the whole TikTok dataset.

The pipeline:

1. Detects every person and 17 COCO body keypoints.
2. Selects the primary dancer using body size, proximity to the frame center, detection confidence, and continuity with the previous frame.
3. Filters low-confidence keypoints.
4. Applies EMA smoothing to valid keypoints.
5. Draws the primary dancer in color and other people in gray.
6. Exports an annotated video, contact sheet, and JSON metrics.
7. Saves `pose_cache.npz` for the Task 2 scoring application.

## Setup

In the existing `visual-computing` Conda environment, keep the `opencv-contrib-python` package required by Parts One and Two. Install the other runtime dependencies, then install Ultralytics without dependency resolution so pip does not add a conflicting `opencv-python` wheel:

```powershell
python -m pip install -r requirements.txt
python -m pip install ultralytics==8.4.98 --no-deps
```

## Run

Analyze the supplied example:

```powershell
python pose_analyzer.py --start-frame 300 --max-frames 300
```

Analyze selected TikTok videos without processing the whole dataset:

```powershell
python pose_analyzer.py `
  "TikTokDataset/TikTok_Raw_Videos/seq_00001_00009/YouTube.mp4" `
  "TikTokDataset/TikTok_Raw_Videos/seq_00010_00028/YouTube.mp4" `
  --max-frames 300 --stride 2
```

Useful options:

- `--device 0`: use the first CUDA GPU if PyTorch detects it.
- `--stride 2`: process every second frame.
- `--start-frame 300`: skip an intro before analysis.
- `--no-video`: collect metrics/contact sheets without writing MP4 output.
- `--show`: display frames while processing; press Q or Esc to stop.
- `--keypoint-confidence 0.35`: change the threshold for visible joints.
- `--smoothing-alpha 0.6`: change EMA responsiveness.

Outputs are stored under `task1_results/<video-name>/`.

See `TASK1_REPORT.md` for the tested-video settings, real measurements, primary-dancer scoring strategy, challenges, and limitations.

## Bonus Task 2

`just_dance_app.py` opens a two-panel GUI. The left panel plays the annotated reference dancer; the right panel shows the mirrored webcam, detected skeleton, similarity, reaction-delay match, score, and combo.

The program uses the Task 1 pose cache for the reference side, so only webcam
frames require YOLO inference during the game. Scoring combines pose similarity
with motion over a 0.4-second history window, tolerates up to 0.8 seconds of
reaction delay, and can accept mirrored moves. Reference hold frames display
`HOLD` without awarding points; standing still while the reference moves
displays `MOVE!`.

First generate the 30-second reference clip and cache (the included output has already been generated):

```powershell
python pose_analyzer.py dance_example_1.mp4 `
  --start-frame 300 --max-frames 300 --stride 3 `
  --image-size 416 --contact-every 60 --output task2_results
```

Validate all Task 2 inputs without opening the camera:

```powershell
python just_dance_app.py --check
python -m unittest -v test_dance_scoring.py
```

Start the game:

```powershell
python danceapp.py
```

`python just_dance_app.py` is equivalent; `danceapp.py` is retained as the course-starter-compatible entry point.

Controls:

- **Start / Restart** starts the camera and resets the score.
- **Pause / Resume** freezes both the game clock and camera inference.
- **Stop** stops the round while keeping the GUI open.
- **Accept mirrored moves** lets a dancer perform a left/right mirrored version of the reference.

On a CPU-only machine, webcam pose inference is expected to update more slowly than the 10 FPS reference video. The reference playback remains clock-synchronized and the temporal matcher scores every newly available camera pose. If necessary, use `--image-size 320` for faster but slightly less precise detection.

Use `--motion-window 0.4` to adjust the motion-comparison interval. A shorter
window responds faster but is more sensitive to keypoint jitter; a longer
window emphasizes larger movements.

### Scoring A/B tester

`scoring_video_tester.py` plays the same cached video on both sides, so lag and
mirror handling can be tested without a camera or additional YOLO inference.
The simulated-player panel can be delayed or mirrored independently of the
scorer's allowed reaction lag and mirror acceptance:

```powershell
python scoring_video_tester.py
```

Use the two sliders and two checkboxes for live A/B comparisons. A deterministic
headless validation is also available:

```powershell
python scoring_video_tester.py --check
```

See `TASK2_REPORT.md` for the scoring formula, temporal alignment design, test results, and limitations.
