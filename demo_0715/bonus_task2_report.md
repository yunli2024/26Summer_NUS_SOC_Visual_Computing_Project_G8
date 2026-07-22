# Bonus Task 2 Report

## Scope

This implementation extends Bonus Task 1 into a Just Dance-like interaction:

- run pose detection on the reference dance video;
- run pose detection on the user webcam stream;
- display both skeletons in a two-panel GUI;
- compare the user's pose with the reference dancer;
- show score, feedback, combo, visible-keypoint coverage, and temporal lag.

The default user input is webcam `0`. A video file can also be used as the user
source for offline testing when a webcam is unavailable.
For smoother demos, the GUI supports starting the reference video from a later
frame, such as frame 900, after the dancer is fully visible.

## Data and Training

No new training is required. The system uses:

- `dance_example_1.mp4` as the reference video;
- laptop webcam as the real user stream;
- `yolov8n-pose.pt` as the pretrained YOLOv8 pose detector.

The optional TikTok dataset is not required for training. It can be used later
to test more dance examples, multi-person scenes, camera angles, and partial
body visibility.

## Pipeline

For each GUI update:

1. Read one frame from the reference video.
2. Read one frame from webcam or an optional user-video source.
3. Run YOLOv8 pose detection on both frames.
4. Select the main dancer using the Task 1 area/visibility/center rule.
5. Draw skeletons and bounding boxes on both panels.
6. Normalize the reference and user poses.
7. Compare the current user pose against a sliding window of recent reference poses.
8. Display score and feedback on the user panel.

## Requirement Checklist

| Requirement | Implementation / Evidence |
|---|---|
| Reference video pose detection | `bonus_task2_app.py` runs YOLOv8 on the left panel |
| Webcam input pose detection | Default `--user-source 0`; camera open path verified locally |
| Both panels display skeletons | `draw_pose_overlay()` is used for reference and user frames |
| Scoring system | `pose_scoring.py` computes normalized pose similarity |
| Spatial alignment explanation | Center and scale normalization described below |
| Temporal alignment explanation | Sliding reference buffer with configurable `Lag frames` |
| Similarity metric explanation | Distance + joint-angle + visibility coverage |
| On-screen feedback | Score, label, combo, common keypoints, and lag are shown in the GUI |
| Offline evidence without webcam | `analyze_task2_simulation.py` writes metrics and sample images |

## Spatial Alignment

The scorer normalizes each pose before comparison:

- center: average of visible shoulders and hips when available;
- scale: maximum of shoulder width, hip width, torso height, and body-box scale;
- coordinates: keypoints are translated to the center and divided by the scale.

This makes the score less sensitive to where the user stands in the webcam
frame and how far the user is from the camera.

## Temporal Alignment

The matcher keeps a sliding buffer of recent reference poses. The current user
pose is compared with every pose in that buffer, and the best score is selected.

This supports user lag: if the user follows the reference dancer slightly late,
the system can still match the user pose to an earlier reference frame. The GUI
shows the selected lag in frames.

## Similarity Metric

The final score combines:

- normalized Euclidean distance over mutually visible keypoints;
- joint-angle agreement for elbows and knees;
- coverage bonus for having enough shared visible keypoints.

Only keypoints visible in both streams are compared. This is important for
partial body visibility and occlusion.

## Feedback

The score is mapped to feedback labels:

| Score | Feedback |
|---:|---|
| 88-100 | Perfect |
| 74-87 | Super |
| 58-73 | Good |
| 38-57 | Keep Going |
| 0-37 | Miss |

If too few shared keypoints are visible, the feedback becomes `Partial` or a
missing-pose status.

The GUI also tracks a simple combo counter. Consecutive `Super` or `Perfect`
matches increase the combo; weaker matches reset it.

For a steadier live display, the GUI shows a 5-frame moving average of the raw
pose score. The offline CSV keeps the raw per-frame score.

## Offline Validation

A simulation test was run using the reference video as both streams. The user
stream was artificially delayed by 8 frames, and the temporal matching window
was set to 15 frames.

Command:

```powershell
python analyze_task2_simulation.py --frames 90 --simulated-user-lag 8 --lag-window 15 --save-samples 4
```

Result:

| Metric | Value |
|---|---:|
| Processed frame pairs | 90 |
| Valid scored pairs after lag | 82 |
| Average score after lag | 100.00 |
| Lag hit rate after lag | 100.00% |
| Pair processing FPS | 8.55 |

This confirms that the temporal window can recover a known 8-frame lag under a
controlled test.

Additional checks:

| Scenario | Frames | Avg score | Lag hit rate | Purpose |
|---|---:|---:|---:|---|
| lag 0, window 0 | 60 | 100.00 | 100.00% | Synchronized streams should match directly |
| lag 8, window 3 | 60 | 91.14 | 0.00% | Window is too small to recover an 8-frame delay |
| lag 8, window 15 | 90 | 100.00 | 100.00% | Window is large enough to recover the delay |

Saved evidence:

- `outputs/task2_simulation/task2_summary.json`
- `outputs/task2_simulation_lag0_window0/task2_summary.json`
- `outputs/task2_simulation_lag8_window3/task2_summary.json`
- `outputs/task2_simulation/task2_metrics.csv`
- `outputs/task2_simulation/task2_sim_frame*.jpg`
- `test_pose_scoring.py`: lightweight checks for spatial normalization and temporal lag matching.

## Known Limitations

- CPU-only inference processes two YOLO streams sequentially, so GUI FPS is
  lower than pure video playback.
- Webcam lighting, camera angle, and limited body visibility can reduce score
  stability.
- The current score compares pose shape, not music beat timing.
- The lag window handles short delays, but not large choreography mistakes.

## 2026-07-16 Optimization Audit

The earlier `100` score was only a positive same-video lag test. It proved that
the temporal window worked, but did not prove that wrong actions received low
scores. The optimized implementation adds:

- batched inference for the reference and user frames;
- continuous main-dancer tracking and keypoint smoothing;
- body-keypoint confidence weighting;
- eight joint-angle comparisons and ten limb-direction comparisons;
- automatic direct/mirrored pose comparison;
- visibility-quality penalties;
- lag continuity penalties;
- stricter, nonlinear score calibration.

New evidence:

| Test | Result |
|---|---:|
| Known 8-frame lag, 90 pairs | 99.30 average score |
| Known 8-frame lag recovery | 100% |
| Pair processing speed | 12.20 FPS |
| Clearly different pose, 600-frame offset | 65.54 average score |
| Wrong pose rated `Super` or above | 0% |
| Near pose, 8-frame offset | 87.19 average score |

Pair processing improved from 8.55 FPS to 12.20 FPS (about 43%) while retaining
perfect lag recovery. The 600-frame negative test demonstrates that the scorer
no longer awards uniformly high scores to unrelated poses. Real webcam quality
still depends on camera placement, lighting, full-body visibility, and the
player's movement, so a short local user test remains necessary before the live
presentation.

An attempted upgrade to the current Ultralytics nano pose model was blocked by
the local GitHub TLS/proxy path. The existing YOLOv8 nano model was retained
because the algorithmic changes already produced measurable speed and scoring
improvements, and an unbenchmarked model should not replace a verified one.
