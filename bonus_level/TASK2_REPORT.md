# Bonus Task 2 - Just Dance Pose-Matching Application

## Goal and implementation

The application displays an annotated reference dancer and a live webcam
dancer side by side. It detects the player with YOLOv8n Pose and compares both
pose and recent movement. Feedback includes `PERFECT`, `GREAT`, `GOOD`,
`MISS`, `MOVE!`, `HOLD`, and `SYNC`, together with total points, average
similarity, and combo count.

No model training is required or performed. The supplied pretrained
`yolov8n-pose.pt` model supplies 17 COCO keypoints.

## Runtime architecture

Task 1 preprocesses the reference video once and saves `pose_cache.npz`.
Consequently, Task 2 performs only one pose inference stream at runtime: the
webcam. This is important on the tested CPU-only environment.

The GUI and inference run independently:

1. Tkinter advances the reference video using a monotonic clock at the cached
   playback rate.
2. A worker thread reads mirrored webcam frames and runs YOLO pose inference.
3. The GUI consumes only the newest completed webcam result, so inference never
   blocks reference playback or the interface.
4. Tkinter widgets are updated only on the main thread.

## Pose normalization

Absolute image coordinates are unsuitable because the reference and player can
appear at different positions and scales. Each pose is translated around the
hip center, falling back to the shoulder center or body-joint centroid, and
divided by a robust body scale derived from shoulder width, hip width, torso
length, and visible-body extent.

Scoring uses the 12 body joints from shoulders to ankles. Face keypoints are
excluded because they add little dance information and are sensitive to image
resolution.

## Pose and motion score

The single-frame pose component combines:

- **Angle similarity (65%)**: Gaussian similarity of eight elbow, shoulder,
  hip, and knee angles, using a 32-degree tolerance.
- **Normalized joint-position similarity (35%)**: Gaussian similarity of
  corresponding normalized joints, with extra weight on wrists and ankles.

The pose score is reduced when too few corresponding body joints are visible.
For an active reference movement, pose contributes 55% of the final score. The
remaining 45% compares normalized joint-displacement vectors over approximately
0.4 seconds:

- Motion-vector agreement contributes 70% of the motion component.
- Player/reference activity-magnitude agreement contributes 30%.
- Wrists and ankles receive additional weight.
- A noise floor reduces the effect of detector jitter.
- An anti-static factor lowers the score when the reference moves but the
  player remains still.

When the reference genuinely holds a pose, motion is not required and pose
quality remains valid. If mirror acceptance is enabled, both the direct pose
and an anatomically left/right-swapped horizontal mirror are tested.

Reference-hold detection is stabilized separately from the raw motion score.
The reference activity is smoothed with an exponential moving average
(`alpha=0.35`). The game enters `HOLD` only after two consecutive scoring
samples fall below `0.07`, and it leaves `HOLD` only after the smoothed activity
reaches `0.10`. This two-threshold hysteresis prevents a short slowdown or
keypoint fluctuation near the boundary from making `HOLD` flicker while the
reference dancer is still moving. The filter resets on start/restart, pause,
stop, or loss of the player pose.

## Temporal alignment

People naturally react after the reference dancer moves. At every new webcam
result, the player is compared against all cached reference poses from the
current time back through the previous 0.8 seconds. For every candidate, both
its pose and its motion over the same player-history duration are evaluated.
The highest-scoring candidate is used and the detected lag is displayed. This
does not allow matching future movements.

## Game feedback

| Condition | Feedback | Points | Combo |
|---|---|---:|---|
| Insufficient motion history | SYNC | 0 | unchanged |
| Reference is holding | HOLD | 0 | unchanged |
| Reference moves but player is nearly static | MOVE! | 0 | resets |
| Score 85-100 | PERFECT | 1000 | increases |
| Score 70-84.99 | GREAT | 700 | increases |
| Score 55-69.99 | GOOD | 400 | increases |
| Score below 55 | MISS | 0 | resets |

`HOLD` frames do not award points, so a stable pose cannot repeatedly earn
PERFECT. Similarity and feedback are refreshed for each newly completed webcam
inference. Score, combo, and average statistics mutate at a fixed 4 Hz
game-time rate. Extra inference frames inside the same 250 ms window cannot
create extra points, so totals are comparable across machines that sustain at
least the scoring rate.

The live overlay displays the final similarity, inference time, selected lag,
mirror state, motion similarity, raw player/reference activity, and the
smoothed reference activity (`ref~`).

## Verification results

The included full-video reference output contains 2,680 cached pose records at
30 FPS, giving an approximately 89.3-second round. A scoreable person was
selected in 2,443 frames (91.16% selected-pose availability). This is an
availability measurement, not ground-truth primary-dancer identity accuracy.

Thirteen deterministic scoring tests pass:

1. Identical poses score approximately 100.
2. Translation and uniform scaling do not reduce the score.
3. An incorrectly positioned arm is penalized.
4. A mirrored pose is recognized when mirror acceptance is enabled.
5. Temporal search locates the correct delayed pose.
6. Matching reference/player motion scores approximately 100.
7. A static player is penalized while the reference is moving.
8. A genuine static hold is not penalized.
9. One isolated low-motion sample does not trigger `HOLD`.
10. Confirmed `HOLD` remains active inside the hysteresis band and exits only
    above the release threshold.
11. A fixed-rate clock permits at most one score event per time window.
12. Extra inference frames do not increase the number of scoring opportunities.
13. Invalid score intervals are rejected.

The side-by-side tester also confirms that a correctly matched player delayed
by 0.8 seconds retains a score of 100.00, including mirrored playback.

## Limitations

- The tested PyTorch build is CPU-only. Webcam inference is slower than
  reference playback, although the GUI remains responsive.
- The 0.4-second displacement comparison captures movement and direction but
  does not model long dance sequences as directly as a learned temporal model.
- Heavy occlusion, leaving the frame, low light, or keypoint jitter can reduce
  motion-score reliability.
- The included demonstration round is 30 seconds long. A longer game can be
  generated by increasing `--max-frames`.
