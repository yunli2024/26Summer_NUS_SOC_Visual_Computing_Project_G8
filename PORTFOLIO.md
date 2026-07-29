# Engineering Case Study

## Real-Time Facial and Body Keypoint Analysis

This project turns sparse face and body keypoints into four local, interactive
computer-vision applications: facial landmark tracking, keypoint-only
expression effects, reaction-lag-aware dance scoring, and gesture-controlled
gameplay. It was built for the NUS School of Computing SWS3026 Visual Computing
group project and then hardened into a reproducible engineering portfolio.

## The Engineering Problem

The difficult part was not drawing keypoints. It was converting noisy,
occasionally missing detections into feedback that feels stable and responsive
on a laptop CPU:

- face geometry must support seven-class expression recognition without giving
  the classifier access to face pixels;
- two dancers with different positions, body sizes, handedness, and reaction
  delays must still be comparable;
- a gesture controller must reject one-frame pose spikes without making
  movement and jumping feel sluggish;
- every demo must run locally with a webcam and fail clearly when a model,
  camera, or incompatible OpenCV package is missing.

## System at a Glance

| Stage | Face applications | Body applications |
|---|---|---|
| Input | Webcam or FER-style image | Webcam and reference video |
| Detection | Haar or YuNet face detector | YOLOv8n Pose |
| Representation | 68 LBF landmarks | 17 COCO body keypoints |
| Normalisation | Eye-centred translation, rotation, and scale | Torso centre and body scale |
| Temporal handling | Landmark/probability smoothing | Tracking, lag window, hysteresis |
| Output | Expression label and animated effect | Dance score or game action |

The same design principle runs through all four demos: models produce
measurements, deterministic stateful logic turns measurements into interaction,
and rendering remains separate from both.

## Decisions and Trade-offs

### 1. Geometry instead of face pixels

The expression classifier receives 136 normalised landmark coordinates and 38
derived geometric features. This satisfies the keypoint-only constraint and
keeps inference lightweight, but it deliberately gives up texture cues such as
wrinkles. A class-balanced RBF-SVM produced the best Macro-F1 among the tested
lightweight candidates; PCA was evaluated rather than assumed to help.

### 2. Explainable dance scoring

Dance similarity combines normalised pose and motion instead of treating raw
screen coordinates as ground truth. Mirror matching handles front-facing
demonstrations, a short temporal search handles player reaction lag, and
visibility masks prevent missing joints from contributing arbitrary error.
`HOLD`/`MOVE!` hysteresis prevents a static pose from farming points.

### 3. Stateful gesture control

Mario controls use upper-body joints only, so the player can remain close to a
laptop camera. Median filtering rejects isolated keypoint spikes; exponential
smoothing, confirmation frames, and release hysteresis stabilise steering; jump
is edge-triggered with cooldown and landing input buffering. Thresholds were
then relaxed using regression tests so responsiveness improved without making
shoulder-height idle poses trigger repeated jumps.

### 4. CPU-first delivery

Models load outside frame loops, pose input size is configurable, reference
poses can be cached, and every application has a camera-free preflight path.
The environment installer protects the required `opencv-contrib-python`
package from being replaced by the incompatible standard OpenCV wheel.

## Measured Evidence

| Question | Evidence |
|---|---|
| Does keypoint-only expression recognition work? | 47.31% test accuracy, **46.33% Macro-F1**, 7,178-image held-out test split |
| Is classifier inference within the course target? | **18.08 ms** per sample, below the 30 ms target |
| Can body pose run on CPU? | **31.40 ms** mean YOLO inference at `imgsz=320` |
| Was the full reference video evaluated? | 2,680 frames, **91.16%** primary-dancer detection, 23.09 FPS end-to-end offline |
| Are interaction rules regression-tested? | **52 local and CI unit tests** covering geometry, scoring, gestures, and game mechanics |

Primary artifacts:

- [Expression metrics](expert_level/artifacts_svm_geometry/metrics.json)
- [Expression confusion matrix](expert_level/artifacts_svm_geometry/confusion_matrix.png)
- [Expression failure cases](expert_level/artifacts_svm_geometry/failure_cases.png)
- [Full-video pose benchmark](bonus_level/task2_results/summary.json)
- [Dance scoring report](bonus_level/LEVEL3_DANCE_SCORING_DETAILED_REPORT.md)

These numbers are scoped claims. Expression latency covers classifier
prediction, not face detection and landmark extraction. Pose throughput is an
offline benchmark and does not guarantee the same FPS on every laptop.

## Reliability and Reproducibility

- One-command, camera-free verification: `.\check_merge.ps1`
- Root launchers for all four demos
- Relative model/data paths and explicit asset checks
- Versioned compact reference-pose cache
- Deterministic tests for geometry, lag alignment, gesture transitions, and
  platform-game mechanics
- CI on Python 3.11 for all 52 portable core-logic tests
- Original course constraints, detailed reports, metrics, plots, and failure
  cases retained beside the implementation

Live webcam behaviour cannot be proved in headless CI. The final acceptance
check therefore remains a target-laptop playtest covering camera index,
lighting, working distance, FPS, gesture comfort, and clean shutdown.

## Thirty-Second Interview Explanation

> We built a CPU-first visual-computing suite that converts sparse keypoints
> into four real-time interactions. I would highlight the boundary between
> perception and interaction: face or pose models produce noisy landmarks,
> then normalisation, temporal filtering, lag compensation, and explicit state
> machines make the output usable. We evaluated the expression model on a
> held-out 7,178-image split, benchmarked all 2,680 reference-video frames, and
> added regression tests and CI around the geometry and control logic.

## Resume Bullet Templates

Adapt ownership wording to match the work you personally completed:

- Built a CPU-first Python/OpenCV visual-computing suite with four webcam/video
  demos spanning 68-point face tracking, keypoint-only expression recognition,
  temporal dance scoring, and pose-controlled gameplay.
- Designed scale-normalised, mirror-aware, reaction-lag-aware body-pose scoring
  and validated the pipeline across a 2,680-frame video, reaching 91.16%
  primary-dancer detection and 31.40 ms mean pose inference on CPU.
- Evaluated a class-balanced keypoint-geometry SVM on 7,178 held-out images,
  achieving 46.33% Macro-F1 and 18.08 ms classifier latency without using face
  pixels as model input.
- Hardened interactive CV demos with temporal filtering, gesture state
  machines, portable assets, 52 deterministic tests, camera-free preflight
  checks, and GitHub Actions CI.

## Honest Next Steps

- Run full Stratified K-fold confirmation for the final expression pipeline.
- Measure end-to-end webcam latency separately from classifier-only latency.
- Calibrate gesture thresholds from multiple players and camera placements
  instead of a single setup.
- Add a recorded side-by-side user study comparing dance feedback with and
  without lag and mirror compensation.
