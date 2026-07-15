# Expert Level Report

## Method

The Expert pipeline keeps the project requirement that expression classification must use facial keypoints instead of the full face image.

1. Read the FER-style dataset directly from `../expert/facial_expression_dataset.zip`.
2. Validate that both `train` and `test` contain all seven classes: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.
3. For every selected image, run the Beginner-style Haar + LBF facial keypoint pipeline.
4. Convert the 68 landmarks into model input:
   - normalize x/y coordinates by the face box center and scale;
   - add lightweight geometric features such as eye opening, mouth opening, mouth width, brow-eye distance, and nose/chin distances.
5. Train a lightweight RBF-kernel SVM classifier with standardized landmark features.
6. Load the trained classifier in `expert_demo.py` for real-time webcam prediction and expression-driven effects.

The classifier does not use the raw/full face image as input.

## Current Evaluation

Training command used for the current saved model:

```powershell
python train_expert.py --workers 8
```

Current saved model:

```text
models/expression_classifier.joblib
```

Metrics:

| Metric | Value |
|---|---:|
| Train samples selected | 28709 |
| Test samples selected | 7178 |
| Landmark extraction failures | 0 |
| Accuracy | 0.4536 |
| Macro F1 | 0.4198 |
| Average classifier prediction time | 3.2867 ms |

The prediction stage is far below the 30 ms real-time target. The heavier part of the webcam pipeline is face detection + LBF landmark fitting, which is already loaded once outside the loop.

## Real-Time Benchmark

Command:

```powershell
python expert_demo.py --mirror --benchmark-frames 30 --min-neighbors 4
```

Measured after 3 warmup frames:

| Metric | Value |
|---|---:|
| Measured frames | 30 |
| Frames with detected face | 0 |
| Predicted faces | 0 |
| Average full pipeline time without detected face | 7.26 ms/frame |
| Full test-set classifier prediction time | 3.2867 ms/sample |

This satisfies the Expert real-time prediction target of less than 30 ms per prediction. In the latest benchmark the webcam view did not contain a Haar-detected face, so the benchmark mainly proves the camera and empty-frame path. Earlier face-positive runs measured the full detection + LBF + prediction path in the 14-19 ms/frame range; the current full-data classifier adds about 3.3 ms per face.

## Confusion Matrix

Rows are actual labels; columns are predicted labels.

| actual \ predicted | angry | disgust | fear | happy | neutral | sad | surprise |
|---|---:|---:|---:|---:|---:|---:|---:|
| angry | 421 | 33 | 91 | 110 | 137 | 110 | 56 |
| disgust | 30 | 48 | 9 | 5 | 5 | 9 | 5 |
| fear | 158 | 35 | 222 | 103 | 174 | 182 | 150 |
| happy | 157 | 20 | 105 | 1102 | 174 | 156 | 60 |
| neutral | 173 | 23 | 108 | 117 | 558 | 176 | 78 |
| sad | 218 | 32 | 172 | 130 | 249 | 394 | 52 |
| surprise | 58 | 17 | 69 | 55 | 77 | 44 | 511 |

Most recognizable classes in this run are `surprise`, `disgust`, and `happy`. The hardest classes are `angry`, `sad`, and `neutral`, which often share similar landmark geometry around the mouth and eyes.

## Failure Cases And Challenges

No landmark extraction failures were recorded in the full train/test run, but the model still makes classification mistakes. Main causes:

- FER images are low-resolution, so LBF landmarks can be noisy after upscaling.
- Some classes have overlapping geometry; for example, `sad`, `neutral`, and `angry` can look similar when only sparse facial landmarks are used.
- Haar sometimes misses a cropped FER face, so the training pipeline uses a centered face-box fallback and records it in the metrics.
- Landmarks capture shape but not texture details such as wrinkles, teeth, or subtle shading, which full-image expression classifiers usually exploit.

## Real-Time Visual Effects

`expert_demo.py` displays the predicted expression and applies effects:

| Expression | Effect |
|---|---|
| happy | sparkle overlay |
| surprise | burst rays |
| angry | red tint |
| sad | blue tint |
| neutral | neutral border |

Run:

```powershell
python expert_demo.py --mirror
```

## Improvement Strategy

Good next steps:

1. Add temporal smoothing for expression labels over a longer window in webcam mode.
2. Add more geometry features, especially mouth aspect ratio, eyebrow angle, and eye-mouth relative distances.
3. Compare Haar with MediaPipe or another modern detector for more stable face boxes before LBF.
4. Try a lightweight model on landmark-rendered images rather than raw face images; this still uses keypoints as input.
5. Use class-specific tuning for visually similar classes such as `fear`, `sad`, and `angry`.
