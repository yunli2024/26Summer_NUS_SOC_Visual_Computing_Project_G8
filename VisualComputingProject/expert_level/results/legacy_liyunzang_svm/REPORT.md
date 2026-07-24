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
python expert_demo.py --mirror --benchmark-frames 60
```

Measured after 3 warmup frames:

| Metric | Value |
|---|---:|
| Measured frames | 60 |
| Frames with detected face | 60 |
| Predicted faces | 60 |
| Stable displayed label | `neutral` for frames 7-63 |
| Confirmed label switches | 0 |
| Average full pipeline time with one face | 21.59 ms/frame |
| Average live classifier prediction time | 4.9031 ms/face |
| Full test-set classifier prediction time | 3.2867 ms/sample |

This face-positive benchmark satisfies the Expert real-time prediction target of less than 30 ms per prediction. After the six-frame initial confirmation, the label remained `neutral` for the rest of the run instead of rapidly alternating. YuNet runs on a proportionally resized input whose longest side is 640 pixels, then maps boxes back to the display frame.

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

## Real-Time Stability Update

The webcam demo keeps multi-person support instead of limiting the scene to one
face. The default remains `--max-faces 4`.

Stability improvements:

- multi-face tracking with one independent track per face;
- YuNet is now the default real-time detector, using a `0.75` confidence threshold and built-in NMS before LBF fitting;
- YuNet candidates use broad landmark boundary checks, while the optional Haar path keeps stricter geometry checks;
- new faces must remain stable for consecutive frames before being shown;
- short missed detections reuse the last stable face to reduce box flicker;
- each face track applies EMA smoothing to all 68 landmarks before rebuilding the classifier feature vector;
- each face has its own probability EMA and label state, so multiple faces do not share temporal history;
- the initial label requires six consecutive eligible frames;
- a new label must lead for eight consecutive frames and the current label must be held for at least 30 frames;
- low-confidence expression output can stay `uncertain` instead of forcing an
  incorrect class/effect.

Default temporal parameters are `--landmark-smoothing 0.85`,
`--prob-smoothing 0.85`, `--min-confidence 0.48`,
`--switch-margin 0.12`, `--initial-label-frames 6`,
`--switch-frames 8`, and `--min-label-hold-frames 30`.

Verification:

```powershell
python test_realtime_stability.py
```

Expected output:

```text
realtime_stability_tests_ok
```

The benchmark mode prints both `raw_faces` and final `faces`; this helps verify
how many detector/LBF candidates are filtered by the real-time stabilizer. This
is not a single-person workaround: `--max-faces 4` remains the default and each
face receives an independent track.

Regression checks:

- On the reported false-positive screenshot, the complete YuNet + LBF + tracker pipeline produced exactly one stable face instead of a nested background box.
- On a fixed 70-image FER test sample, YuNet detected 69 faces at the default threshold; the previous Haar settings detected 35. This is a detector recall check, not an expression-classification metric.
- `test_realtime_stability.py` also checks multi-face tracking, landmark jitter attenuation, rejection of a room-like non-face background, suppression of alternating expression labels, and a valid confirmed label switch.

The bundled 2023 YuNet ONNX model is from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) and does not require project-specific retraining.

## Improvement Strategy

Good next steps:

1. Real-time mode now uses YuNet, multi-face tracking, landmark EMA, short hold, probability EMA, and confirmed label transitions.
2. Add more geometry features, especially mouth aspect ratio, eyebrow angle, and eye-mouth relative distances.
3. Evaluate YuNet thresholds on recorded team-member webcam clips covering distance, pose, lighting, and multi-person scenes.
4. Compare the current RBF SVM with a lightweight Landmark Transformer that treats the 68 keypoints as tokens; this keeps the required keypoints-only input while targeting the current `0.4198` macro F1 ceiling.
5. Use class-specific tuning for visually similar classes such as `fear`, `sad`, and `angry`.
