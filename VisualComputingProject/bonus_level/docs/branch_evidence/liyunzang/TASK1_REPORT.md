# Bonus Task 1 Report

## Scope

This implementation covers Bonus Level Task 1 only:

- load and play a reference dance video;
- run YOLOv8 pose detection on the reference video;
- visualize body keypoints and skeleton;
- analyze keypoint detection challenges.

Webcam comparison and dance scoring are reserved for Bonus Task 2.

## Method

The system uses the provided `yolov8n-pose.pt` model through the Ultralytics YOLO API. The model is loaded once at startup and reused for every frame.

For each frame:

1. Run YOLOv8 pose detection.
2. Read COCO-format 17 body keypoints.
3. Select a main dancer using:
   - bounding-box area;
   - visible keypoint count;
   - closeness to the frame center.
4. Draw the skeleton and keypoints.
5. Display per-frame statistics: detected persons, visible keypoints, and inference time.

## Data and Training

No new model training is required for Bonus Task 1. The implementation uses:

- `dance_example_1.mp4` as the default reference dance video;
- `yolov8n-pose.pt` as the provided pretrained pose detector.

The optional TikTok dataset is useful for extra robustness testing and for
finding more failure cases, but Task 1 does not require training or evaluating
on the full dataset.

## Main Dancer Selection

When multiple people appear, the system scores each detected person:

```text
score = bbox_area_weight + visible_keypoint_weight + center_position_weight
```

This prefers a dancer who is large, complete, and near the center of the reference video. Other people can still be shown unless `Main dancer only` is enabled.

## Evaluation

The headless evaluator was run on five 120-frame segments of the provided
reference video, starting at frames 0, 450, 900, 1350, and 1800. This samples
the opening, middle, and later parts of the dance clip instead of relying on
one short sequence.

Command:

```powershell
python analyze_segments.py --frames 120 --save-samples 3
```

Aggregate result over 600 processed frames:

| Metric | Value |
|---|---:|
| Person detection rate | 84.67% |
| Multi-person frame rate | 57.50% |
| Average visible keypoints on selected dancer | 14.28 / 17 |
| Average measured YOLO inference time | 48.84 ms/frame |

Segment-level observations:

| Start frame | Person detection | Multi-person rate | Visible keypoints | Inference |
|---:|---:|---:|---:|---:|
| 0 | 23.33% | 0.00% | 3.59 | 52.92 ms |
| 450 | 100.00% | 100.00% | 17.00 | 50.90 ms |
| 900 | 100.00% | 11.67% | 17.00 | 47.94 ms |
| 1350 | 100.00% | 78.33% | 17.00 | 44.94 ms |
| 1800 | 100.00% | 97.50% | 16.82 | 47.52 ms |

The opening segment is difficult because the dancer is not yet consistently
visible to the model. Once the choreography is fully in view, detection becomes
stable, but many later frames contain multiple detected people. This confirms
that a main-dancer selection rule is required for Task 1 and will be even more
important for Task 2 scoring.

Saved evidence:

- `outputs/task1_segments_summary.json`: aggregate and segment metrics.
- `outputs/segment_start*_frames120/task1_summary.json`: per-segment summary and sample file manifest.
- `outputs/segment_start*_frames120/sample_frame*.jpg`: annotated sample frames.

## Observed Challenges

| Challenge | Effect | Handling |
|---|---|---|
| Multiple people | Up to 100.00% multi-person frames in sampled segments | Select a main dancer using area, visibility, and center score |
| Opening / no clear dancer | Only 23.33% person detection in the first 120 frames | Report this as a failure case; start demos after the dancer is visible if needed |
| Fast movement | Motion blur can lower keypoint confidence | Keep lower confidence thresholds available in the GUI |
| Occlusion | Hidden arms/legs may disappear or jump | Draw only keypoints above confidence threshold |
| Low resolution | Small joints such as wrists/ankles become unstable | Report visible keypoint count and avoid assuming all points exist |
| Partial body visible | Lower-body or upper-body points may be missing | Preserve partial skeleton; Task 2 should score only mutually visible joints |
| Out-of-frame dancer | Bounding box/keypoints may be incomplete | Main dancer score penalizes low visibility |

## Deliverables

- `bonus_task1_app.py`: interactive reference-video GUI.
- `analyze_reference.py`: headless benchmark and evidence capture.
- `analyze_segments.py`: multi-segment evaluator for broader evidence.
- `outputs/task1_segments_summary.json`: quantitative multi-segment summary.
- `outputs/segment_start*_frames120/sample_frame*.jpg`: annotated example frames.

## Next Step Toward Task 2

Reuse `pose_pipeline.py` for both reference video and webcam. Task 2 should add:

- reference pose buffer;
- webcam pose stream;
- spatial normalization around hips/shoulders;
- temporal alignment using a small sliding window;
- pose similarity score and feedback labels.

## 2026-07-16 Optimization Audit

The original Task 1 implementation met the functional requirement, but its
per-frame main-dancer choice could jump between people. The optimized pipeline
now uses temporal continuity (box overlap and center motion) to keep the same
dancer selected, and applies lightweight keypoint smoothing.

The five-segment, 600-frame evaluation was repeated after optimization:

| Metric | Original | Optimized |
|---|---:|---:|
| Person detection rate | 84.67% | 84.67% |
| Measured inference time | 48.84 ms | 28.73 ms |
| Average visible keypoints | 14.28 / 17 | 14.29 / 17 |

The unchanged aggregate detection rate is expected: the first sampled segment
contains the video opening before the dancer is clearly visible. Every sampled
segment from frame 450 onward retained 100% person detection. The optimized
version therefore improves speed and dancer continuity without reducing pose
coverage.
