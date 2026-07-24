# Bonus Task 1 - TikTok Dance Pose Analysis

## Scope

The project instructions do not require evaluation of the complete TikTok dataset or training a new model. Two short segments were selected for testing:

1. The supplied single-dancer `dance_example_1.mp4`.
2. `TikTokDataset/TikTok_Raw_Videos/seq_00001_00009/YouTube.mp4`, which frequently contains multiple people.

For each video, analysis started at source frame 300, processed every third frame, and stopped after 60 analyzed frames. YOLOv8n Pose used an input size of 416 pixels on CPU.

## Pipeline

1. YOLOv8n Pose detects people, boxes, and 17 COCO body keypoints.
2. Low-confidence keypoints below 0.35 are treated as invisible.
3. A primary dancer is selected from all people.
4. Valid keypoints are smoothed with an EMA current-frame weight of 0.60.
5. The selected dancer is drawn with a yellow box and colored skeleton; other people receive thin gray boxes.
6. An annotated MP4, contact sheet, and JSON metrics are exported.

## Primary dancer selection

On the first visible frame, each person is scored using:

- Body-box area: 60%.
- Proximity to frame center: 30%.
- Detector confidence: 10%.

After a dancer has been selected, temporal continuity becomes dominant:

- Continuity with the previous box: 45%.
- Body-box area: 30%.
- Proximity to frame center: 15%.
- Detector confidence: 10%.

Continuity combines box IoU and normalized center displacement. This prevents identity switches when a nearby background person briefly becomes larger or more confident.

## Results

| Metric | Single-dancer example | Multi-person TikTok sample |
|---|---:|---:|
| Analyzed frames | 60 | 60 |
| Primary dancer detected | 100% | 100% |
| Frames with multiple people | 1.67% | 85.00% |
| Average visible joints | 16.95 / 17 | 13.77 / 17 |
| Average visible-joint confidence | 0.935 | 0.874 |
| Average YOLO inference time | 492.03 ms | 293.74 ms |
| End-to-end processing speed | 1.72 FPS | 2.15 FPS |

These numbers describe only the selected segments and must not be presented as full-dataset accuracy.

## Challenges, strategies, and findings

### Multiple people

The TikTok segment contains multiple people in 85% of analyzed frames. Selecting the largest detection alone can switch to a person entering the foreground. Adding frame-center preference and previous-box continuity kept the yellow primary-dancer box on the central performer throughout the inspected contact sheet.

### Missing and unreliable keypoints

The single-dancer video retained almost all 17 joints. The crowded vertical TikTok clip averaged 13.77 visible joints because arms and legs were partially occluded or outside the narrow video region. Confidence filtering prevents uncertain zero/incorrect coordinates from being connected into long skeleton lines.

### Jitter

YOLO estimates each frame independently, so joint positions move slightly even during stable poses. EMA smoothing reduces this jitter. It only reuses a previous point when that joint is confident in both frames.

### Fast movement and occlusion

Fast hands can blur or overlap the body. The confidence threshold removes many bad points, but temporarily missing joints remain a limitation. The current implementation deliberately does not hallucinate a joint after it becomes invisible.

### Runtime

This environment has CPU-only PyTorch. YOLO inference is therefore much slower than real time. Using `imgsz=416`, frame stride, and short selected clips makes offline analysis practical. A CUDA-capable PyTorch installation would be the main route to real-time performance.

## Limitations and possible improvements

- Replace box-continuity matching with a dedicated tracker such as ByteTrack for long occlusions.
- Use a motion model or Kalman filter to bridge a small number of missing frames.
- Adapt the confidence threshold per joint instead of using one global threshold.
- Evaluate more dance styles, camera motions, and full-body crops.
- Use a CUDA GPU or a smaller inference size for real-time analysis.
- The current statistics measure detection availability and confidence, not ground-truth keypoint accuracy, because the selected videos have no pose annotations.

Visual evidence is stored in `task1_results/dance_example_1/contact_sheet.jpg` and `task1_results/seq_00001_00009_YouTube/contact_sheet.jpg`. Exact machine-readable settings and results are stored in each `analysis.json`.
