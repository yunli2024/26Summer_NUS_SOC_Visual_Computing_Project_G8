# Beginner Level Robustness Test Record

Save screenshots in `beginner_level/outputs/screenshots/`.
Save result files in `beginner_level/outputs/robustness_results/`.

For each case, test both:

- Haar with raw grayscale
- Haar with CLAHE preprocessing
- YuNet with the same camera placement
- The same duration and scene conditions for each run

## Test Table

| Case | Haar success rate | YuNet success rate | DETECTED/CACHED/LOST notes | Landmarks stable? | Jitter/drift? | Haar avg FPS | YuNet avg FPS | Raw result | CLAHE result | False positive notes | Observation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Normal lighting |  |  |  |  |  |  |  |  |  |  |  |
| Dark lighting |  |  |  |  |  |  |  |  |  |  |  |
| Strong light |  |  |  |  |  |  |  |  |  |  |  |
| Frontal face |  |  |  |  |  |  |  |  |  |  |  |
| Near left edge |  |  |  |  |  |  |  |  |  |  |  |
| Near right edge |  |  |  |  |  |  |  |  |  |  |  |
| Near top edge |  |  |  |  |  |  |  |  |  |  |  |
| Near bottom edge |  |  |  |  |  |  |  |  |  |  |  |
| Turn head left/right |  |  |  |  |  |  |  |  |  |  |  |
| Look up |  |  |  |  |  |  |  |  |  |  |  |
| Look down |  |  |  |  |  |  |  |  |  |  |  |
| Nose/mouth false positive |  |  |  |  |  |  |  |  |  |  |  |
| Cover mouth |  |  |  |  |  |  |  |  |  |  |  |
| Cover eyes |  |  |  |  |  |  |  |  |  |  |  |
| Far from camera |  |  |  |  |  |  |  |  |  |  |  |
| Multiple people |  |  |  |  |  |  |  |  |  |  |  |

## Suggested Manual Protocol

For each case:

1. Run Haar for about 20 seconds with `--detector haar`.
2. Record detection success rate and average FPS.
3. Repeat the same scene with `--detector yunet`.
4. For the lighting comparison, repeat Haar with `--preprocess clahe`.
5. Record whether the screen says `DETECTED`, `CACHED`, or `LOST`.
6. If a small nose/mouth box appears, record whether it is green `DETECTED` or yellow `CACHED`.
7. Compare landmark jitter and drift.

## Notes

- If face detection fails, LBF landmarks usually cannot run.
- If landmarks jump between frames, record it as jitter.
- If landmarks slowly move away from the face, record it as drift.
- Compare FPS between original grayscale and CLAHE.
- Improved padding is mainly expected to help when the face is close to image edges.
- Improved landmark smoothing should reduce small frame-to-frame jitter, but can add slight lag.
- `DETECTED` means Haar found a face in the current frame.
- `CACHED` means Haar did not find a face and the program is briefly showing the previous face box.
- `LOST` means the cached box has expired and no face is currently tracked.
- A small green box around nose/mouth is likely a true Haar false positive.
- A yellow box is a cached previous result, not a fresh detection.
