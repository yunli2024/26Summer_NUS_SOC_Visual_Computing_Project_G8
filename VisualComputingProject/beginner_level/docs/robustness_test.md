# Beginner Level Robustness Test Record

Save screenshots in `VisualComputingProject/beginner_level/outputs/screenshots/`.
Save result files in `VisualComputingProject/beginner_level/outputs/robustness_results/`.

For each case, test both:

- Original grayscale
- CLAHE preprocessing
- Baseline version
- Improved version

## Test Table

| Case | Baseline success rate | Improved success rate | DETECTED/CACHED/LOST notes | Landmarks stable? | Jitter/drift? | Baseline avg FPS | Improved avg FPS | Original grayscale result | CLAHE result | False positive notes | Observation |
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

1. Run baseline for about 20 seconds.
2. Record approximate detection success rate and average FPS.
3. Run improved without CLAHE for about 20 seconds.
4. Run improved with CLAHE for about 20 seconds.
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
