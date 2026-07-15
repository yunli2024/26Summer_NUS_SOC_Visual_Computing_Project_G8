# Beginner Robustness Notes

## What To Test

| Scenario | Expected behavior with Haar + LBF | Likely cause when it fails | Practical improvement |
|---|---|---|---|
| Normal front-facing face, stable indoor light | Face box and 68 landmarks should be stable | N/A | Use this as the baseline demo case |
| Strong backlight or dim light | Face may flicker or disappear; landmarks may jump | Haar relies on contrast-like patterns and can miss low-contrast faces | Use `--preprocess clahe`, add front lighting, or tune `--min-neighbors` |
| Large yaw/pitch angle | Haar often misses side faces; LBF landmarks become inaccurate | The cascade and LBF model are strongest for near-frontal faces | Compare with `--detector mediapipe` or use a profile/modern detector |
| Mouth covered by hand | Face box may remain, but mouth landmarks can be pulled toward the occluder | LBF fits a face-shape model even when local evidence is hidden | Report this as an occlusion limitation; reject low-quality fits in later stages |
| Fast motion | Detection and landmarks can lag or jitter | Motion blur reduces the feature evidence used by both models | Increase light, lower camera resolution, or smooth landmarks over time |

## Alternative Detector Experiment

The main script supports an optional MediaPipe face detector:

```powershell
python -m pip install mediapipe
python beginner_demo.py --detector mediapipe --mirror
```

Use the same LBF landmark model after the face box is found. This gives a focused comparison:

- Haar Cascade: fast, lightweight, no extra package beyond OpenCV, but sensitive to lighting, pose, and occlusion.
- MediaPipe Face Detection: usually more robust to scale, mild pose changes, and imperfect lighting because it is trained as a modern detector, but it adds an extra dependency and may be less transparent than the classic cascade.

For the presentation, record one successful baseline case and at least one failure case such as side pose, low light, or hand occlusion. Then run the optional MediaPipe command and compare whether the face box is more stable.
