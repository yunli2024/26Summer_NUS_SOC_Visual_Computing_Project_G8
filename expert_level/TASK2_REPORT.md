# Part Two Task 2 - Real-time Expression Effects

## Pipeline

The real-time application reuses the same processing assumptions as Task 1:

1. Capture and mirror a webcam frame.
2. Detect faces with Haar, optionally using CLAHE preprocessing.
3. Fit the supplied 68-point LBF model inside every detected face.
4. Smooth landmarks over time and normalize them using the eye centers.
5. Load `artifacts_svm_geometry/expression_classifier.joblib`; its Pipeline appends 38 explicit geometry descriptors and predicts seven-class scores.
6. Match faces between frames and EMA-smooth their relative class scores.
7. Draw the predicted class, relative score, landmarks, and expression-specific effect.

The live HUD displays the rolling end-to-end inference pipeline separately
from classifier-only time and total FPS. The pipeline measurement covers Haar,
LBF, feature construction, classifier scoring, and temporal smoothing; it
excludes camera capture and rendering. The final
class-balanced RBF-SVM uses 136 normalized coordinates plus 38
landmark-derived geometry descriptors, with `C=10` and `gamma=0.0114943`. Its
offline repeated single-image benchmark was 18.08 ms, below the required
30 ms classifier-prediction threshold. A target-laptop benchmark can be saved
with:

```powershell
python .\expert_level\task2_realtime.py `
  --benchmark-output .\expert_level\artifacts\webcam_latency.json
```

The JSON records sample counts, mean/p50/p95 stage latency, settings, and
hardware/software metadata. It is written when the operator exits with `Q`.

## Effects

| Predicted expression | Effect |
|---|---|
| Angry | Red face tint, strong eyebrow strokes, and action rays |
| Disgust | Green tint and animated bubbles |
| Fear | Purple tint and expanding echo boxes |
| Happy | Warm tint and orbiting sparkle stars |
| Neutral | Cyan corner markers |
| Sad | Blue tint and falling rain |
| Surprise | Orange tint and an animated exclamation star |

All effects are drawn with OpenCV and require no external image assets. They can be disabled independently of recognition so the prediction labels remain visible for comparison.

## Stabilization and interaction

Raw frame-by-frame predictions can flicker because LBF points move slightly and several FER classes overlap. Two EMA stages are therefore used:

- Landmark EMA uses a current-frame weight of 0.60.
- Relative-score EMA uses a current-frame weight of 0.35.

Faces are matched by normalized center distance before either history is reused, so multiple people do not share smoothing state. Pressing `S` disables smoothing and clears all stored history, making a direct before/after demonstration possible.

## Limitations

- FER contains 48x48 grayscale face crops, while webcam frames have different lighting, resolution, pose, and background. This domain shift can reduce live accuracy.
- Haar and LBF remain sensitive to profile views and occlusion.
- The model uses landmark coordinates and landmark-derived geometry only; skin
  texture and wrinkles are unavailable.
- Temporal smoothing reduces flicker but adds a short response delay.
- SVM decision margins are softmax-normalized for smoothing and display. These
  relative scores are not calibrated confidence probabilities.
- The existing 18.08 ms result excludes Haar and LBF. The new webcam telemetry
  makes that boundary explicit; a target-laptop run is still required before
  claiming a measured end-to-end latency.

The deterministic preview in `artifacts/effects_preview.png` verifies all seven rendering branches without requiring camera access. Final webcam appearance should be checked from a normal local PowerShell session, where Part One camera access has already been confirmed.
