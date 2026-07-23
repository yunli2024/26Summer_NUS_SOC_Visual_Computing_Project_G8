# Part Two Task 1: Facial Expression Classification

This pipeline extracts 68 LBF landmarks from FER-2013, aligns and scales them using the eyes, trains a class-balanced classifier, and evaluates it on the provided test split.

The default classifier is histogram gradient boosting (`hgb`). It trains much faster than a full RBF SVM on all 28,709 training images while retaining nonlinear decision boundaries. RBF SVM remains available with `--classifier svm`, but full-data training can take a long time.

## Dataset strategy

FER images are already centered face crops and are only 48x48 pixels. They are enlarged to 192x192 before LBF fitting. The default `center` mode supplies a centered face region to LBF because Haar misses many low-resolution FER faces. Two diagnostic modes are also available:

- `haar`: require Haar detection and record failures.
- `haar-fallback`: try Haar first, then use the centered FER region.

The centered-region strategy is dataset-specific. Live webcam inference should use a real face detector as in Part One.

## Setup

Activate the same Conda environment used by Part One, then install:

```powershell
python -m pip install -r requirements.txt
```

Only `opencv-contrib-python` should be installed; do not install a second OpenCV wheel alongside it.

## Run the complete Task 1 pipeline

```powershell
python task1_pipeline.py
```

The full dataset contains 35,887 images, so landmark extraction takes time. Progress and extraction speed are printed regularly. Extracted features are cached; training can later be repeated without extracting again:

```powershell
python task1_pipeline.py --stage train
```

For a quick smoke test:

```powershell
python task1_pipeline.py --max-train-per-class 50 --max-test-per-class 20 --output artifacts-smoke
```

## Outputs

The `artifacts` directory contains:

- `fer_landmark_features.npz`: cached normalized features and labels.
- `expression_classifier.joblib`: trained model for real-time use in Task 2.
- `metrics.json`: accuracy, F1 scores, timings, versions, and extraction statistics.
- `classification_report.txt`: per-class precision, recall, and F1.
- `confusion_matrix.png`: count and normalized confusion matrices.
- `failure_cases.png`: high-confidence mistakes for discussion.
- `extraction_metadata.json`: extraction settings and success rates.
- `extraction_failures.tsv`: images that could not produce valid landmarks.

The required real-time threshold is checked using repeated single-image predictions; the result is stored as `meets_30ms_requirement` in `metrics.json`.

See `TASK1_REPORT.md` for the completed results, challenge discussion, confusion analysis, and limitations based on the full dataset run.

## Task 2: real-time expression effects

Run the webcam application after Task 1 has produced `artifacts/expression_classifier.joblib`:

```powershell
python task2_realtime.py
```

The application detects faces with Haar, fits the same 68-point LBF model, applies the Task 1 normalization, predicts expressions, and smooths both landmarks and class probabilities over time.

Controls:

- `E`: toggle expression effects.
- `L`: toggle facial landmarks.
- `S`: toggle temporal smoothing.
- `C`: toggle CLAHE face-detection preprocessing.
- `Q` or `Esc`: quit.

Seven built-in effects require no external image assets: happy sparkles, a surprise star, angry red action lines, sad rain, fear echo boxes, disgust bubbles/green tint, and neutral corner markers. The predicted expression, confidence, FPS, and classifier-only inference time are displayed live.

If another camera index is needed:

```powershell
python task2_realtime.py --camera 1
```

Generate a no-camera effect preview with:

```powershell
python task2_realtime.py --preview artifacts/effects_preview.png
```

See `TASK2_REPORT.md` for the implementation design, effect mapping, stabilization strategy, and limitations to discuss during the presentation.
