# Visual Computing Project

This folder contains the organized project stages:

- `beginner_level`: face detection and 68-point facial landmarks
- `expert_level`: expression classification and realtime effects
- `bonus_level`: pose detection and dance scoring GUI
- `resources`: shared copied resources used by the stages
- `project_materials`: original prompts, starter files, reference notebook, and backup resource copies
- `_cleanup_archive`: archived old/intermediate files moved out of the active project tree

中文运行命令指南见：

```text
VisualComputingProject\RUN_COMMANDS_CN.md
```

Teacher-provided original files are kept in the outer workspace root unchanged.

## Cleanup Archive

Redundant files are not permanently deleted. They are moved to:

```text
VisualComputingProject\_cleanup_archive\
```

The cleanup archive currently contains old Beginner code, old/intermediate Expert models, regenerated caches, Python caches, and macOS archive artifacts. See:

```text
VisualComputingProject\_cleanup_archive\README.md
VisualComputingProject\_cleanup_archive\archive_manifest.csv
```

Current active Expert realtime model is still:

```text
VisualComputingProject\expert_level\models\best_roi_cnn_face_eyes_mouth.pth
```

Original root workspace files are now classified under:

```text
VisualComputingProject\project_materials\
VisualComputingProject\bonus_level\legacy_pose_app\
```

The legacy pose GUI is kept here:

```text
VisualComputingProject\bonus_level\legacy_pose_app\danceapp.py
```

## Python Environment

Use the independent Conda environment:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe
```

GPU PyTorch has been enabled in this same environment:

```text
torch: 2.13.0+cu126
torchvision: 0.28.0+cu126
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

Check the GPU setup:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

If a training process was already running before GPU PyTorch was installed, stop it and start a new terminal/process. A Python process that already imported CPU Torch will not automatically switch to GPU.

All commands below assume you are in the outer workspace root:

```powershell
D:\AAA_SHERRY\NUS_school_computing_summer_workshop\VisualComputingProjects\project
```

## Beginner Level

Goal: face detection and 68 facial landmarks.

Run the safe setup check. This does not open the camera:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\beginner_level\tests\check_part2_setup.py
```

Run the webcam demo with original grayscale preprocessing:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\beginner_level\main.py
```

Run the webcam demo with CLAHE preprocessing:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\beginner_level\main.py --clahe
```

Run the improved setup check. This does not open the camera:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\beginner_level\improved\tests\check_improved_setup.py
```

Run the improved webcam demo:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\beginner_level\improved\main.py
```

Run the improved webcam demo with CLAHE:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\beginner_level\improved\main.py --clahe
```

Try stricter or looser Haar parameters:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\beginner_level\improved\main.py --scale-factor 1.05 --min-neighbors 5 --min-size 80 --max-size 420
```

Press `q` to quit the OpenCV window.

Improved Beginner status labels:

- `DETECTED`: Haar detected a face in the current frame.
- `CACHED`: Haar did not detect a face in the current frame, and the program is briefly showing the previous face box.
- `LOST`: no current face and no cached face box.

Improved Beginner notes:

- Green box means current-frame detection.
- Yellow box means cached previous result.
- A small green box around the nose or mouth is likely a true Haar false positive.
- A yellow box is not a fresh detection.
- Current recommended Haar parameters are `scaleFactor=1.05`, `minNeighbors=5`, `minSize=80`, `maxSize=420`.
- Haar + LBF still has known limits for strong side face, obvious head-down pose, heavy occlusion, and extreme lighting.

Robustness notes should be recorded in:

```text
VisualComputingProject\beginner_level\docs\robustness_test.md
```

## Expert Level

Goal: expression classification and realtime visual effects.

Check Expert Level setup. This does not open the camera:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\tests\check_expert_setup.py
```

Inspect resources and dataset structure:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py inspect
```

Run a small feature extraction test:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py extract --sample
```

Regenerate the small sample cache:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py extract --sample --force
```

Current Expert Level status:

- `inspect` checks dataset and resources.
- `extract --sample` extracts a small landmark-feature cache only.
- Current sample landmark extraction success rate: `82.86%` (`116 / 140`).
- Full landmark extraction success rate: `86.27%` (`30959 / 35887`).
- Full feature cache: `VisualComputingProject\expert_level\data\cache\landmark_features.npz`.
- FER small-image detection now uses multiple preprocessing strategies before giving up.
- Model comparison complete: best model is `random_forest`.
- Best model macro F1: `0.4585`; accuracy: `0.4710`; prediction time: `0.0112 ms`.
- Best model file: `VisualComputingProject\expert_level\models\best_expression_model.joblib`.
- Realtime expression demo is enabled and should be tested manually.
- Realtime demo stability has been improved without retraining: it now displays `Raw`, `Vote`, and `Stable` labels, smooths landmark features, classifies once every 5 frames, filters low-confidence or ambiguous predictions, and only switches expression after several stable updates.
- Hybrid traditional ML pipeline is implemented: landmark geometry + eye-brow HOG + mouth HOG + full-face HOG + calibrated Linear SVM.
- Hybrid SVM result did not beat the old Random Forest landmark baseline: best hybrid macro F1 `0.3643`, old baseline macro F1 `0.4585`.
- Independent CNN image branch is implemented as a separate ResNet18 grayscale baseline. It does not modify or replace the Landmark/SVM pipeline.

Extract full dataset features:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py extract
```

Train the full model:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py train
```

Run sample-first classical ML cross validation:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py cv --sample --no-xgboost --models "SVM RBF"
```

Run all classical CV models on the sample cache:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py cv --sample
```

Run full-cache CV only after the sample run is acceptable:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py cv --full
```

Run feature redundancy audit:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py feature-audit --sample
```

Run pruned sample CV:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py cv --sample --feature-mode pruned
```

Run full Landmark model comparison and Landmark ensemble:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py landmark-ensemble --compare-full
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py landmark-ensemble --fit
```

Current Landmark ensemble result:

```text
old Landmark Random Forest Macro F1: 0.4585
Landmark ensemble Macro F1: 0.4705
Landmark ensemble accuracy: 0.4970
selected strategy: weighted probability
```

The selected Landmark weights were:

```text
Logistic Regression: 0.2
SVM RBF: 0.2
Extra Trees: 0.6
Random Forest: 0.0
HistGradientBoosting: 0.0
XGBoost: 0.0
```

This improves Landmark slightly, but full CNN `fer_vgg_cnn` is still much stronger with Macro F1 `0.6510`.

Run the ROI-CNN ablation for improving `angry`, `fear`, and `sad`:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py roi-cnn --variant all
```

Run individual variants:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py roi-cnn --variant face
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py roi-cnn --variant face_eyes
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py roi-cnn --variant face_eyes_mouth
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py roi-cnn --variant face_eyes_mouth_aux
```

ROI-CNN uses Landmark only to crop:

```text
eye + brow ROI
nose + mouth ROI
```

It does not use Landmark probability fusion. The first run creates:

```text
VisualComputingProject\expert_level\data\cache\roi_landmark_regions.npz
```

ROI-CNN outputs:

```text
VisualComputingProject\expert_level\results\roi_cnn_ablation_comparison.csv
VisualComputingProject\expert_level\results\roi_cnn_classification_report.csv
VisualComputingProject\expert_level\results\roi_cnn_focus_errors.csv
VisualComputingProject\expert_level\models\best_roi_cnn_<variant>.pth
```

The ROI-CNN A/B/C/D experiment guide is in:

```text
VisualComputingProject\expert_level\roi_cnn_ablation\README.md
```

Current ROI-CNN A/B/C/D result:

```text
Best variant: face_eyes_mouth
Accuracy: 0.6747
Balanced Accuracy: 0.6584
Macro F1: 0.6560
```

Focused class F1 for the best variant:

```text
angry F1: 0.5958
fear F1: 0.5105
sad F1: 0.5036
```

Interpretation:

```text
Adding eye/brow and nose/mouth ROI gives a small improvement over the full-face CNN.
The auxiliary-head variant did not help in this run and should not be selected as the current best model.
```

Realtime Expert demo now uses the best ROI-CNN by default when the checkpoint exists:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo
```

Explicitly run the best ROI-CNN realtime model:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo --model-source roi-cnn
```

Compare with older realtime models:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo --model-source landmark
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo --model-source hybrid
```

The ROI-CNN realtime path uses Haar + LBF landmarks only for face/ROI localization. Classification uses:

```text
full face crop
eye + brow ROI crop
nose + mouth ROI crop
best_roi_cnn_face_eyes_mouth.pth
```

Haar/LBF can still use realtime preprocessing for detection, while the ROI-CNN classifier receives raw grayscale crops normalized in the same way as training.

Train the independent CNN image branch in sample mode first. The recommended current CNN baseline is `fer_vgg_cnn`:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py cnn-train --sample --model fer_vgg_cnn
```

Run comparison CNN variants:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py cnn-train --sample --model simple_cnn_gray
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py cnn-train --sample --model resnet18_gray
```

Evaluate the saved CNN checkpoint:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py cnn-evaluate
```

Only after sample CNN training is stable, run full CNN training:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py cnn-train --full
```

CNN outputs are saved in:

```text
VisualComputingProject\expert_level\models\best_cnn_expression_model.pth
VisualComputingProject\expert_level\results\cnn_training_history.csv
VisualComputingProject\expert_level\results\cnn_test_metrics.json
VisualComputingProject\expert_level\results\cnn_classification_report.csv
VisualComputingProject\expert_level\results\cnn_confusion_matrix.png
VisualComputingProject\expert_level\results\cnn_confusion_matrix_normalized.png
VisualComputingProject\expert_level\results\cnn_prediction_distribution.csv
VisualComputingProject\expert_level\results\cnn_vs_landmark.csv
```

Current CNN sample sanity-check result:

```text
model: fer_vgg_cnn
best epoch: 16
validation Macro F1: 0.4183
test Macro F1: 0.3280
test accuracy: 0.3333
average inference time: 13.1397 ms/image on CPU
prediction distribution: all 7 classes predicted
```

The earlier ResNet18 sample run collapsed toward a few classes. The `fer_vgg_cnn` sample run is no longer collapsed and is the best CNN baseline candidate so far. This is still not final CNN accuracy because sample mode uses a small subset.

Current full sample CV ranking:

```text
SVM RBF:              Macro F1 0.2831 +/- 0.0425
Logistic Regression:  Macro F1 0.2678 +/- 0.0756
Extra Trees:          Macro F1 0.2420 +/- 0.0397
Random Forest:        Macro F1 0.2408 +/- 0.0621
SVM Linear:           Macro F1 0.2307 +/- 0.0359
XGBoost:              Macro F1 0.2175 +/- 0.0691
HistGradientBoosting: Macro F1 0.1749 +/- 0.0528
```

XGBoost is now installed in `vc_sws3026` and can be included automatically. Current sample XGBoost result:

```text
Macro F1: 0.2175 +/- 0.0691
Balanced Accuracy: 0.2357 +/- 0.0704
Accuracy: 0.2406 +/- 0.0770
```

On the current sample cache, XGBoost is weaker than SVM-RBF, so it is kept as a comparison model rather than the realtime model.
The CV module forces matplotlib to use the non-GUI `Agg` backend, which avoids `RuntimeError: main thread is not in main loop` when saving confusion matrices.

Run sample probability fusion for `SVM RBF + Logistic Regression + Extra Trees`:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py ensemble
```

Current equal-weight fusion result:

```text
Macro F1: 0.2649 +/- 0.0792
Balanced Accuracy: 0.2929 +/- 0.0785
Accuracy: 0.2935 +/- 0.0808
```

Fusion confusion matrix:

```text
VisualComputingProject\expert_level\results\confusion_matrix_ensemble_probability_fusion.png
```

This does not beat sample `SVM RBF`, so it should remain an experiment rather than the realtime model.

Current feature pruning result:

```text
201 input features -> 80 kept features
Best pruned sample model: Logistic Regression, Macro F1 0.2634
Best full-feature sample model: SVM RBF, Macro F1 0.2831
```

So pruning confirms redundancy, but it does not currently improve sample accuracy.

CV outputs are saved in:

```text
VisualComputingProject\expert_level\results\
```

Feature/model summary and notes about whether too many features can hurt are in:

```text
VisualComputingProject\expert_level\docs\feature_and_model_summary.md
```

The FER cache does not contain `subject_id` or `video_id`, so the CV module reports the leakage risk and uses `StratifiedKFold` fallback. For future webcam/video frames, group metadata must be added before using `StratifiedGroupKFold`.

Extract hybrid landmark + HOG features:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py extract --hybrid --sample --force
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py extract --hybrid --force
```

Train and evaluate the hybrid SVM model:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py train --feature-set hybrid
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py evaluate --feature-set hybrid
```

Evaluate the full model:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py evaluate
```

Run realtime expression effects. This opens the camera:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo
```

The default realtime view is compact: labels follow each face box instead of filling the top-left corner. For detailed Raw/Vote/Margin/Queue debug text:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo --debug-hud
```

For a calmer default view, realtime demo now shows only the primary face and hides pose labels. Enable extras only when needed:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo --multi-face
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo --show-pose
```

Realtime detection now also filters very small/inner face boxes, smooths displayed boxes, and briefly shows `CACHED` when a face is temporarily missed.
Realtime detection now uses dynamic ROI tracking: after a face is found, Haar first searches near the previous smoothed face box, then periodically falls back to full-frame detection. This reduces box jumping and mouth/lip false positives without changing the model.
Realtime landmarks are also smoothed per face track; jawline points are smoothed more strongly and drawn smaller/darker because they are less reliable during head pose changes.
Realtime display can use an expanded face ROI, but LBF fitting now uses a much smaller expansion to avoid pulling contour points into hair/background. Tune `REALTIME_LBF_EXPAND_X`, `REALTIME_LBF_EXPAND_TOP`, and `REALTIME_LBF_EXPAND_BOTTOM` in `expert_level\src\config.py`.
Realtime demo now estimates and smooths yaw/pitch/roll with `cv2.solvePnP()` and shows a simple head direction label near each face box.

Compare realtime preprocessing modes:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo --preprocess raw
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo --preprocess clahe
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo --preprocess gamma
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo --preprocess clahe-gamma --debug-hud
```

Realtime demo default is `--model-source auto`. Since the current hybrid SVM is weaker than the landmark Random Forest baseline, auto mode keeps the stronger landmark model. To force hybrid comparison:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo --model-source hybrid
```

If camera index 0 does not work:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\expert_level\main.py demo --camera 1
```

Press `q` to quit the OpenCV window.

Expert realtime note:

- The camera demo uses original-frame coordinates for face boxes and landmarks.
- FER training extraction still uses upscaling internally, but that is not used for drawing on webcam frames.
- Effects follow the `Stable` label, not the raw per-frame prediction.
- If `Raw` changes quickly but `Stable` changes slowly, the stabilizer is working as intended.
- If `Stable` becomes `uncertain`, the model is not confident enough or the top two expressions are too close.
- Offline confidence check shows the saved Random Forest model has low confidence overall: mean `0.3921`, median `0.3080`. Realtime confidence filtering is therefore set to `0.35`, not `0.45`.

## Bonus Level

Goal: pose and dance scoring. This part is currently a placeholder and has not been implemented.

Run the placeholder entry:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe VisualComputingProject\bonus_level\main.py
```

## Running From Inside Each Folder

If your terminal is already inside `VisualComputingProject\beginner_level`:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe main.py
```

If your terminal is already inside `VisualComputingProject\expert_level`:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe main.py inspect
D:\miniconda3\envs\vc_sws3026\python.exe main.py extract --sample
D:\miniconda3\envs\vc_sws3026\python.exe main.py extract --sample --force
D:\miniconda3\envs\vc_sws3026\python.exe main.py cnn-train --sample
D:\miniconda3\envs\vc_sws3026\python.exe main.py cnn-evaluate
D:\miniconda3\envs\vc_sws3026\python.exe main.py demo
```

If your terminal is already inside `VisualComputingProject\bonus_level`:

```powershell
D:\miniconda3\envs\vc_sws3026\python.exe main.py
```
