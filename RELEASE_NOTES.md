# Visual Computing Suite v1.1.0-rc.2

This release candidate improves the trustworthiness and portability of the
portfolio without changing the course-defined keypoint-only scope.

## Highlights

- Expert webcam inference now shows rolling pipeline latency and can export
  mean/p50/p95 stage timings with hardware/software metadata.
- SVM margin-derived values are labelled as relative scores, not calibrated
  confidence probabilities.
- A profile-aware asset checker catches missing or truncated release, demo, and
  training inputs before presentation day.
- Dance and Mario webcam apps accept `--device cpu`, `--device 0`, and other
  Ultralytics device selectors without changing their CPU-first defaults.
- The portable regression suite contains 54 tests, including primary-dancer
  identity-continuity and smoothing coverage.

## Included runtime assets

- Haar and YuNet face detectors
- Tuned keypoint-only expression classifier
- YOLOv8n pose model
- Full-video reference pose cache
- Two short reference videos

## Assets that must be restored locally

The 54 MiB LBF model and FER-style training archive are excluded from Git.
Follow `resources/README.md` before running face landmarks, expression webcam
inference, or full expression-model training.

## Evidence boundaries

- The 18.08 ms expression figure measures classifier prediction only.
- Selected-pose availability means a scoreable person was returned; the videos
  do not contain ground-truth dancer identity labels.
- Webcam/GUI behaviour still requires target-laptop validation. The Expert app
  now provides `--benchmark-output` so that validation produces a reviewable
  latency artifact instead of an anecdotal FPS observation.

## Run

```powershell
conda env create -f environment_setup\environment.yml
conda activate vc_sws3026
.\environment_setup\install_runtime.ps1
python .\environment_setup\check_assets.py --profile demo
.\check_merge.ps1
```

For exact dependency reproduction, use
`.\environment_setup\install_runtime.ps1 -Locked`.

See `README.md`, `PORTFOLIO.md`, and `docs/development/merge.md` for the
recruiter-facing overview, measured evidence, and maintained architecture.
