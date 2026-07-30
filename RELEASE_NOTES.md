# Visual Computing Suite v1.1.0-rc.1

This release candidate improves the trustworthiness and portability of the
portfolio without changing the course-defined keypoint-only scope.

## Highlights

- Beginner now offers a real Haar/YuNet detector switch:
  `.\run_beginner_level.ps1 --detector yunet`.
- Dance scoring runs on fixed 250 ms game-time windows. Faster inference can
  refresh the display more often, but cannot manufacture extra score events.
- The 2,680-frame pose benchmark is reported as **91.16% selected-pose
  availability**, not identity accuracy.
- Reports, JSON artifacts, README metrics, and the architecture guide now use
  the same evidence vocabulary.
- The portable regression suite contains 50 tests.

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
- Webcam/GUI behaviour and end-to-end camera latency require target-laptop
  validation.

## Run

```powershell
conda env create -f environment_setup\environment.yml
conda activate vc_sws3026
.\environment_setup\install_runtime.ps1
.\check_merge.ps1
```

See `README.md`, `PORTFOLIO.md`, and `docs/development/merge.md` for the
recruiter-facing overview, measured evidence, and maintained architecture.
