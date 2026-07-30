# Current Integrated Architecture

This document replaces the pre-release integration plan. The original merge
record is preserved in [`merge_legacy.md`](merge_legacy.md) for provenance, but
its proposed `src/` paths and verification claims are not current runtime
documentation.

## Canonical applications

| Application | Entry point | Maintained implementation |
|---|---|---|
| Facial landmarks | `run_beginner_level.ps1` | `beginner_level/src/` |
| Expression effects | `run_expert_level.ps1` | `expert_level/task2_realtime.py` |
| Just Dance scoring | `run_bonus_level.ps1` | `bonus_level/just_dance_app.py` |
| Pose-controlled Mario | `run_mario.ps1` | `bonus_level_mario/` |

`bonus_level/mario_demo/` is a frozen historical snapshot. It is not a second
maintained Mario implementation.

## Current perception and interaction paths

```text
Beginner
webcam -> Haar or YuNet face box -> LBF 68 landmarks -> temporal smoothing

Expert
webcam -> Haar face box -> LBF 68 landmarks -> eye-aligned keypoint features
       -> tuned geometry SVM -> temporal smoothing -> expression effect

Just Dance
reference pose cache + webcam YOLO pose
       -> primary-person heuristic -> normalized pose and motion
       -> reaction-lag search -> fixed-rate score events

Mario
webcam YOLO pose -> filtered upper-body gesture state -> platform-game controls
```

Haar remains the required Beginner baseline. YuNet is an alternative detector
that can be selected with:

```powershell
.\run_beginner_level.ps1 --detector yunet
```

## Evidence vocabulary

- **Selected-pose availability** is the fraction of processed frames in which
  the heuristic tracker returned a scoreable person.
- It is not primary-dancer identity accuracy because the videos do not have
  ground-truth identity annotations.
- Expression latency artifacts measure classifier prediction only. Live FPS
  remains the current end-to-end runtime indicator.
- Dance score, combo, and average mutate at a fixed 4 Hz game-time rate so
  additional inference frames do not create additional scoring opportunities.

## Verification

Run the camera-free suite from the repository root:

```powershell
.\check_merge.ps1
```

The suite covers 50 deterministic unit tests plus application preflights.
Webcam behaviour, GUI responsiveness, and end-to-end camera latency still
require a target-laptop playtest.

## Release process

Version tags matching `v*` trigger `.github/workflows/release.yml`. The
workflow publishes a GitHub Release using `RELEASE_NOTES.md`. Update
`CHANGELOG.md` and the release notes before pushing a new tag.
