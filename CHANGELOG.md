# Changelog

All notable portfolio releases are recorded here. The project follows
Semantic Versioning for recruiter-facing snapshots.

## [1.1.0-rc.2] - 2026-07-30

### Added

- Expert webcam stage telemetry with mean, p50, and p95 latency plus
  reproducibility metadata.
- Profile-aware integrity checks for tracked, demo, and training assets.
- Four unit tests for primary-dancer selection, continuity, smoothing, and
  reset behaviour.
- Optional Ultralytics `--device` selection in the dance and Mario live apps.
- `-Locked` runtime installation from the verified dependency snapshot.

### Changed

- Reframed softmax-normalized SVM margins as relative class scores rather than
  calibrated confidence probabilities.
- Expanded the portable local/CI regression suite from 50 to 54 tests.

## [1.1.0-rc.1] - 2026-07-30

### Added

- Selectable YuNet face detection for the Beginner webcam application while
  retaining Haar as the required course baseline.
- Fixed-rate 4 Hz dance score clock and regression tests that make points,
  combo, and average independent of extra inference frames.
- Tag-triggered GitHub Release workflow and release notes.
- Current architecture document with explicit evidence vocabulary.

### Changed

- Renamed the pose benchmark metric from “primary-dancer detection” to
  “selected-pose availability”.
- Synchronized Bonus reports with the tracked 2,680-frame pose cache and
  machine-readable benchmark artifacts.
- Updated the documented portable test count from 47 to 50.

### Fixed

- Corrected the single-dancer Task 1 YOLO inference figure from the stale
  report value to the tracked 170.55 ms result.
- Prevented high-inference-rate machines from accumulating more dance points
  merely by producing more camera results.

## [1.0.0] - 2026-07-29

- Initial recruiter-facing portfolio baseline with four local interactive
  keypoint applications, persisted evaluation artifacts, and CI.
