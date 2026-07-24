from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from keypoint_features import (  # noqa: E402
    LEGACY_FEATURE_VERSION,
    UNIFIED_FEATURE_VERSION,
    align_landmarks_by_eyes,
    build_feature_vector,
)


def synthetic_landmarks() -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, 68, endpoint=False)
    points = np.column_stack([120.0 + 45.0 * np.cos(theta), 100.0 + 60.0 * np.sin(theta)])
    points[36:42] = np.column_stack([np.linspace(88.0, 105.0, 6), np.full(6, 82.0)])
    points[42:48] = np.column_stack([np.linspace(135.0, 152.0, 6), np.full(6, 82.0)])
    points[48] = (102.0, 128.0)
    points[54] = (140.0, 128.0)
    points[51] = (121.0, 123.0)
    points[57] = (121.0, 138.0)
    points[62] = (121.0, 128.0)
    points[66] = (121.0, 134.0)
    return points.astype(np.float32)


def test_unified_features_are_keypoint_only_and_finite() -> None:
    points = synthetic_landmarks()
    vector = build_feature_vector(points, (70, 35, 100, 140), UNIFIED_FEATURE_VERSION)
    assert vector.ndim == 1
    assert vector.shape[0] > 136
    assert np.isfinite(vector).all()


def test_eye_alignment_removes_translation_and_scale() -> None:
    points = synthetic_landmarks()
    transformed = points * 2.4 + np.asarray([73.0, -41.0], dtype=np.float32)
    assert np.allclose(
        align_landmarks_by_eyes(points),
        align_landmarks_by_eyes(transformed),
        atol=1e-5,
    )


def test_legacy_vector_remains_compatible_with_retained_svm() -> None:
    points = synthetic_landmarks()
    vector = build_feature_vector(points, (70, 35, 100, 140), LEGACY_FEATURE_VERSION)
    assert vector.shape == (153,)


def main() -> int:
    test_unified_features_are_keypoint_only_and_finite()
    test_eye_alignment_removes_translation_and_scale()
    test_legacy_vector_remains_compatible_with_retained_svm()
    print("keypoint_feature_tests_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
