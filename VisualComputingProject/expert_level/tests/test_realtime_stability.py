from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from keypoint_features import FaceFeatures, build_feature_vector
from face_pipeline import YuNetFaceDetector
from realtime_stability import FaceStabilizer, MultiFaceStabilizer, ProbabilitySmoother, plausible_face


def make_landmarks(x: int = 100, y: int = 80, w: int = 160, h: int = 190) -> np.ndarray:
    points = np.zeros((68, 2), dtype=np.float32)
    for idx in range(68):
        rx = 0.18 + 0.64 * ((idx % 17) / 16.0)
        ry = 0.18 + 0.70 * ((idx // 17) / 3.0)
        points[idx] = [x + rx * w, y + ry * h]
    points[36:42, 0] = np.linspace(x + 0.28 * w, x + 0.43 * w, 6)
    points[42:48, 0] = np.linspace(x + 0.57 * w, x + 0.72 * w, 6)
    points[36:42, 1] = y + 0.35 * h
    points[42:48, 1] = y + 0.35 * h
    points[30] = [x + 0.50 * w, y + 0.53 * h]
    points[48:68, 0] = np.linspace(x + 0.34 * w, x + 0.66 * w, 20)
    points[48:68, 1] = y + 0.68 * h
    points[8] = [x + 0.50 * w, y + 0.94 * h]
    return points


def make_face(x: int = 100, y: int = 80, w: int = 160, h: int = 190) -> FaceFeatures:
    landmarks = make_landmarks(x, y, w, h)
    box = (x, y, w, h)
    return FaceFeatures(box=box, landmarks=landmarks, vector=build_feature_vector(landmarks, box), source="test")


def test_face_stabilizer() -> None:
    stabilizer = FaceStabilizer(stable_frames=2, hold_frames=2, min_area_ratio=0.01)
    frame_shape = (540, 960, 3)
    face = make_face()
    assert stabilizer.update([face], frame_shape) == []
    assert len(stabilizer.update([face], frame_shape)) == 1
    assert len(stabilizer.update([], frame_shape)) == 1
    assert len(stabilizer.update([], frame_shape)) == 1
    assert stabilizer.update([], frame_shape) == []


def test_multi_face_stabilizer() -> None:
    stabilizer = MultiFaceStabilizer(max_faces=4, stable_frames=2, hold_frames=2, min_area_ratio=0.006)
    frame_shape = (540, 960, 3)
    faces = [make_face(100, 80), make_face(420, 90)]
    assert stabilizer.update(faces, frame_shape) == []
    tracked = stabilizer.update(faces, frame_shape)
    assert len(tracked) == 2
    assert len({item.track_id for item in tracked}) == 2

    tiny_false_positive = make_face(10, 10, 20, 24)
    tracked = stabilizer.update([*faces, tiny_false_positive], frame_shape)
    assert len(tracked) == 2


def test_landmark_jitter_is_smoothed_per_track() -> None:
    stabilizer = MultiFaceStabilizer(
        max_faces=4,
        stable_frames=1,
        landmark_smoothing=0.75,
        min_area_ratio=0.006,
    )
    frame_shape = (540, 960, 3)
    first = make_face()
    tracked = stabilizer.update([first], frame_shape)
    assert len(tracked) == 1

    jittered_landmarks = first.landmarks + np.array([8.0, -4.0], dtype=np.float32)
    jittered = FaceFeatures(
        box=first.box,
        landmarks=jittered_landmarks,
        vector=build_feature_vector(jittered_landmarks, first.box),
        source=first.source,
    )
    tracked = stabilizer.update([jittered], frame_shape)
    expected = first.landmarks + np.array([2.0, -1.0], dtype=np.float32)
    assert np.allclose(tracked[0].face.landmarks, expected)
    # The unified eye-aligned representation intentionally removes global
    # translation, so a whole-face shift does not alter classification input.
    assert np.allclose(tracked[0].face.vector, jittered.vector)


def test_yunet_face_uses_relaxed_landmark_validation() -> None:
    face = make_face()
    distorted = face.landmarks.copy()
    distorted[30, 1] = distorted[36, 1] - 4
    yunet_face = FaceFeatures(
        box=face.box,
        landmarks=distorted,
        vector=face.vector,
        source="yunet",
    )
    assert plausible_face(yunet_face, (540, 960, 3), min_area_ratio=0.006)


def test_yunet_detector_rejects_room_like_background() -> None:
    model_path = (
        Path(__file__).resolve().parents[2]
        / "resources"
        / "face_models"
        / "face_detection_yunet_2023mar.onnx"
    )
    detector = YuNetFaceDetector(model_path)
    background = np.full((540, 960, 3), 215, dtype=np.uint8)
    background[170:182, :] = 90
    background[:, 710:726] = 115
    background[80:95, 540:760] = 145
    assert detector.detect(background) == []


def test_probability_smoother() -> None:
    classes = ["angry", "happy", "neutral"]
    smoother = ProbabilitySmoother(
        alpha=0.0,
        min_confidence=0.40,
        switch_margin=0.10,
        initial_frames=3,
        switch_frames=4,
        min_hold_frames=5,
    )
    happy = np.array([0.05, 0.85, 0.10])
    angry = np.array([0.85, 0.05, 0.10])
    neutral = np.array([0.05, 0.10, 0.85])

    for _ in range(2):
        label, _, _ = smoother.update(happy, classes)
        assert label == "uncertain"
    label, confidence, _ = smoother.update(happy, classes)
    assert label == "happy"
    assert confidence > 0.80

    for probabilities in [angry, neutral] * 6:
        label, _, _ = smoother.update(probabilities, classes)
        assert label == "happy"

    label, _, _ = smoother.update(happy, classes)
    assert label == "happy"
    for _ in range(3):
        label, _, _ = smoother.update(neutral, classes)
        assert label == "happy"
    label, confidence, _ = smoother.update(neutral, classes)
    assert label == "neutral"
    assert confidence > 0.80


def main() -> int:
    test_face_stabilizer()
    test_multi_face_stabilizer()
    test_landmark_jitter_is_smoothed_per_track()
    test_yunet_face_uses_relaxed_landmark_validation()
    test_yunet_detector_rejects_room_like_background()
    test_probability_smoother()
    print("realtime_stability_tests_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
