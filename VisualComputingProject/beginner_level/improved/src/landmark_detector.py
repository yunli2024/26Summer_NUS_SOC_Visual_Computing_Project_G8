"""Improved LBF landmark detector with exponential smoothing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from . import config
except ImportError:
    import config


class SmoothedLandmarkDetector:
    """Detect 68 landmarks and smooth them when the target is stable."""

    def __init__(self, model_path: Path = config.LBF_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"LBF model file not found: {self.model_path}")
        if not hasattr(cv2, "face"):
            raise RuntimeError("cv2.face is not available. Install opencv-contrib-python.")
        self.facemark = cv2.face.createFacemarkLBF()
        try:
            self.facemark.loadModel(str(self.model_path))
        except cv2.error as exc:
            raise RuntimeError(f"Failed to load LBF model: {self.model_path}") from exc
        self.previous_landmarks = None
        self.previous_face = None

    def reset(self) -> None:
        self.previous_landmarks = None
        self.previous_face = None

    def fit(self, gray_frame, faces, detection_status: str):
        if detection_status == "CACHED":
            return False, [], "Cached face box only; landmarks not updated"
        if detection_status == "LOST":
            self.reset()
            return False, [], "Face lost; smoothing reset"
        if len(faces) == 0:
            self.reset()
            return False, [], "No face detected"
        try:
            ok, landmarks = self.facemark.fit(gray_frame, faces)
        except cv2.error as exc:
            return False, [], f"Landmark fitting failed: {exc}"
        if not ok or landmarks is None or len(landmarks) == 0:
            return False, [], "Landmark fitting failed"

        current = [np.asarray(face_points, dtype=np.float32).reshape(-1, 2) for face_points in landmarks]
        smoothed = []
        for idx, points in enumerate(current):
            face = tuple(int(v) for v in faces[idx])
            if len(points) != 68:
                smoothed.append(points)
                continue
            if self._should_reset(points, face):
                smoothed_points = points
            else:
                alpha = config.LANDMARK_SMOOTHING_ALPHA
                smoothed_points = alpha * points + (1.0 - alpha) * self.previous_landmarks
            smoothed.append(smoothed_points)
            self.previous_landmarks = smoothed_points
            self.previous_face = face
        return True, smoothed, "Landmarks detected"

    def _should_reset(self, points, face) -> bool:
        if self.previous_landmarks is None or self.previous_face is None:
            return True
        if len(self.previous_landmarks) != len(points):
            return True
        return _face_iou(self.previous_face, face) < config.FACE_CHANGE_IOU_THRESHOLD


def _face_iou(face_a, face_b) -> float:
    ax, ay, aw, ah = face_a
    bx, by, bw, bh = face_b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
