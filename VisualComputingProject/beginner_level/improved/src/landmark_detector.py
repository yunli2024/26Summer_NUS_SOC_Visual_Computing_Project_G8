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
        self.previous_tracks = []

    def reset(self) -> None:
        self.previous_tracks = []

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
        next_tracks = []
        for idx, points in enumerate(current):
            face = tuple(int(v) for v in faces[idx])
            if len(points) != 68:
                smoothed.append(points)
                next_tracks.append((face, points))
                continue
            previous_points = self._match_previous(face)
            if previous_points is None:
                smoothed_points = points
            else:
                alpha = config.LANDMARK_SMOOTHING_ALPHA
                smoothed_points = alpha * points + (1.0 - alpha) * previous_points
            smoothed.append(smoothed_points)
            next_tracks.append((face, smoothed_points))
        self.previous_tracks = next_tracks
        return True, smoothed, "Landmarks detected"

    def _match_previous(self, face):
        best_iou = 0.0
        best_points = None
        for previous_face, previous_points in self.previous_tracks:
            overlap = _face_iou(previous_face, face)
            if overlap > best_iou:
                best_iou = overlap
                best_points = previous_points
        if best_iou < config.FACE_CHANGE_IOU_THRESHOLD:
            return None
        return best_points


def _face_iou(face_a, face_b) -> float:
    ax, ay, aw, ah = face_a
    bx, by, bw, bh = face_b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0
