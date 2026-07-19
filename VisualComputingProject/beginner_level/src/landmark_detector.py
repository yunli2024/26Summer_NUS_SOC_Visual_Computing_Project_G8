"""LBF 68-point facial landmark detection helpers."""

from __future__ import annotations

from pathlib import Path

import cv2

try:
    from . import config
except ImportError:
    import config


class LandmarkDetector:
    """Load the OpenCV LBF model and fit 68 facial landmarks."""

    def __init__(self, model_path: Path = config.LBF_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"LBF model file not found: {self.model_path}")
        if not hasattr(cv2, "face"):
            raise RuntimeError("cv2.face is not available. Install opencv-contrib-python.")

        try:
            self.facemark = cv2.face.createFacemarkLBF()
            self.facemark.loadModel(str(self.model_path))
        except cv2.error as exc:
            raise RuntimeError(f"Failed to load LBF model: {self.model_path}") from exc

    def fit(self, gray_frame, faces):
        """Fit landmarks for detected faces.

        Returns:
            tuple: (ok, landmarks, message)
        """
        if len(faces) == 0:
            return False, [], "No face detected"

        try:
            ok, landmarks = self.facemark.fit(gray_frame, faces)
        except cv2.error as exc:
            return False, [], f"Landmark fitting failed: {exc}"

        if not ok:
            return False, [], "Landmark fitting failed"
        return True, landmarks, "Landmarks detected"
