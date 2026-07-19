"""Haar cascade face detection helpers."""

from __future__ import annotations

from pathlib import Path

import cv2

try:
    from . import config
except ImportError:
    import config


class FaceDetector:
    """Load a Haar cascade and detect faces in grayscale frames."""

    def __init__(self, cascade_path: Path = config.HAAR_CASCADE_PATH) -> None:
        self.cascade_path = Path(cascade_path)
        if not self.cascade_path.exists():
            raise FileNotFoundError(f"Haar cascade file not found: {self.cascade_path}")

        self.classifier = cv2.CascadeClassifier(str(self.cascade_path))
        if self.classifier.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {self.cascade_path}")

    def detect(self, gray_frame):
        """Return face rectangles as (x, y, w, h)."""
        return self.classifier.detectMultiScale(
            gray_frame,
            scaleFactor=config.FACE_SCALE_FACTOR,
            minNeighbors=config.FACE_MIN_NEIGHBORS,
            minSize=config.FACE_MIN_SIZE,
        )
