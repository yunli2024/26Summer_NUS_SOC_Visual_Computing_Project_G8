"""Shared facial-landmark feature extraction for Part Two."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np


PART_TWO_DIR = Path(__file__).resolve().parent
PART_ONE_DIR = PART_TWO_DIR.parent / "part one"
if str(PART_ONE_DIR) not in sys.path:
    sys.path.insert(0, str(PART_ONE_DIR))

from starter import FaceLandmarkDetector  # noqa: E402


CLASS_NAMES = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
FEATURE_VERSION = "lbf68_eye_aligned_v1"


def read_grayscale(path: Path) -> np.ndarray:
    """Read an image through NumPy so OpenCV supports Unicode Windows paths."""
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Align the eyes, remove translation/scale, and return 136 float features."""
    points = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    if points.shape != (68, 2) or not np.isfinite(points).all():
        raise ValueError("Expected 68 finite two-dimensional landmarks.")

    left_eye = points[36:42].mean(axis=0)
    right_eye = points[42:48].mean(axis=0)
    eye_vector = right_eye - left_eye
    eye_distance = float(np.linalg.norm(eye_vector))
    if eye_distance < 1e-6:
        raise ValueError("Eye distance is too small to normalize landmarks.")

    angle = -float(np.arctan2(eye_vector[1], eye_vector[0]))
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    eye_center = (left_eye + right_eye) / 2.0
    aligned = (points - eye_center) @ rotation.T
    aligned /= eye_distance
    return aligned.reshape(-1).astype(np.float32)


class ExpressionFeatureExtractor:
    """Extract normalized LBF features from centered FER images or Haar faces."""

    def __init__(self, image_size: int = 192, center_inset: float = 0.08) -> None:
        if image_size < 96:
            raise ValueError("image_size must be at least 96 pixels.")
        if not 0.0 <= center_inset < 0.4:
            raise ValueError("center_inset must be in [0, 0.4).")
        self.image_size = image_size
        self.center_inset = center_inset
        self.detector = FaceLandmarkDetector(
            PART_ONE_DIR / "haarcascade_frontalface_default.xml",
            PART_ONE_DIR / "lbfmodel.yaml",
        )

    def extract_path(
        self,
        image_path: Path,
        face_mode: str = "center",
    ) -> tuple[np.ndarray, np.ndarray, str]:
        image = read_grayscale(image_path)
        return self.extract_image(image, face_mode)

    def extract_image(
        self,
        gray: np.ndarray,
        face_mode: str = "center",
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Return normalized features, resized-image landmarks, and method used."""
        if face_mode not in {"center", "haar", "haar-fallback"}:
            raise ValueError(f"Unknown face mode: {face_mode}")

        resized = cv2.resize(
            gray,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_CUBIC,
        )

        if face_mode in {"haar", "haar-fallback"}:
            bgr = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
            faces, all_landmarks = self.detector.detect(bgr, use_clahe=True)
            if all_landmarks:
                areas = faces[:, 2].astype(np.int64) * faces[:, 3].astype(np.int64)
                best = int(np.argmax(areas))
                points = np.asarray(all_landmarks[best], dtype=np.float32).reshape(68, 2)
                return normalize_landmarks(points), points, "haar"
            if face_mode == "haar":
                raise ValueError("Haar/LBF could not extract landmarks.")

        inset = int(round(self.image_size * self.center_inset))
        side = self.image_size - 2 * inset
        face = np.array([[inset, inset, side, side]], dtype=np.int32)
        success, raw_landmarks = self.detector.facemark.fit(resized, face)
        if not success or len(raw_landmarks) == 0:
            raise ValueError("LBF failed on the centered FER face region.")
        points = np.asarray(raw_landmarks[0], dtype=np.float32).reshape(68, 2)
        return normalize_landmarks(points), points, "center"
