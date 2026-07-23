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
GEOMETRY_FEATURE_NAMES = (
    "inner_brow_distance",
    "left_brow_eye_gap",
    "right_brow_eye_gap",
    "mean_brow_eye_gap",
    "brow_eye_gap_asymmetry",
    "left_brow_slope",
    "right_brow_slope",
    "brow_slope_asymmetry",
    "left_eye_width",
    "right_eye_width",
    "left_eye_height",
    "right_eye_height",
    "left_eye_aspect_ratio",
    "right_eye_aspect_ratio",
    "mean_eye_aspect_ratio",
    "eye_aspect_asymmetry",
    "left_eye_area_ratio",
    "right_eye_area_ratio",
    "nose_width",
    "nose_to_upper_lip",
    "nose_to_mouth_center",
    "mouth_width",
    "outer_mouth_height",
    "inner_mouth_height",
    "outer_mouth_aspect_ratio",
    "inner_mouth_aspect_ratio",
    "mouth_corner_angle",
    "smile_curvature",
    "left_corner_lift",
    "right_corner_lift",
    "mouth_corner_asymmetry",
    "upper_lip_thickness",
    "lower_lip_thickness",
    "lip_thickness_ratio",
    "mouth_to_chin_ratio",
    "mouth_center_x",
    "eye_to_mouth_distance",
    "cheek_distance_asymmetry",
)
GEOMETRY_FEATURE_GROUPS = {
    "brow": GEOMETRY_FEATURE_NAMES[0:8],
    "eyes": GEOMETRY_FEATURE_NAMES[8:18],
    "nose": GEOMETRY_FEATURE_NAMES[18:21],
    "mouth": GEOMETRY_FEATURE_NAMES[21:34],
    "global": GEOMETRY_FEATURE_NAMES[34:38],
}


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


def landmark_geometry_features(features: np.ndarray) -> np.ndarray:
    """Derive expression-focused distances, ratios, angles, and asymmetries.

    The input must contain eye-aligned, inter-eye-normalized LBF coordinates.
    A batch produces ``(n, len(GEOMETRY_FEATURE_NAMES))`` and a single feature
    vector produces one row with the same two-dimensional shape.
    """
    rows = np.asarray(features, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.ndim != 2 or rows.shape[1] != 136:
        raise ValueError("Expected normalized landmark features with shape (n, 136).")
    if not np.isfinite(rows).all():
        raise ValueError("Landmark features must be finite.")

    points = rows.reshape(-1, 68, 2)
    epsilon = np.float32(1e-6)

    def distance(first: int, second: int) -> np.ndarray:
        return np.linalg.norm(points[:, first] - points[:, second], axis=1)

    def ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
        return numerator / np.maximum(denominator, epsilon)

    def polygon_area(indices: tuple[int, ...]) -> np.ndarray:
        polygon = points[:, indices]
        x = polygon[:, :, 0]
        y = polygon[:, :, 1]
        return 0.5 * np.abs(
            np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)
        )

    left_eye_center = points[:, 36:42].mean(axis=1)
    right_eye_center = points[:, 42:48].mean(axis=1)
    eye_center = 0.5 * (left_eye_center + right_eye_center)
    left_brow_center = points[:, 17:22].mean(axis=1)
    right_brow_center = points[:, 22:27].mean(axis=1)

    left_brow_gap = left_eye_center[:, 1] - left_brow_center[:, 1]
    right_brow_gap = right_eye_center[:, 1] - right_brow_center[:, 1]
    left_brow_vector = points[:, 21] - points[:, 17]
    right_brow_vector = points[:, 26] - points[:, 22]
    left_brow_slope = np.arctan2(left_brow_vector[:, 1], left_brow_vector[:, 0])
    right_brow_slope = np.arctan2(right_brow_vector[:, 1], right_brow_vector[:, 0])

    left_eye_width = distance(36, 39)
    right_eye_width = distance(42, 45)
    left_eye_height_sum = distance(37, 41) + distance(38, 40)
    right_eye_height_sum = distance(43, 47) + distance(44, 46)
    left_eye_height = 0.5 * left_eye_height_sum
    right_eye_height = 0.5 * right_eye_height_sum
    left_eye_aspect = ratio(left_eye_height_sum, 2.0 * left_eye_width)
    right_eye_aspect = ratio(right_eye_height_sum, 2.0 * right_eye_width)

    mouth_center = points[:, 48:60].mean(axis=1)
    mouth_width = distance(48, 54)
    outer_mouth_height = (
        distance(50, 58) + distance(51, 57) + distance(52, 56)
    ) / 3.0
    inner_mouth_height = (
        distance(61, 67) + distance(62, 66) + distance(63, 65)
    ) / 3.0
    corner_mean_y = 0.5 * (points[:, 48, 1] + points[:, 54, 1])
    left_corner_lift = ratio(mouth_center[:, 1] - points[:, 48, 1], mouth_width)
    right_corner_lift = ratio(mouth_center[:, 1] - points[:, 54, 1], mouth_width)
    mouth_corner_vector = points[:, 54] - points[:, 48]
    mouth_corner_angle = np.arctan2(
        mouth_corner_vector[:, 1],
        mouth_corner_vector[:, 0],
    )
    upper_lip_thickness = distance(51, 62)
    lower_lip_thickness = distance(57, 66)
    face_height = distance(27, 8)

    geometry = np.column_stack(
        (
            distance(21, 22),
            left_brow_gap,
            right_brow_gap,
            0.5 * (left_brow_gap + right_brow_gap),
            np.abs(left_brow_gap - right_brow_gap),
            left_brow_slope,
            right_brow_slope,
            np.abs(left_brow_slope + right_brow_slope),
            left_eye_width,
            right_eye_width,
            left_eye_height,
            right_eye_height,
            left_eye_aspect,
            right_eye_aspect,
            0.5 * (left_eye_aspect + right_eye_aspect),
            np.abs(left_eye_aspect - right_eye_aspect),
            ratio(polygon_area((36, 37, 38, 39, 40, 41)), left_eye_width**2),
            ratio(polygon_area((42, 43, 44, 45, 46, 47)), right_eye_width**2),
            distance(31, 35),
            distance(33, 51),
            np.linalg.norm(points[:, 33] - mouth_center, axis=1),
            mouth_width,
            outer_mouth_height,
            inner_mouth_height,
            ratio(outer_mouth_height, mouth_width),
            ratio(inner_mouth_height, mouth_width),
            mouth_corner_angle,
            ratio(mouth_center[:, 1] - corner_mean_y, mouth_width),
            left_corner_lift,
            right_corner_lift,
            ratio(points[:, 48, 1] - points[:, 54, 1], mouth_width),
            upper_lip_thickness,
            lower_lip_thickness,
            ratio(upper_lip_thickness, lower_lip_thickness),
            ratio(distance(57, 8), face_height),
            mouth_center[:, 0],
            np.linalg.norm(mouth_center - eye_center, axis=1),
            ratio(
                np.abs(distance(36, 48) - distance(45, 54)),
                distance(36, 45),
            ),
        )
    ).astype(np.float32)
    if geometry.shape[1] != len(GEOMETRY_FEATURE_NAMES):
        raise RuntimeError("Geometry feature names and values are out of sync.")
    if not np.isfinite(geometry).all():
        raise ValueError("Geometry feature extraction produced invalid values.")
    return geometry


def append_landmark_geometry(features: np.ndarray) -> np.ndarray:
    """Append explicit geometry descriptors to the 136 normalized coordinates."""
    rows = np.asarray(features, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    geometry = landmark_geometry_features(rows)
    return np.concatenate((rows, geometry), axis=1).astype(np.float32)


def append_landmark_geometry_groups(
    features: np.ndarray,
    groups: tuple[str, ...],
) -> np.ndarray:
    """Append only the requested named groups of landmark geometry features."""
    rows = np.asarray(features, dtype=np.float32)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    unknown = sorted(set(groups).difference(GEOMETRY_FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"Unknown geometry feature groups: {', '.join(unknown)}")
    if len(set(groups)) != len(groups):
        raise ValueError("Geometry feature groups must not be repeated.")
    if not groups:
        return rows.copy()

    geometry = landmark_geometry_features(rows)
    name_to_index = {
        name: index for index, name in enumerate(GEOMETRY_FEATURE_NAMES)
    }
    selected_names = [
        name
        for group in groups
        for name in GEOMETRY_FEATURE_GROUPS[group]
    ]
    selected_indices = [name_to_index[name] for name in selected_names]
    return np.concatenate((rows, geometry[:, selected_indices]), axis=1).astype(
        np.float32
    )


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
