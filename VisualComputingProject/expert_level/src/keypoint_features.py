from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from face_pipeline import Box, FaceDetector, LbfLandmarkEstimator, box_area, limit_boxes_by_area


EMOTION_CLASSES = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")
LEGACY_FEATURE_VERSION = "liyunzang_box_norm_geometry_v1"
ZHANGYX_FEATURE_VERSION = "lbf68_eye_aligned_v1"
UNIFIED_FEATURE_VERSION = "lbf68_eye_aligned_geometry_v2"
DEFAULT_FEATURE_VERSION = UNIFIED_FEATURE_VERSION


@dataclass(frozen=True)
class FaceFeatures:
    box: Box
    landmarks: np.ndarray
    vector: np.ndarray
    source: str
    feature_version: str = DEFAULT_FEATURE_VERSION


def expand_box(box: Box, image_shape: Tuple[int, int], ratio: float = 0.12) -> Box:
    height, width = image_shape[:2]
    x, y, w, h = box
    pad_x = int(w * ratio)
    pad_y = int(h * ratio)
    nx = max(0, x - pad_x)
    ny = max(0, y - pad_y)
    nw = min(width - nx, w + 2 * pad_x)
    nh = min(height - ny, h + 2 * pad_y)
    return nx, ny, nw, nh


def centered_face_box(frame_bgr: np.ndarray, margin: float = 0.08) -> Box:
    height, width = frame_bgr.shape[:2]
    side = int(min(width, height) * (1.0 - margin))
    x = max(0, (width - side) // 2)
    y = max(0, (height - side) // 2)
    return x, y, side, side


def normalize_landmarks_by_box(landmarks: np.ndarray, box: Box) -> np.ndarray:
    """Reproduce the feature normalization used by the retained Liyunzang SVM."""
    x, y, w, h = box
    center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)
    scale = max(float(w), float(h), 1.0)
    normalized = (landmarks.astype(np.float32) - center) / scale
    return normalized.reshape(-1)


def align_landmarks_by_eyes(landmarks: np.ndarray) -> np.ndarray:
    """Remove translation, in-plane rotation, and scale using the eye centers."""
    points = np.asarray(landmarks, dtype=np.float32).reshape(-1, 2)
    if points.shape != (68, 2) or not np.isfinite(points).all():
        raise ValueError("Expected 68 finite two-dimensional facial landmarks.")
    left_eye = points[36:42].mean(axis=0)
    right_eye = points[42:48].mean(axis=0)
    eye_vector = right_eye - left_eye
    eye_distance = float(np.linalg.norm(eye_vector))
    if eye_distance < 1e-6:
        raise ValueError("Inter-eye distance is too small for normalization.")
    angle = -float(np.arctan2(eye_vector[1], eye_vector[0]))
    cosine, sine = np.cos(angle), np.sin(angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
    eye_center = 0.5 * (left_eye + right_eye)
    return ((points - eye_center) @ rotation.T / eye_distance).astype(np.float32)


def euclidean(points: np.ndarray, first: int, second: int) -> float:
    return float(np.linalg.norm(points[first] - points[second]))


def legacy_geometric_features(landmarks: np.ndarray, box: Box) -> np.ndarray:
    scale = max(float(box[2]), float(box[3]), 1.0)
    pairs = [
        (36, 39),  # left eye width
        (37, 41),
        (38, 40),
        (42, 45),  # right eye width
        (43, 47),
        (44, 46),
        (48, 54),  # mouth width
        (51, 57),
        (62, 66),
        (21, 39),  # brow to eye
        (22, 42),
        (31, 35),  # nose width
        (30, 8),   # nose to chin
    ]
    distances = [euclidean(landmarks, a, b) / scale for a, b in pairs]

    mouth_width = max(euclidean(landmarks, 48, 54), 1.0)
    mouth_open = euclidean(landmarks, 62, 66) / mouth_width
    left_eye_open = (euclidean(landmarks, 37, 41) + euclidean(landmarks, 38, 40)) / max(
        2.0 * euclidean(landmarks, 36, 39), 1.0
    )
    right_eye_open = (euclidean(landmarks, 43, 47) + euclidean(landmarks, 44, 46)) / max(
        2.0 * euclidean(landmarks, 42, 45), 1.0
    )
    brow_gap = (euclidean(landmarks, 21, 39) + euclidean(landmarks, 22, 42)) / (2.0 * scale)

    return np.asarray(distances + [mouth_open, left_eye_open, right_eye_open, brow_gap], dtype=np.float32)


def unified_geometric_features(aligned: np.ndarray) -> np.ndarray:
    """Describe expression-sensitive geometry using only aligned keypoints."""
    pairs = [
        (36, 39), (37, 41), (38, 40),
        (42, 45), (43, 47), (44, 46),
        (48, 54), (51, 57), (62, 66),
        (19, 37), (24, 44), (21, 39), (22, 42),
        (31, 35), (30, 8), (48, 36), (54, 45),
    ]
    distances = [euclidean(aligned, first, second) for first, second in pairs]
    mouth_width = max(euclidean(aligned, 48, 54), 1e-6)
    left_eye_width = max(euclidean(aligned, 36, 39), 1e-6)
    right_eye_width = max(euclidean(aligned, 42, 45), 1e-6)
    ratios = [
        euclidean(aligned, 62, 66) / mouth_width,
        euclidean(aligned, 51, 57) / mouth_width,
        (euclidean(aligned, 37, 41) + euclidean(aligned, 38, 40)) / (2.0 * left_eye_width),
        (euclidean(aligned, 43, 47) + euclidean(aligned, 44, 46)) / (2.0 * right_eye_width),
        euclidean(aligned, 21, 39),
        euclidean(aligned, 22, 42),
    ]
    slopes = [
        float(np.arctan2(*(aligned[21] - aligned[17])[[1, 0]])),
        float(np.arctan2(*(aligned[26] - aligned[22])[[1, 0]])),
        float(np.arctan2(*(aligned[54] - aligned[48])[[1, 0]])),
    ]
    return np.asarray(distances + ratios + slopes, dtype=np.float32)


def build_feature_vector(
    landmarks: np.ndarray,
    box: Box,
    feature_version: str = DEFAULT_FEATURE_VERSION,
) -> np.ndarray:
    """Build a keypoint-only vector; no image pixels enter either representation."""
    if feature_version == LEGACY_FEATURE_VERSION:
        coords = normalize_landmarks_by_box(landmarks, box)
        geom = legacy_geometric_features(landmarks, box)
        return np.concatenate([coords, geom]).astype(np.float32)
    if feature_version == ZHANGYX_FEATURE_VERSION:
        # The retained Zhangyx Geometry-SVM appends its 38 geometry values
        # inside the saved sklearn Pipeline.  The runtime must therefore pass
        # exactly the same 136 eye-aligned coordinates used during training.
        return align_landmarks_by_eyes(landmarks).reshape(-1).astype(np.float32)
    if feature_version == UNIFIED_FEATURE_VERSION:
        aligned = align_landmarks_by_eyes(landmarks)
        geom = unified_geometric_features(aligned)
        return np.concatenate([aligned.reshape(-1), geom]).astype(np.float32)
    raise ValueError(f"Unsupported facial feature version: {feature_version}")


def feature_builder(feature_version: str) -> Callable[[np.ndarray, Box], np.ndarray]:
    return lambda landmarks, box: build_feature_vector(landmarks, box, feature_version)


class ExpertFeatureExtractor:
    def __init__(
        self,
        face_detector: FaceDetector,
        landmark_estimator: LbfLandmarkEstimator,
        *,
        max_faces: int = 1,
        use_center_fallback: bool = True,
        feature_version: str = DEFAULT_FEATURE_VERSION,
    ) -> None:
        self._face_detector = face_detector
        self._landmark_estimator = landmark_estimator
        self._max_faces = max_faces
        self._use_center_fallback = use_center_fallback
        self._feature_version = feature_version

    def extract(self, frame_bgr: np.ndarray) -> List[FaceFeatures]:
        boxes = self._face_detector.detect(frame_bgr)
        boxes = [expand_box(box, frame_bgr.shape) for box in boxes]
        boxes = limit_boxes_by_area(boxes, self._max_faces)
        source = self._face_detector.name

        if not boxes and self._use_center_fallback:
            boxes = [centered_face_box(frame_bgr)]
            source = "center_fallback"

        landmarks_list = self._landmark_estimator.fit(frame_bgr, boxes)
        features: List[FaceFeatures] = []
        for box, landmarks in zip(boxes, landmarks_list):
            if landmarks is None or len(landmarks) != 68:
                continue
            features.append(
                FaceFeatures(
                    box=box,
                    landmarks=landmarks,
                    vector=build_feature_vector(landmarks, box, self._feature_version),
                    source=source,
                    feature_version=self._feature_version,
                )
            )
        return features

    def extract_primary(self, frame_bgr: np.ndarray) -> Optional[FaceFeatures]:
        faces = self.extract(frame_bgr)
        if not faces:
            return None
        return max(faces, key=lambda face: box_area(face.box))


def prepare_fer_image(image_bgr: np.ndarray, output_size: int = 192) -> np.ndarray:
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
    height, width = image_bgr.shape[:2]
    if max(height, width) >= output_size:
        return image_bgr
    scale = output_size / max(height, width)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image_bgr, new_size, interpolation=cv2.INTER_CUBIC)
