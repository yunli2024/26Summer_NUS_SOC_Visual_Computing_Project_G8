from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from face_pipeline import Box, FaceDetector, LbfLandmarkEstimator, box_area, limit_boxes_by_area


EMOTION_CLASSES = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")


@dataclass(frozen=True)
class FaceFeatures:
    box: Box
    landmarks: np.ndarray
    vector: np.ndarray
    source: str


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


def normalize_landmarks(landmarks: np.ndarray, box: Box) -> np.ndarray:
    x, y, w, h = box
    center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float32)
    scale = max(float(w), float(h), 1.0)
    normalized = (landmarks.astype(np.float32) - center) / scale
    return normalized.reshape(-1)


def euclidean(points: np.ndarray, first: int, second: int) -> float:
    return float(np.linalg.norm(points[first] - points[second]))


def geometric_features(landmarks: np.ndarray, box: Box) -> np.ndarray:
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


def build_feature_vector(landmarks: np.ndarray, box: Box) -> np.ndarray:
    coords = normalize_landmarks(landmarks, box)
    geom = geometric_features(landmarks, box)
    return np.concatenate([coords, geom]).astype(np.float32)


class ExpertFeatureExtractor:
    def __init__(
        self,
        face_detector: FaceDetector,
        landmark_estimator: LbfLandmarkEstimator,
        *,
        max_faces: int = 1,
        use_center_fallback: bool = True,
    ) -> None:
        self._face_detector = face_detector
        self._landmark_estimator = landmark_estimator
        self._max_faces = max_faces
        self._use_center_fallback = use_center_fallback

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
                    vector=build_feature_vector(landmarks, box),
                    source=source,
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
