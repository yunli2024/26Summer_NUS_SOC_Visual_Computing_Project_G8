from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np


Box = Tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectionResult:
    box: Box
    landmarks: Optional[np.ndarray]


class FpsMeter:
    def __init__(self, smoothing: float = 0.9) -> None:
        self._smoothing = smoothing
        self._last_time = time.perf_counter()
        self.fps = 0.0

    def tick(self) -> float:
        now = time.perf_counter()
        dt = max(now - self._last_time, 1e-9)
        instant_fps = 1.0 / dt
        self.fps = (
            instant_fps if self.fps == 0.0 else self._smoothing * self.fps + (1.0 - self._smoothing) * instant_fps
        )
        self._last_time = now
        return self.fps


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def preprocess_gray(gray: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return gray
    if mode == "equalize":
        return cv2.equalizeHist(gray)
    if mode == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    raise ValueError(f"Unsupported preprocess mode: {mode}")


class HaarFaceDetector:
    def __init__(
        self,
        cascade_path: Path,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_face_size: int = 60,
        preprocess: str = "clahe",
        overlap_threshold: float = 0.55,
    ) -> None:
        require_file(cascade_path, "Haar cascade")
        self._classifier = cv2.CascadeClassifier(str(cascade_path))
        if self._classifier.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")
        self._scale_factor = scale_factor
        self._min_neighbors = min_neighbors
        self._min_face_size = min_face_size
        self._preprocess = preprocess
        self._overlap_threshold = overlap_threshold

    def detect(self, frame_bgr: np.ndarray) -> List[Box]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = preprocess_gray(gray, self._preprocess)
        boxes = self._classifier.detectMultiScale(
            gray,
            scaleFactor=self._scale_factor,
            minNeighbors=self._min_neighbors,
            minSize=(self._min_face_size, self._min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return suppress_overlapping_boxes([tuple(map(int, box)) for box in boxes], self._overlap_threshold)


class MediaPipeFaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "MediaPipe is optional. Install it with: python -m pip install mediapipe"
            ) from exc

        self._mp_face_detection = mp.solutions.face_detection
        self._detector = self._mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, frame_bgr: np.ndarray) -> List[Box]:
        height, width = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._detector.process(frame_rgb)
        boxes: List[Box] = []
        if not result.detections:
            return boxes

        for detection in result.detections:
            rel = detection.location_data.relative_bounding_box
            x = max(0, int(rel.xmin * width))
            y = max(0, int(rel.ymin * height))
            w = min(width - x, int(rel.width * width))
            h = min(height - y, int(rel.height * height))
            if w > 0 and h > 0:
                boxes.append((x, y, w, h))
        return boxes


class LbfLandmarkEstimator:
    def __init__(self, model_path: Path) -> None:
        require_file(model_path, "LBF landmark model")
        if not hasattr(cv2, "face"):
            raise RuntimeError(
                "cv2.face is unavailable. Install opencv-contrib-python, not only opencv-python."
            )
        self._facemark = cv2.face.createFacemarkLBF()
        self._facemark.loadModel(str(model_path))

    def fit(self, frame_bgr: np.ndarray, boxes: Sequence[Box]) -> List[Optional[np.ndarray]]:
        if not boxes:
            return []

        faces = np.array(boxes, dtype=np.int32)
        ok, landmarks = self._facemark.fit(frame_bgr, faces)
        if not ok:
            return [None for _ in boxes]

        fitted: List[Optional[np.ndarray]] = []
        for points in landmarks:
            fitted.append(np.asarray(points, dtype=np.float32).reshape(-1, 2))
        return fitted


def box_area(box: Box) -> int:
    return max(0, box[2]) * max(0, box[3])


def intersection_area(first: Box, second: Box) -> int:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    return max(0, x2 - x1) * max(0, y2 - y1)


def overlap_over_smaller(first: Box, second: Box) -> float:
    smaller_area = min(box_area(first), box_area(second))
    if smaller_area <= 0:
        return 0.0
    return intersection_area(first, second) / smaller_area


def suppress_overlapping_boxes(boxes: Iterable[Box], overlap_threshold: float = 0.55) -> List[Box]:
    kept: List[Box] = []
    # Haar sometimes returns one tight face box plus one larger nested box.
    # Keeping tighter boxes first avoids drawing duplicate rectangles on a single face.
    for box in sorted(boxes, key=box_area):
        if any(overlap_over_smaller(box, kept_box) >= overlap_threshold for kept_box in kept):
            continue
        kept.append(box)
    return kept


def limit_boxes_by_area(boxes: Iterable[Box], max_faces: int) -> List[Box]:
    sorted_boxes = sorted(boxes, key=lambda box: box[2] * box[3], reverse=True)
    return sorted_boxes[:max_faces] if max_faces > 0 else sorted_boxes


def draw_detections(frame_bgr: np.ndarray, detections: Sequence[DetectionResult]) -> None:
    for result in detections:
        x, y, w, h = result.box
        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 210, 255), 2)
        if result.landmarks is None:
            continue

        for px, py in result.landmarks.astype(int):
            cv2.circle(frame_bgr, (int(px), int(py)), 2, (40, 255, 80), -1, lineType=cv2.LINE_AA)


def draw_status(
    frame_bgr: np.ndarray,
    *,
    fps: float,
    detector_name: str,
    preprocess: str,
    face_count: int,
    elapsed_ms: float,
) -> None:
    lines = [
        f"FPS: {fps:5.1f} | faces: {face_count} | {elapsed_ms:5.1f} ms",
        f"detector: {detector_name} | preprocess: {preprocess}",
        "Press q or Esc to quit, s to save snapshot",
    ]
    x, y = 12, 24
    for idx, text in enumerate(lines):
        baseline = y + idx * 24
        cv2.putText(frame_bgr, text, (x, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame_bgr, text, (x, baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
