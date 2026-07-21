"""Improved Haar face detection with padding and short-term fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

try:
    from . import config
except ImportError:
    import config


@dataclass
class FaceDetectionResult:
    faces: np.ndarray
    raw_count: int
    filtered_count: int
    status: str
    detected_now: bool
    using_previous: bool
    failed_frames: int
    selected_size: tuple[int, int] | None
    message: str


def _iou(face_a, face_b) -> float:
    ax, ay, aw, ah = face_a
    bx, by, bw, bh = face_b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class ImprovedFaceDetector:
    """Detect faces and keep the last face briefly during short failures."""

    def __init__(self, cascade_path: Path = config.HAAR_CASCADE_PATH) -> None:
        self.cascade_path = Path(cascade_path)
        if not self.cascade_path.exists():
            raise FileNotFoundError(f"Haar cascade file not found: {self.cascade_path}")
        self.classifier = cv2.CascadeClassifier(str(self.cascade_path))
        if self.classifier.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {self.cascade_path}")
        self.last_faces = np.empty((0, 4), dtype=np.int32)
        self.failed_frames = 0

    def detect(self, gray_frame) -> FaceDetectionResult:
        padded, pad_x, pad_y = self._pad(gray_frame)
        faces = self.classifier.detectMultiScale(
            padded,
            scaleFactor=config.FACE_SCALE_FACTOR,
            minNeighbors=config.FACE_MIN_NEIGHBORS,
            minSize=config.FACE_MIN_SIZE,
            maxSize=config.FACE_MAX_SIZE,
        )
        raw_count = len(faces)
        faces = self._map_to_original(faces, gray_frame.shape, pad_x, pad_y)
        faces = self._filter_faces(faces, gray_frame.shape)
        filtered_count = len(faces)

        if len(faces) > 0:
            selected_faces = self._select_faces(faces)
            if len(selected_faces) > 0:
                faces = np.asarray(selected_faces, dtype=np.int32)
                self.last_faces = faces
                self.failed_frames = 0
                selected = faces[0]
                return FaceDetectionResult(
                    faces=faces,
                    raw_count=raw_count,
                    filtered_count=filtered_count,
                    status="DETECTED",
                    detected_now=True,
                    using_previous=False,
                    failed_frames=self.failed_frames,
                    selected_size=(int(selected[2]), int(selected[3])),
                    message="Current-frame face detected",
                )
            filtered_count = 0

        return self._fallback(raw_count, filtered_count)

    def _fallback(self, raw_count: int, filtered_count: int) -> FaceDetectionResult:
        self.failed_frames += 1
        if len(self.last_faces) > 0 and self.failed_frames <= config.MAX_FACE_HOLD_FRAMES:
            face = self.last_faces[0]
            return FaceDetectionResult(
                faces=self.last_faces.copy(),
                raw_count=raw_count,
                filtered_count=filtered_count,
                status="CACHED",
                detected_now=False,
                using_previous=True,
                failed_frames=self.failed_frames,
                selected_size=(int(face[2]), int(face[3])),
                message="Using previous face box briefly",
            )

        self.last_faces = np.empty((0, 4), dtype=np.int32)
        return FaceDetectionResult(
            faces=self.last_faces,
            raw_count=raw_count,
            filtered_count=filtered_count,
            status="LOST",
            detected_now=False,
            using_previous=False,
            failed_frames=self.failed_frames,
            selected_size=None,
            message="Face lost",
        )

    def _select_faces(self, faces):
        faces = self._sort_faces(faces)
        if not config.SINGLE_FACE_MODE or len(self.last_faces) == 0:
            return [face for face in faces if not self._looks_like_inner_false_positive(face)]

        last = self.last_faces[0]
        scored = []
        for face in faces:
            if self._looks_like_inner_false_positive(face):
                continue
            area_score = face[2] * face[3]
            overlap_score = _iou(last, face)
            scored.append((overlap_score, area_score, face))
        if not scored:
            return []
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [scored[0][2]]

    def _select_face(self, faces):
        selected = self._select_faces(faces)
        return selected[0] if selected else None

    def _looks_like_inner_false_positive(self, face) -> bool:
        if len(self.last_faces) == 0:
            return False
        last = self.last_faces[0]
        old_area = last[2] * last[3]
        new_area = face[2] * face[3]
        if old_area <= 0:
            return False
        if new_area / old_area >= config.SUDDEN_SHRINK_AREA_RATIO:
            return False
        return _inside_ratio(face, last) >= config.INNER_FALSE_POSITIVE_IOU

    def _filter_faces(self, faces, original_shape):
        if len(faces) == 0:
            return faces
        h, w = original_shape[:2]
        image_area = h * w
        kept = []
        for x, y, fw, fh in faces:
            area = fw * fh
            aspect = fw / max(fh, 1)
            if area < image_area * config.FACE_MIN_AREA_RATIO:
                continue
            if area > image_area * config.FACE_MAX_AREA_RATIO:
                continue
            if not (config.FACE_MIN_ASPECT_RATIO <= aspect <= config.FACE_MAX_ASPECT_RATIO):
                continue
            kept.append((int(x), int(y), int(fw), int(fh)))
        return np.asarray(kept, dtype=np.int32)

    def _map_to_original(self, faces, original_shape, pad_x: int, pad_y: int):
        if len(faces) == 0:
            return np.empty((0, 4), dtype=np.int32)
        h, w = original_shape[:2]
        mapped = []
        for x, y, fw, fh in faces:
            x = max(0, int(x - pad_x))
            y = max(0, int(y - pad_y))
            fw = min(int(fw), w - x)
            fh = min(int(fh), h - y)
            if fw > 0 and fh > 0:
                mapped.append((x, y, fw, fh))
        return np.asarray(mapped, dtype=np.int32)

    def _sort_faces(self, faces):
        return np.asarray(sorted(faces, key=lambda f: f[2] * f[3], reverse=True), dtype=np.int32)

    def target_changed(self, faces) -> bool:
        if len(self.last_faces) == 0 or len(faces) == 0:
            return True
        return _iou(self.last_faces[0], faces[0]) < config.FACE_CHANGE_IOU_THRESHOLD

    def _pad(self, gray_frame):
        h, w = gray_frame.shape[:2]
        pad_x = max(1, int(w * config.PADDING_RATIO))
        pad_y = max(1, int(h * config.PADDING_RATIO))
        padded = cv2.copyMakeBorder(gray_frame, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE)
        return padded, pad_x, pad_y


def _inside_ratio(inner, outer) -> float:
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    x1, y1 = max(ix, ox), max(iy, oy)
    x2, y2 = min(ix + iw, ox + ow), min(iy + ih, oy + oh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    inner_area = iw * ih
    return inter / inner_area if inner_area > 0 else 0.0
