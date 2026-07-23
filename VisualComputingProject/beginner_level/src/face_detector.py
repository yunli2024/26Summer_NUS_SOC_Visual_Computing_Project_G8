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
        self.frame_index = 0

    def detect(self, gray_frame) -> FaceDetectionResult:
        self.frame_index += 1
        force_full_frame = len(self.last_faces) == 0 or self.frame_index % config.FULL_DETECT_INTERVAL == 0
        faces = np.empty((0, 4), dtype=np.int32)
        raw_count = 0
        roi_faces = self._detect_roi_faces(gray_frame)
        if len(roi_faces) > 0:
            faces = roi_faces
            raw_count += len(roi_faces)
        if force_full_frame or len(faces) == 0:
            padded, pad_x, pad_y = self._pad(gray_frame)
            full_faces = self.classifier.detectMultiScale(
                padded,
                scaleFactor=config.FACE_SCALE_FACTOR,
                minNeighbors=config.FACE_MIN_NEIGHBORS,
                minSize=config.FACE_MIN_SIZE,
                maxSize=config.FACE_MAX_SIZE,
            )
            mapped_full_faces = self._map_to_original(full_faces, gray_frame.shape, pad_x, pad_y)
            faces = np.vstack([faces, mapped_full_faces]) if len(faces) > 0 else mapped_full_faces
            raw_count += len(full_faces)
        faces = self._dedupe_faces(faces)
        faces = self._filter_faces(faces, gray_frame.shape)
        faces = self._reject_nested_mouth_faces(faces)
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

    def confirm_faces(self, faces) -> None:
        if faces is None or len(faces) == 0:
            return
        self.last_faces = np.asarray(faces, dtype=np.int32)
        self.failed_frames = 0

    def reject_current_detection(self) -> None:
        self.last_faces = np.empty((0, 4), dtype=np.int32)
        self.failed_frames = 0

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
        return any(_looks_like_shrunken_track_false_face(face, last) for last in self.last_faces)

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

    def _dedupe_faces(self, faces):
        if len(faces) == 0:
            return np.empty((0, 4), dtype=np.int32)
        selected = []
        for candidate in self._sort_faces(faces):
            if any(_iou(candidate, existing) > config.FACE_DETECTION_DEDUPE_IOU for existing in selected):
                continue
            selected.append(tuple(int(v) for v in candidate))
        return np.asarray(selected, dtype=np.int32)

    def _reject_nested_mouth_faces(self, faces):
        if len(faces) <= 1:
            return faces
        kept = []
        for candidate in self._sort_faces(faces):
            if any(_looks_like_mouth_false_face(candidate, parent) for parent in kept):
                continue
            kept.append(tuple(int(v) for v in candidate))
        return np.asarray(kept, dtype=np.int32)

    def _detect_roi_faces(self, gray_frame):
        if not config.ROI_TRACKING_ENABLED or len(self.last_faces) == 0:
            return np.empty((0, 4), dtype=np.int32)
        h, w = gray_frame.shape[:2]
        candidates = []
        for search_face in self.last_faces:
            rx, ry, rw, rh = _expand_face_box(
                search_face,
                image_width=w,
                image_height=h,
                expand_x=config.ROI_EXPAND_X,
                expand_top=config.ROI_EXPAND_TOP,
                expand_bottom=config.ROI_EXPAND_BOTTOM,
            )
            if rw <= 0 or rh <= 0:
                continue
            roi_gray = gray_frame[ry : ry + rh, rx : rx + rw]
            min_w = max(24, int(float(search_face[2]) * config.ROI_MIN_FACE_SCALE))
            min_h = max(24, int(float(search_face[3]) * config.ROI_MIN_FACE_SCALE))
            roi_faces = self.classifier.detectMultiScale(
                roi_gray,
                scaleFactor=config.FACE_SCALE_FACTOR,
                minNeighbors=config.FACE_MIN_NEIGHBORS,
                minSize=(min_w, min_h),
                maxSize=config.FACE_MAX_SIZE,
            )
            sorted_roi_faces = sorted(roi_faces, key=lambda face: face[2] * face[3], reverse=True)
            for x, y, fw, fh in sorted_roi_faces[: config.ROI_MAX_RESULTS_PER_FACE]:
                candidates.append((int(x + rx), int(y + ry), int(fw), int(fh)))
        return np.asarray(candidates, dtype=np.int32)

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


def _looks_like_mouth_false_face(candidate, parent) -> bool:
    cx, cy, cw, ch = [float(value) for value in candidate]
    px, py, pw, ph = [float(value) for value in parent]
    parent_area = pw * ph
    candidate_area = cw * ch
    if parent_area <= 0 or candidate_area <= 0:
        return False
    if candidate_area / parent_area > config.MOUTH_FALSE_FACE_AREA_RATIO:
        return False
    if _inside_ratio(candidate, parent) < config.MOUTH_FALSE_FACE_INSIDE_RATIO:
        return False
    center_y_ratio = ((cy + ch / 2.0) - py) / max(ph, 1.0)
    return center_y_ratio >= config.MOUTH_FALSE_FACE_MIN_CENTER_Y


def _looks_like_shrunken_track_false_face(candidate, previous) -> bool:
    cx, cy, cw, ch = [float(value) for value in candidate]
    px, py, pw, ph = [float(value) for value in previous]
    previous_area = pw * ph
    candidate_area = cw * ch
    if previous_area <= 0 or candidate_area <= 0:
        return False
    area_ratio = candidate_area / previous_area
    if area_ratio >= config.SUDDEN_SHRINK_TRACK_AREA_RATIO:
        return False

    candidate_center_x = cx + cw / 2.0
    candidate_center_y = cy + ch / 2.0
    margin_x = pw * config.SUDDEN_SHRINK_TRACK_CENTER_MARGIN
    margin_y = ph * config.SUDDEN_SHRINK_TRACK_CENTER_MARGIN
    center_near_previous = (
        px - margin_x <= candidate_center_x <= px + pw + margin_x
        and py - margin_y <= candidate_center_y <= py + ph + margin_y
    )
    if not center_near_previous:
        return False

    inside_ratio = _inside_ratio(candidate, previous)
    if inside_ratio >= config.SUDDEN_SHRINK_TRACK_MIN_INSIDE_RATIO:
        return True

    previous_center = np.asarray([px + pw / 2.0, py + ph / 2.0], dtype=np.float32)
    candidate_center = np.asarray([candidate_center_x, candidate_center_y], dtype=np.float32)
    center_shift = float(np.linalg.norm(candidate_center - previous_center)) / max(pw, ph, 1.0)
    return center_shift <= config.SUDDEN_SHRINK_TRACK_CENTER_MARGIN


def _expand_face_box(
    box,
    image_width: int,
    image_height: int,
    expand_x: float,
    expand_top: float,
    expand_bottom: float,
):
    x, y, w, h = [float(value) for value in box]
    new_x = x - w * expand_x
    new_y = y - h * expand_top
    new_w = w * (1.0 + 2.0 * expand_x)
    new_h = h * (1.0 + expand_top + expand_bottom)

    x1 = max(0, int(round(new_x)))
    y1 = max(0, int(round(new_y)))
    x2 = min(image_width, int(round(new_x + new_w)))
    y2 = min(image_height, int(round(new_y + new_h)))
    if x2 <= x1 or y2 <= y1:
        return tuple(int(v) for v in box)
    return x1, y1, x2 - x1, y2 - y1
