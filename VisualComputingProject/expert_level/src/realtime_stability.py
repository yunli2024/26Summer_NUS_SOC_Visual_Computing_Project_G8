from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from keypoint_features import FaceFeatures, build_feature_vector
from face_pipeline import Box, box_area, intersection_area


@dataclass(frozen=True)
class TrackedFace:
    track_id: int
    face: FaceFeatures


@dataclass
class _FaceTrack:
    track_id: int
    box: Box
    face: FaceFeatures
    stable_count: int = 1
    missing_count: int = 0


def box_iou(first: Box, second: Box) -> float:
    union = box_area(first) + box_area(second) - intersection_area(first, second)
    if union <= 0:
        return 0.0
    return intersection_area(first, second) / union


def smooth_box(previous: Box, current: Box, alpha: float) -> Box:
    px, py, pw, ph = previous
    cx, cy, cw, ch = current
    return (
        int(alpha * px + (1.0 - alpha) * cx),
        int(alpha * py + (1.0 - alpha) * cy),
        int(alpha * pw + (1.0 - alpha) * cw),
        int(alpha * ph + (1.0 - alpha) * ch),
    )


def smooth_face_features(
    previous: FaceFeatures,
    current: FaceFeatures,
    box: Box,
    alpha: float,
) -> FaceFeatures:
    if previous.landmarks.shape != current.landmarks.shape:
        return replace(current, box=box)
    smoothed_landmarks = (
        alpha * previous.landmarks.astype(np.float32)
        + (1.0 - alpha) * current.landmarks.astype(np.float32)
    )
    return replace(
        current,
        box=box,
        landmarks=smoothed_landmarks,
        vector=build_feature_vector(
            smoothed_landmarks,
            box,
            current.feature_version,
        ),
    )


def plausible_face(face: FaceFeatures, frame_shape: Tuple[int, int, int], *, min_area_ratio: float) -> bool:
    height, width = frame_shape[:2]
    x, y, w, h = face.box
    if w <= 0 or h <= 0:
        return False
    area_ratio = (w * h) / max(float(width * height), 1.0)
    if area_ratio < min_area_ratio or area_ratio > 0.70:
        return False
    aspect = w / max(float(h), 1.0)
    if aspect < 0.55 or aspect > 1.75:
        return False

    landmarks = face.landmarks
    pad_x = 0.24 * w
    pad_y = 0.24 * h
    inside_x = (landmarks[:, 0] >= x - pad_x) & (landmarks[:, 0] <= x + w + pad_x)
    inside_y = (landmarks[:, 1] >= y - pad_y) & (landmarks[:, 1] <= y + h + pad_y)
    if float(np.mean(inside_x & inside_y)) < 0.75:
        return False

    # YuNet already validates facial appearance with a learned confidence score.
    # Keep only broad LBF sanity checks here so real side-view faces are not lost.
    if face.source == "yunet":
        return True

    left_eye_y = float(np.mean(landmarks[36:42, 1]))
    right_eye_y = float(np.mean(landmarks[42:48, 1]))
    left_eye_x = float(np.mean(landmarks[36:42, 0]))
    right_eye_x = float(np.mean(landmarks[42:48, 0]))
    mouth_y = float(np.mean(landmarks[48:68, 1]))
    chin_y = float(landmarks[8, 1])
    nose_y = float(landmarks[30, 1])
    eye_y = 0.5 * (left_eye_y + right_eye_y)
    eye_distance = abs(right_eye_x - left_eye_x)
    if not (0.14 * w <= eye_distance <= 0.66 * w):
        return False
    if abs(left_eye_y - right_eye_y) > 0.24 * h:
        return False
    if not (y + 0.14 * h <= eye_y <= y + 0.58 * h):
        return False
    if not (eye_y < nose_y < mouth_y < chin_y + 0.12 * h):
        return False
    if not (y + 0.40 * h <= mouth_y <= y + 0.92 * h):
        return False

    spread_x = float(np.percentile(landmarks[:, 0], 90) - np.percentile(landmarks[:, 0], 10))
    spread_y = float(np.percentile(landmarks[:, 1], 90) - np.percentile(landmarks[:, 1], 10))
    if spread_x < 0.34 * w or spread_y < 0.38 * h:
        return False

    return True


class FaceStabilizer:
    def __init__(
        self,
        *,
        stable_frames: int = 2,
        hold_frames: int = 5,
        iou_threshold: float = 0.22,
        box_smoothing: float = 0.65,
        landmark_smoothing: float = 0.85,
        min_area_ratio: float = 0.012,
    ) -> None:
        self.stable_frames = max(1, stable_frames)
        self.hold_frames = max(0, hold_frames)
        self.iou_threshold = iou_threshold
        self.box_smoothing = box_smoothing
        self.landmark_smoothing = min(max(landmark_smoothing, 0.0), 0.98)
        self.min_area_ratio = min_area_ratio
        self._box: Optional[Box] = None
        self._last_face: Optional[FaceFeatures] = None
        self._stable_count = 0
        self._missing_count = 0

    def reset(self) -> None:
        self._box = None
        self._last_face = None
        self._stable_count = 0
        self._missing_count = 0

    def update(self, faces: Sequence[FaceFeatures], frame_shape: Tuple[int, int, int]) -> List[FaceFeatures]:
        candidates = [
            face
            for face in faces
            if plausible_face(face, frame_shape, min_area_ratio=self.min_area_ratio)
        ]

        if not candidates:
            self._missing_count += 1
            if self._last_face is not None and self._missing_count <= self.hold_frames and self._stable_count >= self.stable_frames:
                return [self._last_face]
            self.reset()
            return []

        selected = self._select_candidate(candidates)
        if self._box is None:
            self._box = selected.box
            self._stable_count = 1
        else:
            iou = box_iou(self._box, selected.box)
            if iou < self.iou_threshold and self._stable_count >= self.stable_frames:
                self._box = selected.box
                self._stable_count = 1
            else:
                self._box = smooth_box(self._box, selected.box, self.box_smoothing)
                self._stable_count += 1

        self._missing_count = 0
        if self._last_face is None:
            stabilized = replace(selected, box=self._box)
        else:
            stabilized = smooth_face_features(
                self._last_face,
                selected,
                self._box,
                self.landmark_smoothing,
            )
        self._last_face = stabilized
        if self._stable_count < self.stable_frames:
            return []
        return [stabilized]

    def _select_candidate(self, candidates: Sequence[FaceFeatures]) -> FaceFeatures:
        if self._box is None:
            return max(candidates, key=lambda face: box_area(face.box))
        return max(
            candidates,
            key=lambda face: 2.5 * box_iou(self._box, face.box) + box_area(face.box) / 100000.0,
        )


class MultiFaceStabilizer:
    def __init__(
        self,
        *,
        max_faces: int = 4,
        stable_frames: int = 2,
        hold_frames: int = 4,
        iou_threshold: float = 0.22,
        box_smoothing: float = 0.65,
        landmark_smoothing: float = 0.85,
        min_area_ratio: float = 0.006,
    ) -> None:
        self.max_faces = max(1, max_faces)
        self.stable_frames = max(1, stable_frames)
        self.hold_frames = max(0, hold_frames)
        self.iou_threshold = iou_threshold
        self.box_smoothing = box_smoothing
        self.landmark_smoothing = min(max(landmark_smoothing, 0.0), 0.98)
        self.min_area_ratio = min_area_ratio
        self._tracks: Dict[int, _FaceTrack] = {}
        self._next_track_id = 1

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1

    def update(self, faces: Sequence[FaceFeatures], frame_shape: Tuple[int, int, int]) -> List[TrackedFace]:
        candidates = [
            face
            for face in faces
            if plausible_face(face, frame_shape, min_area_ratio=self.min_area_ratio)
        ]
        candidates = sorted(candidates, key=lambda face: box_area(face.box), reverse=True)[: self.max_faces * 2]

        assigned_tracks: set[int] = set()
        assigned_candidates: set[int] = set()

        pairs = []
        for track_id, track in self._tracks.items():
            for candidate_idx, candidate in enumerate(candidates):
                pairs.append((box_iou(track.box, candidate.box), track_id, candidate_idx))
        for iou, track_id, candidate_idx in sorted(pairs, reverse=True):
            if iou < self.iou_threshold:
                break
            if track_id in assigned_tracks or candidate_idx in assigned_candidates:
                continue
            self._update_track(self._tracks[track_id], candidates[candidate_idx])
            assigned_tracks.add(track_id)
            assigned_candidates.add(candidate_idx)

        for candidate_idx, candidate in enumerate(candidates):
            if candidate_idx in assigned_candidates:
                continue
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _FaceTrack(track_id=track_id, box=candidate.box, face=candidate)
            assigned_tracks.add(track_id)

        stale_track_ids = []
        for track_id, track in self._tracks.items():
            if track_id in assigned_tracks:
                continue
            track.missing_count += 1
            if track.missing_count > self.hold_frames:
                stale_track_ids.append(track_id)
        for track_id in stale_track_ids:
            self._tracks.pop(track_id, None)

        visible_tracks = [
            track
            for track in self._tracks.values()
            if track.stable_count >= self.stable_frames and track.missing_count <= self.hold_frames
        ]
        visible_tracks = sorted(visible_tracks, key=lambda track: box_area(track.box), reverse=True)[: self.max_faces]
        return [TrackedFace(track_id=track.track_id, face=track.face) for track in visible_tracks]

    def _update_track(self, track: _FaceTrack, candidate: FaceFeatures) -> None:
        track.box = smooth_box(track.box, candidate.box, self.box_smoothing)
        track.face = smooth_face_features(
            track.face,
            candidate,
            track.box,
            self.landmark_smoothing,
        )
        track.stable_count += 1
        track.missing_count = 0


class ProbabilitySmoother:
    def __init__(
        self,
        *,
        alpha: float = 0.85,
        min_confidence: float = 0.48,
        switch_margin: float = 0.12,
        initial_frames: int = 6,
        switch_frames: int = 8,
        min_hold_frames: int = 30,
    ) -> None:
        self.alpha = min(max(alpha, 0.0), 0.98)
        self.min_confidence = min_confidence
        self.switch_margin = switch_margin
        self.initial_frames = max(1, initial_frames)
        self.switch_frames = max(1, switch_frames)
        self.min_hold_frames = max(0, min_hold_frames)
        self._ema: Optional[np.ndarray] = None
        self._label: Optional[str] = None
        self._candidate_label: Optional[str] = None
        self._candidate_count = 0
        self._label_age = 0

    def reset(self) -> None:
        self._ema = None
        self._label = None
        self._candidate_label = None
        self._candidate_count = 0
        self._label_age = 0

    def _observe_candidate(self, label: str) -> int:
        if label == self._candidate_label:
            self._candidate_count += 1
        else:
            self._candidate_label = label
            self._candidate_count = 1
        return self._candidate_count

    def _clear_candidate(self) -> None:
        self._candidate_label = None
        self._candidate_count = 0

    def update(self, probabilities: np.ndarray, classes: Sequence[str]) -> Tuple[str, float, np.ndarray]:
        probs = np.asarray(probabilities, dtype=np.float32)
        probs = probs / max(float(np.sum(probs)), 1e-9)
        if self._ema is None:
            self._ema = probs
        else:
            self._ema = self.alpha * self._ema + (1.0 - self.alpha) * probs
            self._ema = self._ema / max(float(np.sum(self._ema)), 1e-9)

        order = np.argsort(self._ema)[::-1]
        top_idx = int(order[0])
        second_idx = int(order[1]) if len(order) > 1 else top_idx
        top_label = str(classes[top_idx])
        top_conf = float(self._ema[top_idx])
        margin = top_conf - float(self._ema[second_idx])
        eligible = top_conf >= self.min_confidence and margin >= self.switch_margin

        if self._label is None:
            if eligible and self._observe_candidate(top_label) >= self.initial_frames:
                self._label = top_label
                self._label_age = 0
                self._clear_candidate()
            if self._label is None:
                return "uncertain", top_conf, self._ema.copy()
        else:
            self._label_age += 1
            if top_label == self._label:
                self._clear_candidate()
            elif self._label_age < self.min_hold_frames or not eligible:
                self._clear_candidate()
            elif self._observe_candidate(top_label) >= self.switch_frames:
                self._label = top_label
                self._label_age = 0
                self._clear_candidate()

        label_idx = int(list(classes).index(self._label))
        return self._label, float(self._ema[label_idx]), self._ema.copy()
