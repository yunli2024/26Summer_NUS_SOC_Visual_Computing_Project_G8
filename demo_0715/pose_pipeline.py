from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


Keypoint = Tuple[int, int, float]

COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

COCO_SKELETON = (
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 5),
    (0, 6),
)


@dataclass(frozen=True)
class PoseDetection:
    keypoints: List[Keypoint]
    box: Tuple[int, int, int, int]
    box_confidence: float
    visible_count: int
    main_score: float


@dataclass(frozen=True)
class FramePoseResult:
    frame_index: int
    detections: List[PoseDetection]
    main_index: Optional[int]
    inference_ms: float
    draw_ms: float

    @property
    def person_count(self) -> int:
        return len(self.detections)

    @property
    def main_detection(self) -> Optional[PoseDetection]:
        if self.main_index is None:
            return None
        return self.detections[self.main_index]


class PoseEstimator:
    def __init__(self, model_path: Path, *, device: Optional[str] = None, imgsz: int = 512) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO pose model not found: {model_path}")
        self.model = YOLO(str(model_path))
        self.device = device
        self.imgsz = imgsz

    def infer(
        self,
        frame_bgr: np.ndarray,
        *,
        frame_index: int,
        conf: float = 0.3,
        keypoint_conf: float = 0.25,
        main_only: bool = False,
    ) -> FramePoseResult:
        return self.infer_batch(
            [frame_bgr],
            frame_indices=[frame_index],
            conf=conf,
            keypoint_conf=keypoint_conf,
            main_only=main_only,
        )[0]

    def infer_batch(
        self,
        frames_bgr: Sequence[np.ndarray],
        *,
        frame_indices: Sequence[int],
        conf: float = 0.3,
        keypoint_conf: float = 0.25,
        main_only: bool = False,
    ) -> List[FramePoseResult]:
        if len(frames_bgr) != len(frame_indices):
            raise ValueError("frames_bgr and frame_indices must have the same length")
        if not frames_bgr:
            return []

        start = time.perf_counter()
        raw_results = self.model(
            list(frames_bgr),
            conf=conf,
            device=self.device,
            imgsz=self.imgsz,
            verbose=False,
        )
        per_frame_ms = (time.perf_counter() - start) * 1000.0 / len(frames_bgr)
        parsed: List[FramePoseResult] = []
        for frame_bgr, frame_index, result in zip(frames_bgr, frame_indices, raw_results):
            parsed.append(
                self._parse_result(
                    frame_bgr,
                    result,
                    frame_index=frame_index,
                    keypoint_conf=keypoint_conf,
                    inference_ms=per_frame_ms,
                    main_only=main_only,
                )
            )
        return parsed

    @staticmethod
    def _parse_result(
        frame_bgr: np.ndarray,
        result,
        *,
        frame_index: int,
        keypoint_conf: float,
        inference_ms: float,
        main_only: bool,
    ) -> FramePoseResult:
        height, width = frame_bgr.shape[:2]
        detections: List[PoseDetection] = []
        if result.keypoints is not None:
            kpts_xyn = result.keypoints.xyn.cpu().numpy()
            if result.keypoints.conf is not None:
                kpts_conf = result.keypoints.conf.cpu().numpy()
            else:
                kpts_conf = np.ones(kpts_xyn.shape[:2], dtype=np.float32)

            boxes_xyxy = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else np.zeros((len(kpts_xyn), 4))
            boxes_conf = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(kpts_xyn))

            for person_idx, person_kpts in enumerate(kpts_xyn):
                keypoints: List[Keypoint] = []
                visible_count = 0
                for point_idx, (x_norm, y_norm) in enumerate(person_kpts):
                    confidence = float(kpts_conf[person_idx][point_idx])
                    x = int(np.clip(x_norm, 0.0, 1.0) * width)
                    y = int(np.clip(y_norm, 0.0, 1.0) * height)
                    if confidence >= keypoint_conf:
                        visible_count += 1
                    keypoints.append((x, y, confidence))

                if person_idx < len(boxes_xyxy):
                    x1, y1, x2, y2 = boxes_xyxy[person_idx]
                    box = (
                        int(np.clip(x1, 0, width - 1)),
                        int(np.clip(y1, 0, height - 1)),
                        int(np.clip(x2, 0, width - 1)),
                        int(np.clip(y2, 0, height - 1)),
                    )
                    box_conf = float(boxes_conf[person_idx])
                else:
                    box = infer_box_from_keypoints(keypoints, keypoint_conf, width, height)
                    box_conf = 0.0

                score = main_dancer_score(box, visible_count, width, height)
                detections.append(
                    PoseDetection(
                        keypoints=keypoints,
                        box=box,
                        box_confidence=box_conf,
                        visible_count=visible_count,
                        main_score=score,
                    )
                )

        main_index = select_main_dancer(detections)
        if main_only and main_index is not None:
            detections = [detections[main_index]]
            main_index = 0
        return FramePoseResult(frame_index=frame_index, detections=detections, main_index=main_index, inference_ms=inference_ms, draw_ms=0.0)


class PoseStreamTracker:
    """Keep the selected dancer stable and smooth the selected pose over time."""

    def __init__(self, *, smoothing: float = 0.55, max_missing: int = 5) -> None:
        self.smoothing = float(np.clip(smoothing, 0.0, 0.95))
        self.max_missing = max(0, max_missing)
        self.previous_box: Optional[Tuple[int, int, int, int]] = None
        self.previous_keypoints: Optional[np.ndarray] = None
        self.missing_frames = 0

    def reset(self) -> None:
        self.previous_box = None
        self.previous_keypoints = None
        self.missing_frames = 0

    def update(self, result: FramePoseResult) -> FramePoseResult:
        if not result.detections:
            self.missing_frames += 1
            if self.missing_frames > self.max_missing:
                self.reset()
            return replace(result, main_index=None)

        main_index = self._select(result.detections)
        selected = result.detections[main_index]
        current = np.asarray(selected.keypoints, dtype=np.float32)
        if self.previous_keypoints is not None and self.previous_keypoints.shape == current.shape:
            reliable = (current[:, 2] >= 0.15) & (self.previous_keypoints[:, 2] >= 0.15)
            current[reliable, :2] = (
                self.smoothing * self.previous_keypoints[reliable, :2]
                + (1.0 - self.smoothing) * current[reliable, :2]
            )
            current[reliable, 2] = np.maximum(current[reliable, 2], self.previous_keypoints[reliable, 2] * 0.85)

        smoothed = replace(
            selected,
            keypoints=[(int(round(x)), int(round(y)), float(c)) for x, y, c in current],
            visible_count=int((current[:, 2] >= 0.25).sum()),
        )
        detections = list(result.detections)
        detections[main_index] = smoothed
        self.previous_box = selected.box
        self.previous_keypoints = current
        self.missing_frames = 0
        return replace(result, detections=detections, main_index=main_index)

    def _select(self, detections: Sequence[PoseDetection]) -> int:
        if self.previous_box is None:
            return int(max(range(len(detections)), key=lambda idx: detections[idx].main_score))

        previous_center = box_center(self.previous_box)
        previous_diag = max(box_diagonal(self.previous_box), 1.0)

        def continuity_score(detection: PoseDetection) -> float:
            center_distance = np.linalg.norm(np.asarray(box_center(detection.box)) - np.asarray(previous_center))
            continuity = float(np.exp(-center_distance / (previous_diag * 0.8)))
            return (
                2.2 * box_iou(self.previous_box, detection.box)
                + 1.0 * continuity
                + 0.35 * (detection.visible_count / 17.0)
                + 0.20 * detection.box_confidence
                + 0.15 * detection.main_score
            )

        return int(max(range(len(detections)), key=lambda idx: continuity_score(detections[idx])))


def infer_box_from_keypoints(keypoints: Sequence[Keypoint], conf: float, width: int, height: int) -> Tuple[int, int, int, int]:
    xs = [x for x, _, c in keypoints if c >= conf]
    ys = [y for _, y, c in keypoints if c >= conf]
    if not xs or not ys:
        return 0, 0, 0, 0
    return max(0, min(xs)), max(0, min(ys)), min(width - 1, max(xs)), min(height - 1, max(ys))


def box_area(box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def box_diagonal(box: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = box
    return float(np.hypot(max(0, x2 - x1), max(0, y2 - y1)))


def box_iou(first: Tuple[int, int, int, int], second: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = box_area(first) + box_area(second) - intersection
    return intersection / max(float(union), 1.0)


def main_dancer_score(box: Tuple[int, int, int, int], visible_count: int, width: int, height: int) -> float:
    area_ratio = box_area(box) / max(float(width * height), 1.0)
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) * 0.5
    center_y = (y1 + y2) * 0.5
    center_dist = abs(center_x - width * 0.5) / max(width * 0.5, 1.0) + abs(center_y - height * 0.5) / max(height * 0.5, 1.0)
    center_bonus = max(0.0, 1.0 - 0.5 * center_dist)
    visibility_ratio = visible_count / 17.0
    return 2.2 * area_ratio + 0.9 * visibility_ratio + 0.4 * center_bonus


def select_main_dancer(detections: Sequence[PoseDetection]) -> Optional[int]:
    if not detections:
        return None
    return int(max(range(len(detections)), key=lambda idx: detections[idx].main_score))


def draw_pose_overlay(
    frame_bgr: np.ndarray,
    result: FramePoseResult,
    *,
    keypoint_conf: float = 0.25,
    show_video: bool = True,
    main_only: bool = False,
) -> Tuple[np.ndarray, FramePoseResult]:
    start = time.perf_counter()
    canvas = frame_bgr.copy() if show_video else np.full_like(frame_bgr, 245)
    detections = result.detections
    main_index = result.main_index

    for idx, detection in enumerate(detections):
        is_main = idx == main_index
        if main_only and not is_main:
            continue
        draw_detection(canvas, detection, keypoint_conf=keypoint_conf, is_main=is_main)

    draw_status(canvas, result)
    draw_ms = (time.perf_counter() - start) * 1000.0
    return canvas, FramePoseResult(
        frame_index=result.frame_index,
        detections=result.detections,
        main_index=result.main_index,
        inference_ms=result.inference_ms,
        draw_ms=draw_ms,
    )


def draw_detection(frame_bgr: np.ndarray, detection: PoseDetection, *, keypoint_conf: float, is_main: bool) -> None:
    line_color = (40, 220, 80) if is_main else (180, 180, 180)
    point_color = (20, 60, 255) if is_main else (130, 130, 130)
    box_color = (0, 220, 255) if is_main else (180, 180, 180)
    thickness = 3 if is_main else 1

    x1, y1, x2, y2 = detection.box
    if x2 > x1 and y2 > y1:
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), box_color, thickness, cv2.LINE_AA)

    for start_idx, end_idx in COCO_SKELETON:
        if start_idx >= len(detection.keypoints) or end_idx >= len(detection.keypoints):
            continue
        x1, y1, c1 = detection.keypoints[start_idx]
        x2, y2, c2 = detection.keypoints[end_idx]
        if c1 >= keypoint_conf and c2 >= keypoint_conf:
            cv2.line(frame_bgr, (x1, y1), (x2, y2), line_color, thickness, cv2.LINE_AA)

    for x, y, confidence in detection.keypoints:
        if confidence >= keypoint_conf:
            cv2.circle(frame_bgr, (x, y), 4 if is_main else 3, point_color, -1, cv2.LINE_AA)

    label = f"{'main ' if is_main else ''}vis {detection.visible_count}/17 box {detection.box_confidence:.2f}"
    y_text = max(18, y1 - 8)
    cv2.putText(frame_bgr, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame_bgr, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)


def draw_status(frame_bgr: np.ndarray, result: FramePoseResult) -> None:
    main_visible = result.main_detection.visible_count if result.main_detection is not None else 0
    text = (
        f"frame {result.frame_index} | persons {result.person_count} | "
        f"main visible {main_visible}/17 | infer {result.inference_ms:.1f} ms"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.58
    thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = 12, 24
    cv2.rectangle(frame_bgr, (6, 6), (x + text_width + 8, y + baseline + 6), (0, 0, 0), -1)
    cv2.putText(frame_bgr, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
