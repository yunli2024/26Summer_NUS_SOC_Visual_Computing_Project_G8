"""YOLOv8 pose detector wrapper."""

from __future__ import annotations

import threading
import time
from typing import Iterable, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from . import config
from .pose_types import PersonPose


def choose_device():
    """Return GPU index when CUDA is available, otherwise CPU."""
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class PoseDetector:
    def __init__(self, model_path=config.MODEL_PATH):
        if not model_path.exists():
            raise FileNotFoundError(f"YOLO pose model not found: {model_path}")
        self.model_path = model_path
        self.device = choose_device()
        self.model = YOLO(str(model_path))
        self.lock = threading.Lock()

    def detect(self, frame: np.ndarray) -> Tuple[List[PersonPose], float]:
        start = time.perf_counter()
        with self.lock:
            results = self.model(
                frame,
                conf=config.YOLO_CONF_THRESHOLD,
                device=self.device,
                verbose=False,
            )
        infer_ms = (time.perf_counter() - start) * 1000.0

        people: List[PersonPose] = []
        height, width = frame.shape[:2]
        for result in results:
            if result.keypoints is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else []
            kpts = result.keypoints.xyn.cpu().numpy()
            conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
            for idx, person_kpts in enumerate(kpts):
                keypoints = person_kpts.astype(np.float32).copy()
                keypoints[:, 0] *= width
                keypoints[:, 1] *= height
                confidences = conf[idx].astype(np.float32) if conf is not None else np.ones(17, dtype=np.float32)
                valid = confidences >= config.KEYPOINT_CONF_THRESHOLD
                if idx < len(boxes):
                    bbox = tuple(float(v) for v in boxes[idx])
                else:
                    bbox = self._bbox_from_keypoints(keypoints, valid)
                people.append(PersonPose(bbox=bbox, keypoints=keypoints, confidences=confidences, valid_mask=valid))
        return people, infer_ms

    @staticmethod
    def _bbox_from_keypoints(keypoints: np.ndarray, valid: np.ndarray):
        if not np.any(valid):
            return (0.0, 0.0, 0.0, 0.0)
        pts = keypoints[valid]
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        return (float(x1), float(y1), float(x2), float(y2))


def draw_pose(
    frame: np.ndarray,
    pose: PersonPose | None,
    people_count: int,
    infer_ms: float,
    score_text: str = "",
    highlight_keypoints: Iterable[int] | None = None,
    highlight_joints: Iterable[str] | None = None,
    error_text: str = "",
):
    overlay = frame.copy()
    highlighted = set(highlight_keypoints or [])
    for joint in highlight_joints or []:
        for idx in config.ANGLE_TRIPLES.get(joint, ()):
            highlighted.add(idx)
    if pose is not None:
        x1, y1, x2, y2 = [int(v) for v in pose.bbox]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 180, 0), 2)
        for a, b in config.SKELETON:
            if pose.valid_mask[a] and pose.valid_mask[b]:
                p1 = tuple(pose.keypoints[a].astype(int))
                p2 = tuple(pose.keypoints[b].astype(int))
                color = (0, 120, 255) if a in highlighted or b in highlighted else (0, 210, 70)
                thickness = 4 if a in highlighted or b in highlighted else 2
                cv2.line(overlay, p1, p2, color, thickness)
        for idx, point in enumerate(pose.keypoints):
            if not pose.valid_mask[idx]:
                continue
            x, y = point.astype(int)
            if idx in highlighted:
                cv2.circle(overlay, (x, y), 9, (0, 0, 255), -1)
                cv2.circle(overlay, (x, y), 12, (255, 255, 255), 2)
            else:
                cv2.circle(overlay, (x, y), 5, (30, 30, 255), -1)
            cv2.putText(
                overlay,
                f"{pose.confidences[idx]:.2f}",
                (x + 4, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (20, 20, 180),
                1,
            )

    cv2.putText(overlay, f"People {people_count} | Infer {infer_ms:.1f} ms", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 180, 255), 2)
    if score_text:
        cv2.putText(overlay, score_text, (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 40), 2)
    if error_text:
        cv2.putText(overlay, f"Fix: {error_text}", (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 80, 255), 2)
    return overlay
