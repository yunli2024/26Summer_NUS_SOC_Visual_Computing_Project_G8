"""Select the main dancer from all detected people."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from . import config
from .pose_types import PersonPose


class DancerTracker:
    def __init__(self):
        self.previous_center: Optional[np.ndarray] = None

    def reset(self):
        self.previous_center = None

    def select(self, people: List[PersonPose], frame_shape: Tuple[int, int, int]) -> Tuple[Optional[int], Optional[PersonPose]]:
        if not people:
            return None, None
        height, width = frame_shape[:2]
        frame_area = float(max(width * height, 1))
        frame_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        diag = float(np.hypot(width, height))

        best_idx = 0
        best_score = -1.0
        for idx, pose in enumerate(people):
            x1, y1, x2, y2 = pose.bbox
            area_score = max(0.0, (x2 - x1) * (y2 - y1)) / frame_area
            center_score = 1.0 - min(1.0, float(np.linalg.norm(pose.center - frame_center)) / max(diag * 0.5, 1.0))
            if self.previous_center is None:
                continuity_score = 0.5
            else:
                continuity_score = 1.0 - min(1.0, float(np.linalg.norm(pose.center - self.previous_center)) / max(diag * 0.35, 1.0))
            confidence_score = pose.pose_confidence
            score = (
                config.TRACK_WEIGHTS["area"] * area_score
                + config.TRACK_WEIGHTS["center"] * center_score
                + config.TRACK_WEIGHTS["continuity"] * continuity_score
                + config.TRACK_WEIGHTS["confidence"] * confidence_score
            )
            pose.score = float(score)
            if score > best_score:
                best_score = score
                best_idx = idx
        self.previous_center = people[best_idx].center
        return best_idx, people[best_idx]
