"""Low-latency keypoint smoothing and short missing-point recovery."""

from __future__ import annotations

import numpy as np

from . import config
from .pose_types import PersonPose


class PoseFilter:
    def __init__(self, alpha: float = config.EMA_ALPHA, max_missing: int = config.MAX_MISSING_FRAMES):
        self.alpha = alpha
        self.max_missing = max_missing
        self.prev_points = np.zeros((17, 2), dtype=np.float32)
        self.prev_conf = np.zeros(17, dtype=np.float32)
        self.prev_valid = np.zeros(17, dtype=bool)
        self.missing = np.zeros(17, dtype=np.int32)

    def reset(self):
        self.prev_points[:] = 0
        self.prev_conf[:] = 0
        self.prev_valid[:] = False
        self.missing[:] = 0

    def apply(self, pose: PersonPose | None) -> PersonPose | None:
        if pose is None:
            self.missing += 1
            return None

        points = pose.keypoints.copy()
        conf = pose.confidences.copy()
        valid = pose.valid_mask.copy()

        for idx in range(17):
            if valid[idx]:
                if self.prev_valid[idx]:
                    points[idx] = self.alpha * points[idx] + (1.0 - self.alpha) * self.prev_points[idx]
                self.missing[idx] = 0
            elif self.prev_valid[idx] and self.missing[idx] < self.max_missing:
                # Reuse a briefly missing keypoint with decaying confidence.
                self.missing[idx] += 1
                points[idx] = self.prev_points[idx]
                conf[idx] = max(0.05, self.prev_conf[idx] * (0.75 ** self.missing[idx]))
                valid[idx] = True
            else:
                self.missing[idx] += 1

        self.prev_points = points.copy()
        self.prev_conf = conf.copy()
        self.prev_valid = valid.copy()
        return PersonPose(bbox=pose.bbox, keypoints=points, confidences=conf, valid_mask=valid, score=pose.score)
