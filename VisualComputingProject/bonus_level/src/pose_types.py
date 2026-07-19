"""Small data structures shared by the pose pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np


@dataclass
class PersonPose:
    bbox: Tuple[float, float, float, float]
    keypoints: np.ndarray
    confidences: np.ndarray
    valid_mask: np.ndarray
    score: float = 0.0

    @property
    def pose_confidence(self) -> float:
        if self.confidences.size == 0:
            return 0.0
        valid = self.confidences[self.valid_mask]
        return float(valid.mean()) if valid.size else 0.0

    @property
    def center(self) -> np.ndarray:
        x1, y1, x2, y2 = self.bbox
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


@dataclass
class PoseFrame:
    timestamp: float
    frame_index: int
    source: str
    bbox: Optional[Tuple[float, float, float, float]]
    keypoints: np.ndarray
    confidences: np.ndarray
    valid_mask: np.ndarray
    normalized_keypoints: Optional[np.ndarray] = None
    joint_angles: Dict[str, float] = field(default_factory=dict)
    pose_confidence: float = 0.0
    people_count: int = 0
    selected_index: int = -1
    infer_ms: float = 0.0

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    @classmethod
    def empty(cls, timestamp: float, frame_index: int, source: str, people_count: int = 0) -> "PoseFrame":
        return cls(
            timestamp=timestamp,
            frame_index=frame_index,
            source=source,
            bbox=None,
            keypoints=np.zeros((17, 2), dtype=np.float32),
            confidences=np.zeros(17, dtype=np.float32),
            valid_mask=np.zeros(17, dtype=bool),
            people_count=people_count,
        )
