"""Feature extraction for normalized poses."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np

from . import config
from .pose_types import PoseFrame


def angle_at(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return math.nan
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def add_joint_angles(frame: PoseFrame) -> PoseFrame:
    if frame.normalized_keypoints is None:
        frame.joint_angles = {}
        return frame
    angles: Dict[str, float] = {}
    for name, (a, b, c) in config.ANGLE_TRIPLES.items():
        if frame.valid_mask[a] and frame.valid_mask[b] and frame.valid_mask[c]:
            angles[name] = angle_at(frame.normalized_keypoints[a], frame.normalized_keypoints[b], frame.normalized_keypoints[c])
    frame.joint_angles = angles
    return frame


def valid_bone_vectors(frame: PoseFrame) -> Dict[Tuple[int, int], np.ndarray]:
    if frame.normalized_keypoints is None:
        return {}
    vectors = {}
    for a, b in config.SKELETON:
        if frame.valid_mask[a] and frame.valid_mask[b]:
            vec = frame.normalized_keypoints[b] - frame.normalized_keypoints[a]
            norm = float(np.linalg.norm(vec))
            if norm > 1e-6:
                vectors[(a, b)] = vec / norm
    return vectors
