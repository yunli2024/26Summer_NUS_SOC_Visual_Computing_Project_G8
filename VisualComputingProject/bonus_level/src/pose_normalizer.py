"""Pose normalization and webcam mirror handling."""

from __future__ import annotations

import numpy as np

from . import config
from .pose_types import PoseFrame


def maybe_mirror_keypoints(keypoints: np.ndarray, valid: np.ndarray, confidences: np.ndarray, frame_width: int):
    mirrored = keypoints.copy()
    mirrored[:, 0] = frame_width - mirrored[:, 0]
    valid_out = valid.copy()
    conf_out = confidences.copy()
    if config.SWAP_LEFT_RIGHT:
        for left, right in config.LEFT_RIGHT_PAIRS:
            mirrored[[left, right]] = mirrored[[right, left]]
            valid_out[[left, right]] = valid_out[[right, left]]
            conf_out[[left, right]] = conf_out[[right, left]]
    return mirrored, valid_out, conf_out


def swap_left_right_keypoints(keypoints: np.ndarray, valid: np.ndarray, confidences: np.ndarray):
    swapped = keypoints.copy()
    valid_out = valid.copy()
    conf_out = confidences.copy()
    for left, right in config.LEFT_RIGHT_PAIRS:
        swapped[[left, right]] = swapped[[right, left]]
        valid_out[[left, right]] = valid_out[[right, left]]
        conf_out[[left, right]] = conf_out[[right, left]]
    return swapped, valid_out, conf_out


def normalize_pose(frame: PoseFrame) -> PoseFrame:
    kpts = frame.keypoints.astype(np.float32).copy()
    valid = frame.valid_mask.copy() & np.isfinite(kpts).all(axis=1)
    body_indices = np.asarray(config.BODY_JOINTS, dtype=np.int32)
    if np.count_nonzero(valid[body_indices]) < config.MIN_COMMON_BODY_KEYPOINTS:
        frame.normalized_keypoints = None
        return frame

    def center(indices: tuple[int, ...]):
        available = [index for index in indices if valid[index]]
        if not available:
            return None
        return np.mean(kpts[available], axis=0)

    hip_center = center((11, 12))
    shoulder_center = center((5, 6))
    origin = hip_center if hip_center is not None else shoulder_center
    if origin is None:
        origin = np.mean(kpts[body_indices[valid[body_indices]]], axis=0)

    scale_candidates = []
    for first, second in ((5, 6), (11, 12)):
        if valid[first] and valid[second]:
            scale_candidates.append(float(np.linalg.norm(kpts[first] - kpts[second])))
    if hip_center is not None and shoulder_center is not None:
        scale_candidates.append(float(np.linalg.norm(hip_center - shoulder_center)) * 1.35)
    body = kpts[body_indices[valid[body_indices]]]
    if len(body) >= 2:
        scale_candidates.append(float(np.linalg.norm(np.ptp(body, axis=0))) * 0.35)

    scale = max(scale_candidates, default=0.0)
    if scale < 1e-6:
        frame.normalized_keypoints = None
        return frame

    normalized = np.full((17, 2), np.nan, dtype=np.float32)
    normalized[valid] = (kpts[valid] - origin) / scale
    frame.valid_mask = valid
    frame.normalized_keypoints = normalized
    return frame
