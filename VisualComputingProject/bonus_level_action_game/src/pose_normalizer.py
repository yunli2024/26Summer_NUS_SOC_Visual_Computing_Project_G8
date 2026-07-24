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
    valid = frame.valid_mask.copy()

    if valid[11] and valid[12]:
        origin = (kpts[11] + kpts[12]) / 2.0
    elif valid[5] and valid[6]:
        origin = (kpts[5] + kpts[6]) / 2.0
    elif frame.bbox is not None:
        x1, y1, x2, y2 = frame.bbox
        origin = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
    else:
        frame.normalized_keypoints = None
        return frame

    if valid[5] and valid[6]:
        shoulder_width = float(np.linalg.norm(kpts[5] - kpts[6]))
    else:
        shoulder_width = 0.0
    if valid[11] and valid[12]:
        hip_width = float(np.linalg.norm(kpts[11] - kpts[12]))
    else:
        hip_width = 0.0
    bbox_height = float(frame.bbox[3] - frame.bbox[1]) if frame.bbox is not None else 0.0
    scale = max(shoulder_width * 2.2, hip_width * 2.5, bbox_height, 1.0)
    frame.normalized_keypoints = (kpts - origin) / scale
    return frame
