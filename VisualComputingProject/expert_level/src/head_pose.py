"""Head pose estimation from 68 facial landmarks."""

from __future__ import annotations

import cv2
import numpy as np

try:
    from . import config
except ImportError:
    import config


MODEL_POINTS_3D = np.asarray(
    [
        (0.0, 0.0, 0.0),          # nose tip
        (0.0, -330.0, -65.0),     # chin
        (-225.0, 170.0, -135.0),  # left eye outer corner
        (225.0, 170.0, -135.0),   # right eye outer corner
        (-150.0, -150.0, -125.0), # left mouth corner
        (150.0, -150.0, -125.0),  # right mouth corner
    ],
    dtype=np.float64,
)


LANDMARK_INDEXES = [30, 8, 36, 45, 48, 54]


def estimate_head_pose(landmarks: np.ndarray, image_shape) -> tuple[float, float, float] | None:
    """Estimate yaw, pitch, and roll in degrees using solvePnP."""

    height, width = image_shape[:2]
    image_points = np.asarray([landmarks[index] for index in LANDMARK_INDEXES], dtype=np.float64)
    focal_length = float(width)
    camera_matrix = np.asarray(
        [
            [focal_length, 0.0, width / 2.0],
            [0.0, focal_length, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    distortion = np.zeros((4, 1), dtype=np.float64)
    ok, rotation_vector, _translation_vector = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rotation_matrix, _jacobian = cv2.Rodrigues(rotation_vector)
    angles, _mtx_r, _mtx_q, _qx, _qy, _qz = cv2.RQDecomp3x3(rotation_matrix)
    pitch, yaw, roll = [float(angle) for angle in angles]
    return yaw, pitch, roll


def classify_head_pose(yaw: float, pitch: float) -> str:
    """Convert yaw and pitch angles into a simple pose label."""

    if config.HEAD_POSE_INVERT_YAW:
        yaw = -yaw
    if config.HEAD_POSE_INVERT_PITCH:
        pitch = -pitch

    if abs(yaw) < config.HEAD_POSE_YAW_THRESHOLD and abs(pitch) < config.HEAD_POSE_PITCH_THRESHOLD:
        return "Facing Forward"
    if yaw <= -config.HEAD_POSE_YAW_THRESHOLD:
        return "Looking Left"
    if yaw >= config.HEAD_POSE_YAW_THRESHOLD:
        return "Looking Right"
    if pitch >= config.HEAD_POSE_PITCH_THRESHOLD:
        return "Looking Down"
    if pitch <= -config.HEAD_POSE_PITCH_THRESHOLD:
        return "Looking Up"
    return "Facing Forward"


class HeadPoseSmoother:
    """EMA smoother for yaw, pitch, and roll."""

    def __init__(self, alpha: float = config.HEAD_POSE_SMOOTHING_ALPHA) -> None:
        self.alpha = alpha
        self.smoothed: np.ndarray | None = None

    def reset(self) -> None:
        self.smoothed = None

    def update(self, pose: tuple[float, float, float] | None) -> tuple[float, float, float] | None:
        if pose is None:
            return None
        current = np.asarray(pose, dtype=np.float32)
        if self.smoothed is None:
            self.smoothed = current
        else:
            self.smoothed = self.alpha * current + (1.0 - self.alpha) * self.smoothed
        return tuple(float(value) for value in self.smoothed)
