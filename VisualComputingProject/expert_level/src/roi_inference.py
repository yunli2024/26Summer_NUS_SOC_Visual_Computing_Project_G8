"""Inference-only tensor transforms and landmark ROI geometry."""

from __future__ import annotations

import numpy as np
import torch
from torchvision import transforms

try:
    from . import config
except ImportError:
    import config


def inference_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def roi_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((config.CNN_INPUT_SIZE, config.CNN_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([config.CNN_NORMALIZATION_MEAN], [config.CNN_NORMALIZATION_STD]),
        ]
    )


def clamp_box(box, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0.0, min(x1, width - 1))
    y1 = max(0.0, min(y1, height - 1))
    x2 = max(x1 + 1.0, min(x2, width))
    y2 = max(y1 + 1.0, min(y2, height))
    return np.asarray([round(x1), round(y1), round(x2), round(y2)], dtype=np.int32)


def _box_from_landmarks(
    points: np.ndarray,
    indexes: list[int],
    width: int,
    height: int,
    padding: float,
) -> np.ndarray:
    region = points[indexes]
    x1, y1 = region.min(axis=0)
    x2, y2 = region.max(axis=0)
    region_width = max(float(x2 - x1), 1.0)
    region_height = max(float(y2 - y1), 1.0)
    return clamp_box(
        [
            x1 - region_width * padding,
            y1 - region_height * padding,
            x2 + region_width * padding,
            y2 + region_height * padding,
        ],
        width,
        height,
    )


def roi_boxes_from_landmarks(points: np.ndarray, image_shape) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape[:2]
    eye_brow_indexes = list(range(17, 27)) + list(range(36, 48))
    nose_mouth_indexes = list(range(27, 36)) + list(range(48, 68))
    eye_box = _box_from_landmarks(
        points,
        eye_brow_indexes,
        width,
        height,
        config.ROI_CNN_EYE_BROW_PADDING,
    )
    mouth_box = _box_from_landmarks(
        points,
        nose_mouth_indexes,
        width,
        height,
        config.ROI_CNN_NOSE_MOUTH_PADDING,
    )
    return eye_box, mouth_box
