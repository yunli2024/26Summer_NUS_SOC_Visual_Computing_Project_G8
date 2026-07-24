"""Image preprocessing helpers for realtime Expert Level demo."""

from __future__ import annotations

import cv2
import numpy as np

try:
    from . import config
except ImportError:
    import config


PREPROCESS_MODES = ("raw", "clahe", "gamma", "clahe-gamma")


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gamma(gray: np.ndarray, gamma: float = config.REALTIME_GAMMA_VALUE) -> np.ndarray:
    gamma = max(float(gamma), 1e-6)
    inv_gamma = 1.0 / gamma
    table = np.asarray([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, table)


def apply_clahe(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(
        clipLimit=config.REALTIME_CLAHE_CLIP_LIMIT,
        tileGridSize=config.REALTIME_CLAHE_TILE_GRID,
    )
    return clahe.apply(gray)


def preprocess_realtime_image(image: np.ndarray, mode: str = config.REALTIME_PREPROCESS_MODE) -> np.ndarray:
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"Unknown realtime preprocess mode: {mode}")
    gray = to_gray(image)
    if mode == "raw":
        return gray
    if mode == "clahe":
        return apply_clahe(gray)
    if mode == "gamma":
        return apply_gamma(gray)
    if mode == "clahe-gamma":
        return apply_gamma(apply_clahe(gray))
    return gray


def image_stats(gray: np.ndarray) -> tuple[float, float]:
    return float(np.mean(gray)), float(np.std(gray))
