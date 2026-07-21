"""Frame preprocessing helpers."""

from __future__ import annotations

import cv2
import numpy as np

try:
    from . import config
except ImportError:
    import config


PREPROCESS_MODES = ("raw", "clahe", "gamma", "clahe-gamma")


def to_gray(frame, use_clahe: bool = False, mode: str | None = None):
    """Convert BGR frame to gray and apply the selected realtime preprocessing."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if mode is None:
        mode = "clahe" if use_clahe else "raw"
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"Unknown preprocess mode: {mode}")
    if mode == "raw":
        return gray, "gray"
    if mode == "gamma":
        return _apply_gamma(gray), "gamma"
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE,
    )
    gray = clahe.apply(gray)
    if mode == "clahe-gamma":
        gray = _apply_gamma(gray)
    return gray, mode


def _apply_gamma(gray):
    gamma = max(float(config.GAMMA_VALUE), 1e-6)
    inv_gamma = 1.0 / gamma
    table = np.asarray([((value / 255.0) ** inv_gamma) * 255 for value in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, table)
