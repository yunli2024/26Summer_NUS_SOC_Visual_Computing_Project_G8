"""Frame preprocessing helpers."""

from __future__ import annotations

import cv2

try:
    from . import config
except ImportError:
    import config


def to_gray(frame, use_clahe: bool):
    """Convert BGR frame to gray and optionally apply CLAHE."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not use_clahe:
        return gray, "gray"
    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE,
    )
    return clahe.apply(gray), "CLAHE"
