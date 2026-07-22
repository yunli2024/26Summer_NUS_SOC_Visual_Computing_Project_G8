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


def enhance_frame_for_detection(frame):
    """Enhance low-light BGR video before detection, landmarks, and display."""
    if not config.ENHANCE_VIDEO_FRAME:
        return frame, "off"

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    mean_y = float(np.mean(y))

    gamma_value = config.LOW_LIGHT_GAMMA_VALUE if mean_y < config.LOW_LIGHT_MEAN_THRESHOLD else config.GAMMA_VALUE
    enhanced_y = _apply_gamma(y, gamma=gamma_value)
    clahe = cv2.createCLAHE(
        clipLimit=config.LOW_LIGHT_CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE,
    )
    enhanced_y = clahe.apply(enhanced_y)
    enhanced_ycrcb = cv2.merge((enhanced_y, cr, cb))
    enhanced = cv2.cvtColor(enhanced_ycrcb, cv2.COLOR_YCrCb2BGR)
    enhanced = _unsharp_mask(enhanced, amount=config.LOW_LIGHT_SHARPEN_AMOUNT)

    alpha = float(config.LOW_LIGHT_BLEND_ALPHA)
    blended = cv2.addWeighted(enhanced, alpha, frame, 1.0 - alpha, 0)
    mode = f"video-lowlight-sharp(mean={mean_y:.0f})" if mean_y < config.LOW_LIGHT_MEAN_THRESHOLD else "video-enhanced-sharp"
    return blended, mode


def _apply_gamma(gray, gamma: float | None = None):
    gamma = config.GAMMA_VALUE if gamma is None else gamma
    gamma = max(float(gamma), 1e-6)
    inv_gamma = 1.0 / gamma
    table = np.asarray([((value / 255.0) ** inv_gamma) * 255 for value in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, table)


def _unsharp_mask(image, amount: float):
    amount = max(0.0, float(amount))
    if amount <= 0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
