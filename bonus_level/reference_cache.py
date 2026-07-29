"""Portable reference-video playback backed by a compact pose cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pose_analyzer import draw_pose


def load_pose_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Reference pose cache not found: {path}\n"
            "Run pose_analyzer.py first to generate pose_cache.npz."
        )
    with np.load(path, allow_pickle=False) as cache:
        required = {"points", "valid", "playback_fps"}
        missing = required.difference(cache.files)
        if missing:
            raise ValueError(f"Invalid pose cache; missing: {', '.join(sorted(missing))}")
        points = cache["points"].astype(np.float32)
        valid = cache["valid"].astype(bool)
        playback_fps = float(np.asarray(cache["playback_fps"]).reshape(()))
        boxes = (
            cache["boxes"].astype(np.float32)
            if "boxes" in cache.files
            else _derive_boxes(points, valid)
        )
        source_frames = (
            cache["source_frames"].astype(np.int32)
            if "source_frames" in cache.files
            else np.arange(len(points), dtype=np.int32)
        )

    if points.ndim != 3 or points.shape[1:] != (17, 2) or valid.shape != points.shape[:2]:
        raise ValueError("Invalid pose cache dimensions.")
    if boxes.shape != (len(points), 4):
        raise ValueError("Invalid pose-cache boxes.")
    if source_frames.shape != (len(points),) or np.any(source_frames < 0):
        raise ValueError("Invalid pose-cache source-frame mapping.")
    if playback_fps <= 0:
        raise ValueError("Invalid playback FPS in pose cache.")
    return {
        "points": points,
        "valid": valid,
        "boxes": boxes,
        "source_frames": source_frames,
        "playback_fps": playback_fps,
    }


def _derive_boxes(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    boxes = np.zeros((len(points), 4), dtype=np.float32)
    for index, (frame_points, frame_valid) in enumerate(zip(points, valid)):
        visible = frame_points[frame_valid]
        if len(visible) == 0:
            continue
        x1, y1 = np.min(visible, axis=0)
        x2, y2 = np.max(visible, axis=0)
        padding = max(8.0, 0.08 * max(float(x2 - x1), float(y2 - y1)))
        boxes[index] = (x1 - padding, y1 - padding, x2 + padding, y2 + padding)
    return boxes


def read_cached_reference_frame(
    capture: cv2.VideoCapture,
    cache: dict[str, Any],
    index: int,
    *,
    label: str = "REFERENCE",
) -> np.ndarray:
    points = cache["points"]
    if index < 0 or index >= len(points):
        raise IndexError(f"Pose-cache frame out of range: {index}")
    source_index = int(cache["source_frames"][index])
    capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read source-video frame {source_index}.")

    valid = cache["valid"][index]
    if np.any(valid):
        draw_pose(
            frame,
            points[index],
            valid,
            cache["boxes"][index],
            0.0,
            label_prefix=label,
            show_confidence=False,
        )
    return frame
