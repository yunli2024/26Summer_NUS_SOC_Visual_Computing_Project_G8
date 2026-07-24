"""Geometry helpers for face boxes."""

from __future__ import annotations

import numpy as np


def expand_face_box(
    box,
    image_width: int,
    image_height: int,
    expand_x: float = 0.15,
    expand_top: float = 0.18,
    expand_bottom: float = 0.15,
) -> np.ndarray:
    """Expand a face box and clamp it inside the image boundary."""

    x, y, w, h = [float(value) for value in box]
    new_x = x - w * expand_x
    new_y = y - h * expand_top
    new_w = w * (1.0 + 2.0 * expand_x)
    new_h = h * (1.0 + expand_top + expand_bottom)

    x1 = max(0, int(round(new_x)))
    y1 = max(0, int(round(new_y)))
    x2 = min(image_width, int(round(new_x + new_w)))
    y2 = min(image_height, int(round(new_y + new_h)))

    if x2 <= x1 or y2 <= y1:
        return np.asarray(box, dtype=np.int32)
    return np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=np.int32)


def face_center(face) -> np.ndarray:
    """Return the center point of a face box."""

    x, y, w, h = [float(value) for value in face]
    return np.asarray([x + w / 2.0, y + h / 2.0], dtype=np.float32)


def face_area(face) -> float:
    """Return face box area."""

    _x, _y, w, h = [float(value) for value in face]
    return max(w, 0.0) * max(h, 0.0)


def face_iou(a, b) -> float:
    """Return intersection-over-union for two face boxes."""

    ax, ay, aw, ah = [float(value) for value in a]
    bx, by, bw, bh = [float(value) for value in b]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter_w = max(0.0, min(ax2, bx2) - max(ax, bx))
    inter_h = max(0.0, min(ay2, by2) - max(ay, by))
    inter = inter_w * inter_h
    union = face_area(a) + face_area(b) - inter
    return inter / union if union > 1e-6 else 0.0


def face_inside(inner, outer) -> bool:
    """Return True when one face box is fully inside another."""

    ix, iy, iw, ih = [float(value) for value in inner]
    ox, oy, ow, oh = [float(value) for value in outer]
    return ix >= ox and iy >= oy and ix + iw <= ox + ow and iy + ih <= oy + oh
