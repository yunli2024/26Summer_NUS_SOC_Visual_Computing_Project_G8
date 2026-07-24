"""Visualization helpers for the improved Beginner Level."""

from __future__ import annotations

import cv2

try:
    from . import config
except ImportError:
    import config


FACE_COLOR = (0, 255, 0)
FALLBACK_FACE_COLOR = (0, 220, 255)
LOST_COLOR = (0, 0, 255)
LANDMARK_COLOR = (0, 0, 255)
LANDMARK_GROUPS = [
    (range(0, 17), (255, 180, 0), "jaw"),
    (range(17, 27), (0, 220, 255), "brow"),
    (range(27, 36), (255, 0, 255), "nose"),
    (range(36, 48), (0, 255, 255), "eye"),
    (range(48, 68), (0, 0, 255), "mouth"),
]
LANDMARK_LABEL_INDICES = {0, 8, 16, 27, 30, 36, 45, 48, 54, 66}
TEXT_COLOR = (255, 255, 255)


def draw_faces(frame, faces, status: str, pose_labels=None) -> None:
    color = FALLBACK_FACE_COLOR if status == "CACHED" else FACE_COLOR
    if status == "REJECTED":
        color = LOST_COLOR
    pose_labels = pose_labels or []
    for index, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)
        if config.SHOW_FACE_POSE_LABELS and index < len(pose_labels):
            cv2.putText(
                frame,
                pose_labels[index],
                (int(x), max(18, int(y) - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                2,
                cv2.LINE_AA,
            )


def draw_landmarks(frame, landmarks) -> None:
    for points in landmarks:
        for point_index, (x, y) in enumerate(points):
            x, y = int(x), int(y)
            cv2.circle(frame, (x, y), 2, _landmark_color(point_index), -1)
            if point_index in LANDMARK_LABEL_INDICES:
                cv2.putText(
                    frame,
                    str(point_index),
                    (x + 3, y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    TEXT_COLOR,
                    1,
                    cv2.LINE_AA,
                )


def _landmark_color(point_index: int):
    for indices, color, _name in LANDMARK_GROUPS:
        if point_index in indices:
            return color
    return LANDMARK_COLOR


def draw_status(
    frame,
    fps: float,
    status: str,
    raw_count: int,
    filtered_count: int,
    selected_size,
    clahe_enabled: bool,
    failed_frames: int,
    message: str,
    preprocess_name: str = "",
    video_preprocess_name: str = "",
) -> None:
    face_size = "none" if selected_size is None else f"{selected_size[0]}x{selected_size[1]}"
    preprocess_text = preprocess_name or ("clahe" if clahe_enabled else "gray")
    lines = [
        f"STATE: {status}",
        f"FPS: {fps:.1f}",
        f"Faces: {filtered_count}/{raw_count}",
        f"Box: {face_size}",
        f"Mode: {preprocess_text}  video: {video_preprocess_name}",
        f"Miss: {failed_frames}",
    ]
    status_color = FACE_COLOR if status == "DETECTED" else FALLBACK_FACE_COLOR if status == "CACHED" else LOST_COLOR
    panel_x, panel_y = 8, 8
    panel_w = 290
    panel_h = 22 + 22 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, frame)
    y = panel_y + 24
    for line in lines:
        color = status_color if line.startswith("STATE:") else TEXT_COLOR
        cv2.putText(frame, line, (panel_x + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
        y += 22
