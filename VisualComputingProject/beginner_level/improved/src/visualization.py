"""Visualization helpers for the improved Beginner Level."""

from __future__ import annotations

import cv2


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


def draw_faces(frame, faces, status: str) -> None:
    color = FALLBACK_FACE_COLOR if status == "CACHED" else FACE_COLOR
    for x, y, w, h in faces:
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)


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
) -> None:
    face_size = "none" if selected_size is None else f"{selected_size[0]}x{selected_size[1]}"
    lines = [
        f"STATE: {status}",
        f"FPS: {fps:.1f}",
        f"Candidates raw/kept: {raw_count}/{filtered_count}",
        f"Face box size: {face_size}",
        "Keypoints: 68 LBF points, colored by facial region",
        f"CLAHE: {clahe_enabled}",
        f"Failed frames: {failed_frames}",
        message,
    ]
    y = 25
    status_color = FACE_COLOR if status == "DETECTED" else FALLBACK_FACE_COLOR if status == "CACHED" else LOST_COLOR
    for line in lines:
        color = status_color if line.startswith("STATE:") else TEXT_COLOR
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2, cv2.LINE_AA)
        y += 26
