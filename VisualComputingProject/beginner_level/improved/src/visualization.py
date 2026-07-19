"""Visualization helpers for the improved Beginner Level."""

from __future__ import annotations

import cv2


FACE_COLOR = (0, 255, 0)
FALLBACK_FACE_COLOR = (0, 220, 255)
LOST_COLOR = (0, 0, 255)
LANDMARK_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)


def draw_faces(frame, faces, status: str) -> None:
    color = FALLBACK_FACE_COLOR if status == "CACHED" else FACE_COLOR
    for x, y, w, h in faces:
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)


def draw_landmarks(frame, landmarks) -> None:
    for points in landmarks:
        for x, y in points:
            cv2.circle(frame, (int(x), int(y)), 2, LANDMARK_COLOR, -1)


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
