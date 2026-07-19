"""Drawing helpers for face boxes, landmarks, FPS, and status text."""

from __future__ import annotations

import cv2


FACE_COLOR = (0, 255, 0)
LANDMARK_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
WARNING_COLOR = (0, 220, 255)


def draw_faces(frame, faces) -> None:
    """Draw rectangles around detected faces."""
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), FACE_COLOR, 2)


def draw_landmarks(frame, landmarks) -> None:
    """Draw small circles at each landmark point."""
    for face_landmarks in landmarks:
        for point in face_landmarks[0]:
            x, y = int(point[0]), int(point[1])
            cv2.circle(frame, (x, y), 2, LANDMARK_COLOR, -1)


def draw_fps(frame, fps: float) -> None:
    """Draw FPS in the top-left corner."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )


def draw_status(frame, message: str, preprocessing_name: str) -> None:
    """Draw current preprocessing mode and status message."""
    cv2.putText(
        frame,
        f"Preprocess: {preprocessing_name}",
        (10, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        TEXT_COLOR,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        message,
        (10, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        WARNING_COLOR,
        2,
        cv2.LINE_AA,
    )
