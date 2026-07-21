"""Drawing helpers for face boxes, landmarks, FPS, and status text."""

from __future__ import annotations

import cv2


FACE_COLOR = (0, 255, 0)
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
WARNING_COLOR = (0, 220, 255)


def draw_faces(frame, faces) -> None:
    """Draw rectangles around detected faces."""
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), FACE_COLOR, 2)


def draw_landmarks(frame, landmarks) -> None:
    """Draw and label the 68 facial keypoints."""
    for face_landmarks in landmarks:
        points = face_landmarks[0]
        for point_index, point in enumerate(points):
            x, y = int(point[0]), int(point[1])
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
