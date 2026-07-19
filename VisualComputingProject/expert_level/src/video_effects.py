"""Simple visual effects for realtime expression demo."""

from __future__ import annotations

import cv2
import numpy as np


EFFECT_COLORS = {
    "happy": (0, 255, 255),
    "sad": (255, 120, 120),
    "angry": (0, 0, 255),
    "surprise": (255, 255, 255),
    "fear": (180, 80, 255),
    "disgust": (0, 160, 0),
    "neutral": (180, 220, 180),
    "uncertain": (170, 170, 170),
}

TRACK_COLORS = [
    (0, 255, 0),
    (0, 220, 255),
    (255, 160, 80),
    (220, 120, 255),
    (120, 220, 120),
]


def draw_effect(frame, expression: str) -> None:
    color = EFFECT_COLORS.get(expression, (255, 255, 255))
    height, width = frame.shape[:2]
    if expression == "happy":
        for x in range(40, width, 110):
            cv2.putText(frame, "*", (x, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)
    elif expression == "sad":
        for x in range(30, width, 85):
            cv2.line(frame, (x, 25), (x - 10, 60), color, 2)
    elif expression == "angry":
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), color, 8)
        cv2.putText(frame, "!", (width - 80, 85), cv2.FONT_HERSHEY_SIMPLEX, 2.3, color, 5)
    elif expression == "surprise":
        cv2.putText(frame, "!", (width - 80, 95), cv2.FONT_HERSHEY_SIMPLEX, 3.0, color, 6)
    elif expression == "neutral":
        cv2.rectangle(frame, (8, 8), (width - 9, height - 9), color, 2)
    elif expression == "uncertain":
        cv2.rectangle(frame, (8, 8), (width - 9, height - 9), color, 1)


def draw_landmarks(frame, landmarks, color=(0, 0, 255)) -> None:
    jaw_color = tuple(int(channel * 0.55) for channel in color)
    jaw_points = [(int(x), int(y)) for x, y in landmarks[:17]]
    if len(jaw_points) >= 2:
        cv2.polylines(frame, [np.asarray(jaw_points, dtype=np.int32)], False, jaw_color, 1, cv2.LINE_AA)
    for index, (x, y) in enumerate(landmarks):
        if index < 17:
            cv2.circle(frame, (int(x), int(y)), 1, jaw_color, -1)
        else:
            cv2.circle(frame, (int(x), int(y)), 2, color, -1)


def draw_face_label(
    frame,
    face,
    label: str,
    confidence,
    fps: float,
    color,
    status: str = "TRACKED",
    pose_label: str = "",
) -> None:
    x, y, w, h = [int(v) for v in face]
    height, width = frame.shape[:2]
    confidence_text = "n/a" if confidence is None else f"{confidence:.2f}"
    lines = [f"{label} {confidence_text}"]
    if status != "TRACKED":
        lines.append(status)
    if pose_label:
        lines.append(pose_label)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    line_height = 22
    text_width = max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines)
    block_height = line_height * len(lines) + 6
    label_x = max(0, min(x, width - text_width - 12))
    label_y = y - block_height - 4
    if label_y < 0:
        label_y = min(y + h + 6, height - block_height - 1)
    label_y = max(0, label_y)
    cv2.rectangle(
        frame,
        (label_x, label_y),
        (min(width - 1, label_x + text_width + 10), min(height - 1, label_y + block_height)),
        (0, 0, 0),
        -1,
    )
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text_y = label_y + 19
    for line in lines:
        cv2.putText(frame, line, (label_x + 5, text_y), font, scale, color, thickness, cv2.LINE_AA)
        text_y += line_height


def draw_hud(
    frame,
    raw_expression: str,
    voted_expression: str,
    stable_expression: str,
    confidence,
    margin,
    fps: float,
    message: str,
    queue_text: str = "",
    stability_status: str = "",
) -> None:
    confidence_text = "n/a" if confidence is None else f"{confidence:.2f}"
    margin_text = "n/a" if margin is None else f"{margin:.2f}"
    lines = [
        f"Raw: {raw_expression}",
        f"Vote: {voted_expression}",
        f"Stable: {stable_expression}",
        f"Confidence: {confidence_text}",
        f"Margin: {margin_text}",
        f"State: {stability_status}",
        f"Queue: {queue_text}",
        f"FPS: {fps:.1f}",
        message,
    ]
    y = 28
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        y += 28


def draw_debug_hud(frame, track_debug: list[str]) -> None:
    y = 24
    for line in track_debug[:10]:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        y += 22
