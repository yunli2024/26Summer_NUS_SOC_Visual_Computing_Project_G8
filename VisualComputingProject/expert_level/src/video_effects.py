"""Simple visual effects for realtime expression demo."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


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

PHOTO_DIR = Path(__file__).resolve().parents[1] / "data" / "photo"
EXPRESSION_PHOTOS = {
    "angry": "angry.png",
    "disgust": "disgust.png",
    "fear": "fear.png",
    "happy": "happy.png",
    "neutral": "neural.png",
    "sad": "sad.png",
    "surprise": "surprise.png",
}
EXPRESSION_TEXT = {
    "angry": "Angry",
    "disgust": "Disgust",
    "fear": "Fear",
    "happy": "Happy",
    "neutral": "Neutral",
    "sad": "Sad",
    "surprise": "Surprise",
    "uncertain": "Uncertain",
}
FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\arialbd.ttf"),
    Path(r"C:\Windows\Fonts\bahnschrift.ttf"),
    Path(r"C:\Windows\Fonts\arial.ttf"),
]


def draw_effect(frame, expression: str) -> None:
    return None


def draw_landmarks(frame, landmarks, color=(0, 0, 255)) -> None:
    for index, (x, y) in enumerate(landmarks):
        if index < 17:
            continue
        cv2.circle(frame, (int(x), int(y)), 2, color, -1)


@lru_cache(maxsize=16)
def _load_expression_photo(label: str):
    filename = EXPRESSION_PHOTOS.get(label)
    if not filename:
        return None
    path = PHOTO_DIR / filename
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_UNCHANGED)


@lru_cache(maxsize=8)
def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _overlay_image(frame, image, x: int, y: int, size: int) -> tuple[int, int, int, int] | None:
    if image is None or size <= 0:
        return None
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    height, width = frame.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + size)
    y2 = min(height, y + size)
    if x1 >= x2 or y1 >= y2:
        return None

    crop = resized[y1 - y : y2 - y, x1 - x : x2 - x]
    if crop.shape[2] == 4:
        alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
        rgb = crop[:, :, :3].astype(np.float32)
        base = frame[y1:y2, x1:x2].astype(np.float32)
        frame[y1:y2, x1:x2] = (alpha * rgb + (1.0 - alpha) * base).astype(np.uint8)
    else:
        frame[y1:y2, x1:x2] = crop
    return x1, y1, x2, y2


def _text_size(text: str, font) -> tuple[int, int]:
    box = ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def _draw_text(frame, text: str, x: int, y: int, color, font_size: int = 26) -> None:
    font = _load_font(font_size)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    b, g, r = [int(channel) for channel in color]
    draw.text((x, y), text, font=font, fill=(r, g, b))
    frame[:] = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


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
    display_text = EXPRESSION_TEXT.get(label, label)
    confidence_text = "n/a" if confidence is None else f"{confidence:.2f}"
    title = f"{display_text} {confidence_text}"
    font = _load_font(26)
    text_width, text_height = _text_size(title, font)
    photo_size = max(72, min(150, int(max(w, h) * 0.45)))
    badge_width = max(photo_size, text_width) + 18
    badge_height = photo_size + text_height + 22
    badge_x = x + w + 10
    if badge_x + badge_width >= width:
        badge_x = x - badge_width - 10
    badge_x = max(0, min(badge_x, width - badge_width - 1))
    badge_y = max(0, min(y, height - badge_height - 1))

    cv2.rectangle(
        frame,
        (badge_x, badge_y),
        (badge_x + badge_width, badge_y + badge_height),
        (0, 0, 0),
        -1,
    )
    cv2.rectangle(
        frame,
        (badge_x, badge_y),
        (badge_x + badge_width, badge_y + badge_height),
        color,
        2,
    )
    photo = _load_expression_photo(label)
    photo_x = badge_x + (badge_width - photo_size) // 2
    photo_y = badge_y + 8
    if _overlay_image(frame, photo, photo_x, photo_y, photo_size) is None:
        cv2.circle(frame, (photo_x + photo_size // 2, photo_y + photo_size // 2), photo_size // 3, color, 3)

    text_x = badge_x + (badge_width - text_width) // 2
    text_y = photo_y + photo_size + 7
    _draw_text(frame, title, text_x, text_y, color, 26)

    detail_lines = []
    if status != "TRACKED":
        detail_lines.append(status)
    if pose_label:
        detail_lines.append(pose_label)
    for index, line in enumerate(detail_lines[:2]):
        cv2.putText(
            frame,
            line,
            (badge_x + 8, min(height - 8, badge_y + badge_height + 18 + index * 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


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
