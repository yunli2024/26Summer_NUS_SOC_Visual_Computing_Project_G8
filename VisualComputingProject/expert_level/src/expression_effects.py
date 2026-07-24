from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Iterable, List, Sequence

import cv2
import numpy as np

from keypoint_features import FaceFeatures


@dataclass(frozen=True)
class ExpressionPrediction:
    face: FaceFeatures
    label: str
    confidence: float
    prediction_ms: float


class LabelSmoother:
    def __init__(self, window: int = 5) -> None:
        self._labels: Deque[str] = deque(maxlen=max(1, window))

    def update(self, label: str) -> str:
        self._labels.append(label)
        counts = Counter(self._labels)
        return counts.most_common(1)[0][0]


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    scores = scores - np.max(scores)
    exp = np.exp(scores)
    return exp / max(float(np.sum(exp)), 1e-9)


def draw_expression_label(frame_bgr: np.ndarray, prediction: ExpressionPrediction) -> None:
    x, y, w, h = prediction.face.box
    text = f"{prediction.label} {prediction.confidence * 100:.0f}% ({prediction.prediction_ms:.2f} ms)"
    y_text = max(24, y - 10)
    color = (210, 210, 210) if prediction.label == "uncertain" else (255, 255, 255)
    cv2.putText(frame_bgr, text, (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame_bgr, text, (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 1, cv2.LINE_AA)


def apply_expression_effect(frame_bgr: np.ndarray, label: str, frame_index: int) -> None:
    if label == "happy":
        draw_sparkles(frame_bgr, frame_index)
    elif label == "surprise":
        draw_burst(frame_bgr, frame_index)
    elif label == "angry":
        apply_tint(frame_bgr, (0, 0, 90), alpha=0.18)
    elif label == "sad":
        apply_tint(frame_bgr, (90, 40, 0), alpha=0.18)
    elif label == "neutral":
        draw_neutral_border(frame_bgr)


def apply_tint(frame_bgr: np.ndarray, color_bgr: tuple[int, int, int], alpha: float) -> None:
    overlay = np.full_like(frame_bgr, color_bgr, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0, dst=frame_bgr)


def draw_neutral_border(frame_bgr: np.ndarray) -> None:
    height, width = frame_bgr.shape[:2]
    cv2.rectangle(frame_bgr, (8, 8), (width - 8, height - 8), (200, 200, 200), 2, cv2.LINE_AA)


def draw_sparkles(frame_bgr: np.ndarray, frame_index: int) -> None:
    height, width = frame_bgr.shape[:2]
    points = [
        (0.16, 0.20),
        (0.82, 0.18),
        (0.24, 0.72),
        (0.74, 0.70),
        (0.50, 0.12),
    ]
    for idx, (rx, ry) in enumerate(points):
        phase = (frame_index + idx * 9) * 0.15
        radius = int(6 + 3 * (1.0 + math.sin(phase)))
        cx = int(width * rx)
        cy = int(height * ry)
        color = (30, 255, 255) if idx % 2 else (80, 255, 120)
        cv2.line(frame_bgr, (cx - radius, cy), (cx + radius, cy), color, 2, cv2.LINE_AA)
        cv2.line(frame_bgr, (cx, cy - radius), (cx, cy + radius), color, 2, cv2.LINE_AA)
        cv2.circle(frame_bgr, (cx, cy), max(2, radius // 3), color, -1, cv2.LINE_AA)


def draw_burst(frame_bgr: np.ndarray, frame_index: int) -> None:
    height, width = frame_bgr.shape[:2]
    center = (width // 2, height // 2)
    radius = 34 + int(10 * math.sin(frame_index * 0.2))
    for idx in range(12):
        angle = 2.0 * math.pi * idx / 12.0 + frame_index * 0.02
        start = (
            int(center[0] + math.cos(angle) * radius),
            int(center[1] + math.sin(angle) * radius),
        )
        end = (
            int(center[0] + math.cos(angle) * (radius + 34)),
            int(center[1] + math.sin(angle) * (radius + 34)),
        )
        cv2.line(frame_bgr, start, end, (255, 255, 40), 2, cv2.LINE_AA)


def draw_predictions(frame_bgr: np.ndarray, predictions: Sequence[ExpressionPrediction]) -> None:
    for prediction in predictions:
        draw_expression_label(frame_bgr, prediction)
