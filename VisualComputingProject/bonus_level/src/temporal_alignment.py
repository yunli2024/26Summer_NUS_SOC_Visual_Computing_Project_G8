"""Sliding-window temporal alignment for realtime scoring."""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

from . import config
from .pose_similarity import compare_poses
from .pose_types import PoseFrame


class TemporalAligner:
    def __init__(self):
        self.reference_frames: Deque[PoseFrame] = deque()

    def reset(self):
        self.reference_frames.clear()

    def add_reference(self, frame: PoseFrame):
        self.reference_frames.append(frame)
        latest = frame.timestamp
        while self.reference_frames and latest - self.reference_frames[0].timestamp > config.REFERENCE_BUFFER_SECONDS:
            self.reference_frames.popleft()

    def match(self, user_frame: PoseFrame) -> Tuple[Optional[PoseFrame], dict, float]:
        if not self.reference_frames:
            return None, {"score": 0.0, "feedback": "Load reference"}, 0.0
        best_frame = None
        best_result = None
        best_offset = 0.0
        for ref in self.reference_frames:
            offset = user_frame.timestamp - ref.timestamp
            if abs(offset) > config.ALIGNMENT_WINDOW_SECONDS:
                continue
            result = compare_poses(ref, user_frame)
            if best_result is None or float(result["score"]) > float(best_result["score"]):
                best_frame = ref
                best_result = result
                best_offset = offset
        if best_result is None:
            ref = self.reference_frames[-1]
            best_result = compare_poses(ref, user_frame)
            best_frame = ref
            best_offset = user_frame.timestamp - ref.timestamp
        return best_frame, best_result, best_offset
