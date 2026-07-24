"""One-way reaction-delay search with short-window motion comparison."""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

from . import config
from .pose_similarity import (
    HoldStateFilter,
    apply_motion_score,
    compare_poses,
)
from .pose_types import PoseFrame


class TemporalAligner:
    """Match the current player to a recent reference frame.

    A non-negative lag means the player is reacting after the displayed
    reference.  Future reference frames are never considered.
    """

    def __init__(self):
        self.reference_frames: Deque[PoseFrame] = deque()
        self.user_frames: Deque[PoseFrame] = deque(
            maxlen=config.USER_BUFFER_FRAME_COUNT
        )
        self.hold_filter = HoldStateFilter()

    def reset(self):
        self.reference_frames.clear()
        self.user_frames.clear()
        self.hold_filter.reset()

    def add_reference(self, frame: PoseFrame):
        self.reference_frames.append(frame)
        latest = frame.timestamp
        while (
            self.reference_frames
            and latest - self.reference_frames[0].timestamp
            > config.REFERENCE_BUFFER_SECONDS
        ):
            self.reference_frames.popleft()

    def match(
        self,
        user_frame: PoseFrame,
    ) -> Tuple[Optional[PoseFrame], dict, float]:
        if not self.reference_frames:
            self.user_frames.append(user_frame)
            return None, {"score": 0.0, "feedback": "Load reference"}, 0.0

        candidates = [
            reference
            for reference in self.reference_frames
            if 0.0
            <= user_frame.timestamp - reference.timestamp
            <= config.ALIGNMENT_WINDOW_SECONDS
        ]
        if not candidates:
            candidates = [self.reference_frames[-1]]

        previous_user = closest_earlier_frame(
            self.user_frames,
            user_frame.timestamp - config.MOTION_WINDOW_SECONDS,
        )
        best_frame: Optional[PoseFrame] = None
        best_result: Optional[dict] = None
        best_lag = 0.0
        for reference in candidates:
            result = compare_poses(reference, user_frame)
            if previous_user is not None:
                motion_age = user_frame.timestamp - previous_user.timestamp
                previous_reference = closest_earlier_frame(
                    self.reference_frames,
                    reference.timestamp - motion_age,
                )
                if (
                    previous_reference is not None
                    and previous_reference.frame_index < reference.frame_index
                    and motion_age >= 0.10
                ):
                    result = apply_motion_score(
                        result,
                        previous_reference,
                        reference,
                        previous_user,
                        user_frame,
                    )
            result_rank = (
                int(bool(result.get("motion_used", False)))
                if previous_user is not None
                else 0,
                float(result["score"]),
            )
            best_rank = (
                (
                    int(bool(best_result.get("motion_used", False))),
                    float(best_result["score"]),
                )
                if best_result is not None
                else (-1, -1.0)
            )
            if best_result is None or result_rank > best_rank:
                best_frame = reference
                best_result = result
                best_lag = max(0.0, user_frame.timestamp - reference.timestamp)

        self.user_frames.append(user_frame)
        assert best_result is not None
        assert best_frame is not None
        reference_is_holding = self.hold_filter.update(
            float(best_result.get("reference_motion", 0.0)),
            bool(best_result.get("motion_used", False)),
        )
        if not bool(best_result.get("motion_used", False)):
            best_result["feedback"] = "Sync"
            best_result["score_event"] = False
        elif reference_is_holding:
            best_result["feedback"] = "Hold"
            best_result["score_event"] = False
        elif float(best_result.get("player_motion", 0.0)) < max(
            0.06,
            0.45 * float(best_result.get("reference_motion", 0.0)),
        ):
            best_result["feedback"] = "Move!"
            best_result["score_event"] = True

        best_result.update(
            {
                "matched_reference_frame": best_frame.frame_index,
                "matched_user_frame": user_frame.frame_index,
                "user_buffer_size": len(self.user_frames),
                "lag_seconds": best_lag,
                "hold_reference_motion": self.hold_filter.smoothed_motion,
            }
        )
        return best_frame, best_result, best_lag


def closest_earlier_frame(
    frames: Deque[PoseFrame],
    target_timestamp: float,
) -> Optional[PoseFrame]:
    eligible = [frame for frame in frames if frame.timestamp <= target_timestamp]
    if not eligible:
        return None
    return min(eligible, key=lambda frame: abs(frame.timestamp - target_timestamp))
