"""Upper-body hand gestures for the camera-controlled platform game."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class GestureState:
    ready: bool
    horizontal: float
    jump: bool
    crouching: bool
    label: str
    left_active: bool
    right_active: bool
    both_up: bool
    wrist_gap: float
    left_score: float
    right_score: float
    coverage: float


class GestureController:
    """Recognize gestures using only shoulders, elbows, and wrists.

    Screen-space left/right is used after the camera frame is mirrored, so the
    controls feel like a mirror and do not depend on anatomical label order.
    """

    UPPER_JOINTS = np.asarray((5, 6, 7, 8, 9, 10), dtype=np.int32)
    MIN_ELBOW_ANGLE = 120.0

    def __init__(
        self,
        confirmation_frames: int = 3,
        release_frames: int = 4,
        jump_cooldown: float = 0.65,
    ) -> None:
        self.confirmation_frames = max(1, confirmation_frames)
        # Jump is an edge-triggered action and already has a latch/cooldown, so
        # it can use one fewer confirmation frame than continuous movement.
        self.jump_confirmation_frames = max(2, self.confirmation_frames - 1)
        self.release_frames = max(1, release_frames)
        self.jump_cooldown = jump_cooldown
        self.reset()

    def reset(self) -> None:
        self._left_streak = 0
        self._right_streak = 0
        self._jump_streak = 0
        self._crouch_streak = 0
        self._missing_streak = 0
        self._release_streak = 0
        self._direction = 0
        self._left_score = 0.0
        self._right_score = 0.0
        self._jump_latched = False
        self._last_jump_time = -math.inf
        self._joint_history = {int(index): deque(maxlen=3) for index in self.UPPER_JOINTS}

    @staticmethod
    def empty_state(label: str = "UPPER BODY NOT FOUND") -> GestureState:
        return GestureState(
            ready=False,
            horizontal=0.0,
            jump=False,
            crouching=False,
            label=label,
            left_active=False,
            right_active=False,
            both_up=False,
            wrist_gap=float("nan"),
            left_score=0.0,
            right_score=0.0,
            coverage=0.0,
        )

    @staticmethod
    def _advance_streak(current: int, active: bool) -> int:
        return current + 1 if active else 0

    @staticmethod
    def _joint_angle(first: np.ndarray, middle: np.ndarray, last: np.ndarray) -> float:
        first_ray = first - middle
        second_ray = last - middle
        denominator = float(np.linalg.norm(first_ray) * np.linalg.norm(second_ray))
        if denominator < 1e-6:
            return 0.0
        cosine = float(np.clip(np.dot(first_ray, second_ray) / denominator, -1.0, 1.0))
        return math.degrees(math.acos(cosine))

    def update(self, points: np.ndarray, valid: np.ndarray, timestamp: float) -> GestureState:
        points = np.asarray(points, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool) & np.isfinite(points).all(axis=1)
        if points.shape != (17, 2) or valid.shape != (17,):
            raise ValueError("Expected points (17, 2) and valid (17,).")

        coverage = float(np.count_nonzero(valid[self.UPPER_JOINTS]) / len(self.UPPER_JOINTS))
        if not (valid[5] and valid[6]) or not (valid[9] or valid[10]):
            self._missing_streak += 1
            self._left_streak = self._right_streak = 0
            self._jump_streak = self._crouch_streak = 0
            if self._missing_streak >= 2:
                self._jump_latched = False
                self._direction = 0
                self._left_score = self._right_score = 0.0
                self._release_streak = 0
                for history in self._joint_history.values():
                    history.clear()
            return self.empty_state()
        self._missing_streak = 0

        # A three-frame median rejects isolated pose-estimation spikes while
        # adding very little latency. Only currently valid joints are used for
        # decisions, so an old wrist cannot masquerade as a fresh detection.
        filtered = points.copy()
        for index in self.UPPER_JOINTS:
            joint = int(index)
            if valid[joint]:
                self._joint_history[joint].append(points[joint].copy())
                filtered[joint] = np.median(
                    np.stack(self._joint_history[joint], axis=0), axis=0
                )

        shoulder_points = filtered[[5, 6]]
        shoulder_left_x = float(np.min(shoulder_points[:, 0]))
        shoulder_right_x = float(np.max(shoulder_points[:, 0]))
        shoulder_y = float(np.mean(shoulder_points[:, 1]))
        shoulder_center_x = float(np.mean(shoulder_points[:, 0]))
        scale = float(abs(shoulder_points[0, 0] - shoulder_points[1, 0]))
        if scale < 20.0:
            return self.empty_state("MOVE CLOSER")

        # Pair each wrist with its own elbow angle, then sort by screen X. This
        # keeps mirrored controls intuitive and rejects a bent arm whose wrist
        # briefly jitters beyond the shoulder.
        wrists = []
        for shoulder, elbow, wrist in ((5, 7, 9), (6, 8, 10)):
            if valid[wrist]:
                elbow_angle = (
                    self._joint_angle(filtered[shoulder], filtered[elbow], filtered[wrist])
                    if valid[shoulder] and valid[elbow]
                    else 0.0
                )
                wrists.append((filtered[wrist], elbow_angle))
        wrists.sort(key=lambda item: float(item[0][0]))
        leftmost, left_elbow_angle = wrists[0]
        rightmost, right_elbow_angle = wrists[-1]
        usable_height = shoulder_y + 0.35 * scale
        left_extension = (shoulder_left_x - float(leftmost[0])) / scale
        right_extension = (float(rightmost[0]) - shoulder_right_x) / scale
        raw_left_score = (
            float(np.clip((left_extension - 0.18) / 0.62, 0.0, 1.0))
            if leftmost[1] < usable_height and left_elbow_angle >= self.MIN_ELBOW_ANGLE else 0.0
        )
        raw_right_score = (
            float(np.clip((right_extension - 0.18) / 0.62, 0.0, 1.0))
            if rightmost[1] < usable_height and right_elbow_angle >= self.MIN_ELBOW_ANGLE else 0.0
        )
        # EMA plus hysteresis prevents a wrist near the threshold from rapidly
        # switching between movement and neutral.
        self._left_score = 0.58 * raw_left_score + 0.42 * self._left_score
        self._right_score = 0.58 * raw_right_score + 0.42 * self._right_score
        raw_left = self._left_score > 0.28
        raw_right = self._right_score > 0.28

        two_wrists = valid[9] and valid[10]
        both_up = bool(
            two_wrists
            and filtered[9, 1] < shoulder_y - 0.25 * scale
            and filtered[10, 1] < shoulder_y - 0.25 * scale
        )

        wrist_gap = float("nan")
        raw_crouch = False
        if two_wrists:
            wrist_midpoint = (filtered[9] + filtered[10]) * 0.5
            wrist_gap = float(np.linalg.norm(filtered[9] - filtered[10]) / scale)
            hands_near_chest = (
                abs(float(wrist_midpoint[0]) - shoulder_center_x) < 0.50 * scale
                and shoulder_y + 0.08 * scale < wrist_midpoint[1] < shoulder_y + 1.05 * scale
            )
            raw_crouch = bool(wrist_gap < 0.55 and hands_near_chest)

        # Jump preserves the current direction, allowing a running jump. A
        # crouch deliberately stops horizontal movement.
        self._jump_streak = self._advance_streak(self._jump_streak, both_up)
        self._crouch_streak = self._advance_streak(self._crouch_streak, raw_crouch and not both_up)
        crouching = self._crouch_streak >= self.confirmation_frames

        if both_up:
            # Do not clear the direction while changing from one raised hand to
            # two raised hands for a jump.
            self._release_streak = 0
        elif raw_crouch:
            self._direction = 0
            self._left_streak = self._right_streak = 0
            self._release_streak = 0
        elif raw_left and not raw_right:
            self._left_streak += 1
            self._right_streak = 0
            self._release_streak = 0
            if self._left_streak >= self.confirmation_frames:
                self._direction = -1
        elif raw_right and not raw_left:
            self._right_streak += 1
            self._left_streak = 0
            self._release_streak = 0
            if self._right_streak >= self.confirmation_frames:
                self._direction = 1
        else:
            self._left_streak = self._right_streak = 0
            self._release_streak += 1
            if self._release_streak >= self.release_frames:
                self._direction = 0

        left_active = self._direction < 0
        right_active = self._direction > 0

        confirmed_up = self._jump_streak >= self.jump_confirmation_frames
        jump = bool(
            confirmed_up
            and not self._jump_latched
            and timestamp - self._last_jump_time >= self.jump_cooldown
        )
        if jump:
            self._jump_latched = True
            self._last_jump_time = timestamp
        if not both_up:
            self._jump_latched = False

        if jump:
            label = "JUMP"
        elif confirmed_up:
            label = "HANDS UP"
        elif crouching:
            label = "CROUCH"
        elif left_active:
            label = "LEFT"
        elif right_active:
            label = "RIGHT"
        else:
            label = "NEUTRAL"

        if left_active:
            horizontal = -(0.55 + 0.45 * self._left_score)
        elif right_active:
            horizontal = 0.55 + 0.45 * self._right_score
        else:
            horizontal = 0.0
        return GestureState(
            ready=True,
            horizontal=horizontal,
            jump=jump,
            crouching=crouching,
            label=label,
            left_active=left_active,
            right_active=right_active,
            both_up=both_up,
            wrist_gap=wrist_gap,
            left_score=self._left_score,
            right_score=self._right_score,
            coverage=coverage,
        )
