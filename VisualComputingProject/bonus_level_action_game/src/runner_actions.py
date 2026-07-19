"""Upper-body gesture state machine for the three-lane runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque
from collections import deque

import numpy as np

from . import config
from .pose_types import PoseFrame


class Action(Enum):
    NONE = auto()
    LEFT = auto()
    RIGHT = auto()
    JUMP = auto()
    SLIDE = auto()


@dataclass
class CalibrationState:
    calibrated: bool = False
    center_x: float = 0.0
    center_y: float = 0.0
    left_shoulder_x: float = 0.0
    right_shoulder_x: float = 0.0
    shoulder_width: float = 1.0
    samples: list[tuple[float, float, float, float, float]] = field(default_factory=list)

    @property
    def progress(self) -> int:
        return len(self.samples)

    def reset(self):
        self.calibrated = False
        self.center_x = 0.0
        self.center_y = 0.0
        self.left_shoulder_x = 0.0
        self.right_shoulder_x = 0.0
        self.shoulder_width = 1.0
        self.samples.clear()


class GestureRecognizer:
    def __init__(self):
        self.calibration = CalibrationState()
        self.smoothed_keypoints: np.ndarray | None = None
        self.smoothed_confidences: np.ndarray | None = None
        self.wrist_history: Deque[tuple[float, float]] = deque(maxlen=8)
        self.body_history: Deque[tuple[float, float, float, float]] = deque(
            maxlen=max(2, config.BODY_VERTICAL_MOVE_HISTORY_FRAMES, config.BODY_RELATIVE_MOVE_HISTORY_FRAMES)
        )
        self.last_action_time = {action: -999.0 for action in Action}
        self.lane_locked = False
        self.jump_locked = False
        self.slide_locked = False
        self.left_count = 0
        self.right_count = 0
        self.jump_count = 0
        self.slide_count = 0
        self.last_pose_time = 0.0
        self.status = "Stand centered for calibration"
        self.debug = {}

    @property
    def calibrated(self) -> bool:
        return self.calibration.calibrated

    def reset_calibration(self):
        self.calibration.reset()
        self.smoothed_keypoints = None
        self.smoothed_confidences = None
        self.wrist_history.clear()
        self.body_history.clear()
        self.lane_locked = False
        self.jump_locked = False
        self.slide_locked = False
        self.left_count = self.right_count = self.jump_count = self.slide_count = 0
        self.status = "Stand centered for calibration"

    def update(self, frame: PoseFrame) -> Action:
        now = frame.timestamp
        if not self._valid_upper_pose(frame):
            if now - self.last_pose_time > config.POSE_LOST_GRACE_SECONDS:
                self.status = "No stable upper body; keyboard available"
                self._decay_counts()
            return Action.NONE

        self.last_pose_time = now
        keypoints, confidences = self._smooth(frame)
        metrics = self._metrics(frame, keypoints, confidences)
        if metrics is None:
            self.status = "Low pose confidence; keyboard available"
            self._decay_counts()
            return Action.NONE

        if not self.calibration.calibrated:
            self._collect_calibration(metrics)
            return Action.NONE

        self.debug = metrics
        self._update_body_history(metrics, now)
        self._update_resets(metrics)
        action = self._detect_jump_or_slide(metrics, now)
        if action is not Action.NONE:
            return action
        return self._detect_lane_action(metrics, now)

    def _smooth(self, frame: PoseFrame):
        keypoints = frame.keypoints.astype(np.float32)
        confidences = frame.confidences.astype(np.float32)
        if self.smoothed_keypoints is None:
            self.smoothed_keypoints = keypoints.copy()
            self.smoothed_confidences = confidences.copy()
        else:
            alpha = config.SMOOTHING_ALPHA
            self.smoothed_keypoints = alpha * keypoints + (1.0 - alpha) * self.smoothed_keypoints
            self.smoothed_confidences = alpha * confidences + (1.0 - alpha) * self.smoothed_confidences
        return self.smoothed_keypoints.copy(), self.smoothed_confidences.copy()

    def _valid_upper_pose(self, frame: PoseFrame) -> bool:
        required = [5, 6]
        return all(frame.valid_mask[idx] and frame.confidences[idx] >= config.POSE_CONFIDENCE_THRESHOLD for idx in required)

    def _metrics(self, frame: PoseFrame, kpts: np.ndarray, conf: np.ndarray):
        required = [5, 6]
        if any(conf[idx] < config.POSE_CONFIDENCE_THRESHOLD for idx in required):
            return None
        left_shoulder = kpts[5]
        right_shoulder = kpts[6]
        shoulder_width = float(np.linalg.norm(left_shoulder - right_shoulder))
        if shoulder_width < 12:
            return None
        shoulder_center = (left_shoulder + right_shoulder) / 2.0
        body_center_x = float(shoulder_center[0])
        if frame.bbox is not None:
            x1, _, x2, _ = frame.bbox
            body_center_x = float((x1 + x2) / 2.0)
        move_offset = (body_center_x - self.calibration.center_x) / max(self.calibration.shoulder_width, 1.0)
        vertical_offset = (float(shoulder_center[1]) - self.calibration.center_y) / max(self.calibration.shoulder_width, 1.0)
        left_shoulder_offset = (float(left_shoulder[0]) - self.calibration.left_shoulder_x) / max(self.calibration.shoulder_width, 1.0)
        right_shoulder_offset = (float(right_shoulder[0]) - self.calibration.right_shoulder_x) / max(self.calibration.shoulder_width, 1.0)
        return {
            "shoulder_center_x": float(shoulder_center[0]),
            "body_center_x": body_center_x,
            "shoulder_center_y": float(shoulder_center[1]),
            "body_center_y": float(shoulder_center[1]),
            "shoulder_width": shoulder_width,
            "left_shoulder_offset": left_shoulder_offset,
            "right_shoulder_offset": right_shoulder_offset,
            "left_shoulder": left_shoulder,
            "right_shoulder": right_shoulder,
            "move_offset": move_offset,
            "vertical_offset": vertical_offset,
            "relative_move": 0.0,
            "relative_vertical_move": 0.0,
        }

    def _update_body_history(self, metrics, now: float):
        self.body_history.append((now, metrics["body_center_x"], metrics["body_center_y"], metrics["shoulder_width"]))
        if len(self.body_history) < 2:
            metrics["relative_move"] = 0.0
            metrics["relative_vertical_move"] = 0.0
            return
        _, start_x, start_y, start_width = self.body_history[0]
        _, end_x, end_y, end_width = self.body_history[-1]
        width = max(1.0, (start_width + end_width) / 2.0)
        metrics["relative_move"] = (end_x - start_x) / width
        metrics["relative_vertical_move"] = (end_y - start_y) / width

    def _collect_calibration(self, metrics):
        self.calibration.samples.append(
            (
                metrics["body_center_x"],
                metrics["body_center_y"],
                metrics["shoulder_width"],
                float(metrics["left_shoulder"][0]),
                float(metrics["right_shoulder"][0]),
            )
        )
        needed = config.CALIBRATION_VALID_FRAMES
        self.status = f"Stand centered for calibration {self.calibration.progress}/{needed}"
        if self.calibration.progress >= needed:
            samples = np.array(self.calibration.samples, dtype=np.float32)
            self.calibration.center_x = float(np.mean(samples[:, 0]))
            self.calibration.center_y = float(np.mean(samples[:, 1]))
            self.calibration.shoulder_width = max(1.0, float(np.mean(samples[:, 2])))
            self.calibration.left_shoulder_x = float(np.mean(samples[:, 3]))
            self.calibration.right_shoulder_x = float(np.mean(samples[:, 4]))
            self.calibration.calibrated = True
            self.status = "Calibration complete; game started"

    def _update_resets(self, metrics):
        if abs(metrics.get("relative_move", 0.0)) <= config.BODY_RELATIVE_MOVE_RETURN_THRESHOLD:
            self.lane_locked = False
            self.left_count = 0
            self.right_count = 0
        if abs(metrics.get("relative_vertical_move", 0.0)) <= config.JUMP_BODY_RETURN_THRESHOLD and metrics["vertical_offset"] >= -config.JUMP_BODY_RETURN_THRESHOLD:
            self.jump_locked = False
            self.jump_count = 0
        if metrics["vertical_offset"] <= config.SLIDE_BODY_RETURN_THRESHOLD:
            self.slide_locked = False
            self.slide_count = 0

    def _detect_jump_or_slide(self, metrics, now: float) -> Action:
        vertical_move = metrics.get("relative_vertical_move", 0.0)
        vertical_offset = metrics["vertical_offset"]
        jump_raw = (
            vertical_move <= -config.JUMP_BODY_UP_TRIGGER_THRESHOLD
            and vertical_offset <= config.JUMP_BODY_ABOVE_NEUTRAL_THRESHOLD
            and not self.slide_locked
        )
        slide_raw = vertical_offset >= config.SLIDE_BODY_DOWN_TRIGGER_THRESHOLD or vertical_move >= config.SLIDE_BODY_DOWN_TRIGGER_THRESHOLD
        self.jump_count = self.jump_count + 1 if jump_raw else max(0, self.jump_count - 1)
        self.slide_count = self.slide_count + 1 if slide_raw else max(0, self.slide_count - 1)
        if slide_raw:
            self.jump_count = 0
        if (
            self.jump_count >= config.JUMP_CONFIRM_FRAMES
            and not self.jump_locked
            and now - self.last_action_time[Action.JUMP] >= config.JUMP_ACTION_COOLDOWN
        ):
            self.jump_locked = True
            self.last_action_time[Action.JUMP] = now
            self.status = f"Action: JUMP dy {vertical_move:+.2f}"
            return Action.JUMP
        if (
            self.slide_count >= config.SLIDE_CONFIRM_FRAMES
            and not self.slide_locked
            and now - self.last_action_time[Action.SLIDE] >= config.SLIDE_ACTION_COOLDOWN
        ):
            self.slide_locked = True
            self.last_action_time[Action.SLIDE] = now
            self.status = f"Action: SLIDE dy {vertical_offset:+.2f}"
            return Action.SLIDE
        return Action.NONE

    def _detect_lane_action(self, metrics, now: float) -> Action:
        if self.lane_locked:
            return Action.NONE
        movement = metrics.get("relative_move", 0.0)
        left_body_move = movement < -config.BODY_RELATIVE_MOVE_TRIGGER_THRESHOLD
        right_body_move = movement > config.BODY_RELATIVE_MOVE_TRIGGER_THRESHOLD
        self.left_count = self.left_count + 1 if left_body_move else max(0, self.left_count - 1)
        self.right_count = self.right_count + 1 if right_body_move else max(0, self.right_count - 1)
        if self.left_count >= config.BODY_MOVE_CONFIRM_FRAMES and now - self.last_action_time[Action.LEFT] >= config.LANE_ACTION_COOLDOWN:
            self.lane_locked = True
            self.last_action_time[Action.LEFT] = now
            self.status = "Action: LEFT"
            return Action.LEFT
        if self.right_count >= config.BODY_MOVE_CONFIRM_FRAMES and now - self.last_action_time[Action.RIGHT] >= config.LANE_ACTION_COOLDOWN:
            self.lane_locked = True
            self.last_action_time[Action.RIGHT] = now
            self.status = "Action: RIGHT"
            return Action.RIGHT
        self.status = f"Move dx {movement:+.2f} dy {metrics.get('relative_vertical_move', 0.0):+.2f}"
        return Action.NONE

    def _decay_counts(self):
        self.left_count = max(0, self.left_count - 1)
        self.right_count = max(0, self.right_count - 1)
        self.jump_count = max(0, self.jump_count - 1)
        self.slide_count = max(0, self.slide_count - 1)
