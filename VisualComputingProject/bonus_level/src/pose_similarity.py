"""Motion-aware pose similarity adapted from Zhangyx Part Three."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from . import config
from .pose_types import PoseFrame


def mirror_normalized_pose(
    points: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror x and exchange anatomical left/right joint identities."""
    mirrored = np.asarray(points, dtype=np.float32).copy()
    mirrored[:, 0] *= -1.0
    mirrored_valid = np.asarray(valid, dtype=bool).copy()
    for left, right in config.LEFT_RIGHT_PAIRS:
        mirrored[[left, right]] = mirrored[[right, left]]
        mirrored_valid[[left, right]] = mirrored_valid[[right, left]]
    return mirrored, mirrored_valid


def compare_poses(
    reference: PoseFrame,
    user: PoseFrame,
    *,
    allow_mirror: bool = config.ALLOW_MIRROR_MATCH,
) -> Dict[str, Any]:
    """Score one normalized pose with position, angle, and coverage terms."""
    direct = _compare_normalized(reference, user)
    if not allow_mirror or reference.normalized_keypoints is None:
        return direct

    mirrored_points, mirrored_valid = mirror_normalized_pose(
        reference.normalized_keypoints,
        reference.valid_mask,
    )
    mirrored_reference = PoseFrame(
        timestamp=reference.timestamp,
        frame_index=reference.frame_index,
        source=reference.source,
        bbox=reference.bbox,
        keypoints=reference.keypoints,
        confidences=reference.confidences,
        valid_mask=mirrored_valid,
        normalized_keypoints=mirrored_points,
        joint_angles=reference.joint_angles,
        pose_confidence=reference.pose_confidence,
        people_count=reference.people_count,
        selected_index=reference.selected_index,
        infer_ms=reference.infer_ms,
    )
    mirrored = _compare_normalized(mirrored_reference, user)
    mirrored["mirrored"] = True
    return mirrored if float(mirrored["score"]) > float(direct["score"]) else direct


def _compare_normalized(reference: PoseFrame, user: PoseFrame) -> Dict[str, Any]:
    if reference.normalized_keypoints is None or user.normalized_keypoints is None:
        return empty_result("Pose not detected")

    body = np.asarray(config.BODY_JOINTS, dtype=np.int32)
    common = reference.valid_mask & user.valid_mask
    common_indices = body[common[body]]
    coverage = float(len(common_indices) / len(body))
    if len(common_indices) < config.MIN_COMMON_BODY_KEYPOINTS:
        return empty_result("Pose not detected", coverage=coverage)

    ref_points = reference.normalized_keypoints[common_indices]
    user_points = user.normalized_keypoints[common_indices]
    distances = np.linalg.norm(ref_points - user_points, axis=1)
    weights = np.ones(len(common_indices), dtype=np.float32)
    weights[np.isin(common_indices, (9, 10, 15, 16))] = 1.35
    position_similarities = np.exp(
        -0.5 * (distances / config.POSITION_SIGMA) ** 2
    )
    position_score = float(100.0 * np.average(position_similarities, weights=weights))

    keypoint_errors = {
        int(index): float(distance)
        for index, distance in zip(common_indices, distances)
        if float(distance) >= config.KEYPOINT_ERROR_THRESHOLD
    }
    angle_similarities = []
    angle_errors: Dict[str, float] = {}
    for name, triple in config.ANGLE_TRIPLES.items():
        first, middle, last = triple
        if not (common[first] and common[middle] and common[last]):
            continue
        reference_angle = joint_angle(
            reference.normalized_keypoints, first, middle, last
        )
        user_angle = joint_angle(user.normalized_keypoints, first, middle, last)
        if not (np.isfinite(reference_angle) and np.isfinite(user_angle)):
            continue
        difference = abs(reference_angle - user_angle)
        if difference >= config.ANGLE_ERROR_THRESHOLD:
            angle_errors[name] = float(difference)
        angle_similarities.append(
            math.exp(-0.5 * (difference / config.ANGLE_SIGMA_DEGREES) ** 2)
        )

    angle_score = (
        float(100.0 * np.mean(angle_similarities))
        if angle_similarities
        else position_score
    )
    pose_score = (
        config.POSE_SCORE_WEIGHTS["position"] * position_score
        + config.POSE_SCORE_WEIGHTS["angle"] * angle_score
    )
    pose_score *= 0.60 + 0.40 * coverage
    pose_score = float(np.clip(pose_score, 0.0, 100.0))
    return {
        "score": pose_score,
        "feedback": feedback_for_score(pose_score, user.valid_count),
        "pose": pose_score,
        "position": position_score,
        "angle": angle_score,
        "motion": 100.0,
        "coverage": coverage,
        "mirrored": False,
        "player_motion": 0.0,
        "reference_motion": 0.0,
        "motion_used": False,
        "score_event": True,
        "error_keypoints": sorted(keypoint_errors),
        "error_joints": sorted(angle_errors),
        "error_summary": error_summary(keypoint_errors, angle_errors),
    }


def apply_motion_score(
    pose_result: Dict[str, Any],
    reference_previous: PoseFrame,
    reference_current: PoseFrame,
    user_previous: PoseFrame,
    user_current: PoseFrame,
) -> Dict[str, Any]:
    """Combine current-pose quality with motion over a short history window."""
    frames = (
        reference_previous,
        reference_current,
        user_previous,
        user_current,
    )
    if any(frame.normalized_keypoints is None for frame in frames):
        return pose_result

    ref_previous_points = reference_previous.normalized_keypoints
    ref_current_points = reference_current.normalized_keypoints
    ref_previous_valid = reference_previous.valid_mask
    ref_current_valid = reference_current.valid_mask
    if bool(pose_result.get("mirrored")):
        ref_previous_points, ref_previous_valid = mirror_normalized_pose(
            ref_previous_points, ref_previous_valid
        )
        ref_current_points, ref_current_valid = mirror_normalized_pose(
            ref_current_points, ref_current_valid
        )

    common = (
        ref_previous_valid
        & ref_current_valid
        & user_previous.valid_mask
        & user_current.valid_mask
    )
    body = np.asarray(config.BODY_JOINTS, dtype=np.int32)
    common_indices = body[common[body]]
    if len(common_indices) < config.MIN_COMMON_BODY_KEYPOINTS:
        return pose_result

    player_delta = (
        user_current.normalized_keypoints[common_indices]
        - user_previous.normalized_keypoints[common_indices]
    )
    reference_delta = (
        ref_current_points[common_indices] - ref_previous_points[common_indices]
    )
    weights = np.ones(len(common_indices), dtype=np.float32)
    weights[np.isin(common_indices, (9, 10, 15, 16))] = 1.35
    player_motion = float(
        np.sqrt(np.average(np.sum(player_delta * player_delta, axis=1), weights=weights))
    )
    reference_motion = float(
        np.sqrt(
            np.average(
                np.sum(reference_delta * reference_delta, axis=1),
                weights=weights,
            )
        )
    )

    updated = dict(pose_result)
    updated.update(
        {
            "player_motion": player_motion,
            "reference_motion": reference_motion,
            "motion_used": True,
        }
    )
    if reference_motion < config.MOTION_ACTIVE_THRESHOLD:
        updated["motion"] = 100.0
        return updated

    errors = np.linalg.norm(player_delta - reference_delta, axis=1)
    vector_similarities = np.exp(
        -0.5 * (errors / config.MOTION_VECTOR_SIGMA) ** 2
    )
    vector_similarity = float(np.average(vector_similarities, weights=weights))

    effective_player = max(0.0, player_motion - config.MOTION_NOISE_FLOOR)
    effective_reference = max(
        1e-6, reference_motion - config.MOTION_NOISE_FLOOR
    )
    activity_ratio = effective_player / effective_reference
    activity_similarity = float(
        math.exp(
            -0.5
            * (
                math.log(
                    (effective_player + 0.02) / (effective_reference + 0.02)
                )
                / 0.70
            )
            ** 2
        )
    )
    motion_similarity = 0.70 * vector_similarity + 0.30 * activity_similarity

    # Prevent a stationary player from earning a high score merely because
    # the current pose is broadly similar while the reference is moving.
    activity_progress = float(
        np.clip((activity_ratio - 0.20) / 0.55, 0.0, 1.0)
    )
    anti_static_factor = 0.45 + 0.55 * activity_progress
    final_score = (
        config.FINAL_SCORE_WEIGHTS["pose"] * float(pose_result["pose"])
        + config.FINAL_SCORE_WEIGHTS["motion"] * (100.0 * motion_similarity)
    )
    final_score *= anti_static_factor
    final_score = float(np.clip(final_score, 0.0, 100.0))
    updated.update(
        {
            "score": final_score,
            "feedback": feedback_for_score(final_score, user_current.valid_count),
            "motion": 100.0 * motion_similarity,
        }
    )
    return updated


class HoldStateFilter:
    """Debounce low reference activity with EMA and hysteresis."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.low_samples = 0
        self.smoothed_motion: float | None = None

    def update(self, reference_motion: float, motion_used: bool) -> bool:
        if not motion_used or not np.isfinite(reference_motion):
            self.reset()
            return False
        motion = max(0.0, float(reference_motion))
        if self.smoothed_motion is None:
            self.smoothed_motion = motion
        else:
            alpha = config.MOTION_HOLD_EMA_ALPHA
            self.smoothed_motion = (
                alpha * motion + (1.0 - alpha) * self.smoothed_motion
            )

        if self.active:
            if self.smoothed_motion >= config.MOTION_HOLD_EXIT_THRESHOLD:
                self.active = False
                self.low_samples = 0
        elif self.smoothed_motion < config.MOTION_ACTIVE_THRESHOLD:
            self.low_samples += 1
            if self.low_samples >= config.MOTION_HOLD_CONFIRM_SAMPLES:
                self.active = True
        else:
            self.low_samples = 0
        return self.active


def joint_angle(points: np.ndarray, first: int, middle: int, last: int) -> float:
    left = points[first] - points[middle]
    right = points[last] - points[middle]
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator < 1e-8:
        return float("nan")
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def feedback_for_score(score: float, valid_count: int) -> str:
    if valid_count < config.MIN_VALID_KEYPOINTS:
        return "Pose not detected"
    if score >= config.FEEDBACK_THRESHOLDS["perfect"]:
        return "Perfect"
    if score >= config.FEEDBACK_THRESHOLDS["super"]:
        return "Super"
    if score >= config.FEEDBACK_THRESHOLDS["good"]:
        return "Good"
    return "Miss"


def empty_result(feedback: str, *, coverage: float = 0.0) -> Dict[str, Any]:
    return {
        "score": 0.0,
        "feedback": feedback,
        "pose": 0.0,
        "position": 0.0,
        "angle": 0.0,
        "motion": 0.0,
        "coverage": coverage,
        "mirrored": False,
        "player_motion": 0.0,
        "reference_motion": 0.0,
        "motion_used": False,
        "score_event": False,
        "error_keypoints": [],
        "error_joints": [],
        "error_summary": "",
    }


def error_summary(
    keypoint_errors: Dict[int, float],
    angle_errors: Dict[str, float],
) -> str:
    labels = []
    for index in sorted(keypoint_errors, key=keypoint_errors.get, reverse=True)[:3]:
        labels.append(config.COCO_KEYPOINT_NAMES[index])
    labels.extend(
        name.replace("_", " ")
        for name in sorted(angle_errors, key=angle_errors.get, reverse=True)[:2]
    )
    return ", ".join(labels[:4])
