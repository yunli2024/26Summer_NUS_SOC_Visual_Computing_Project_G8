"""Explainable pose similarity scoring."""

from __future__ import annotations

import math
from typing import Any, Dict

import numpy as np

from . import config
from .pose_features import valid_bone_vectors
from .pose_types import PoseFrame


def compare_poses(reference: PoseFrame, user: PoseFrame) -> Dict[str, Any]:
    if reference.normalized_keypoints is None or user.normalized_keypoints is None:
        return empty_result("Pose not detected")

    common = reference.valid_mask & user.valid_mask
    if int(np.count_nonzero(common)) < config.MIN_VALID_KEYPOINTS:
        return empty_result("Pose not detected")

    distances = np.linalg.norm(reference.normalized_keypoints[common] - user.normalized_keypoints[common], axis=1)
    weights = (reference.confidences[common] + user.confidences[common]) / 2.0
    common_indices = np.flatnonzero(common)
    keypoint_errors = {
        int(idx): float(distance)
        for idx, distance in zip(common_indices, distances)
        if float(distance) >= config.KEYPOINT_ERROR_THRESHOLD
    }
    if float(weights.sum()) <= 1e-6:
        position_score = 0.0
    else:
        weighted_distance = float(np.average(distances, weights=weights))
        # Forgiving curve: rough imitation should still receive partial credit.
        position_score = float(np.clip(100.0 * math.exp(-1.45 * weighted_distance), 0.0, 100.0))

    angle_scores = []
    angle_errors = {}
    for name, ref_angle in reference.joint_angles.items():
        if name not in user.joint_angles:
            continue
        diff = abs(ref_angle - user.joint_angles[name])
        if diff >= config.ANGLE_ERROR_THRESHOLD:
            angle_errors[name] = float(diff)
        angle_scores.append(max(0.0, 100.0 - diff / 130.0 * 100.0))
    angle_score = float(np.mean(angle_scores)) if angle_scores else position_score

    ref_vectors = valid_bone_vectors(reference)
    user_vectors = valid_bone_vectors(user)
    vector_scores = []
    for bone, ref_vec in ref_vectors.items():
        if bone not in user_vectors:
            continue
        cosine = float(np.clip(np.dot(ref_vec, user_vectors[bone]), -1.0, 1.0))
        # Treat broadly similar directions as acceptable; opposite directions still score low.
        vector_scores.append(max(0.0, cosine) * 70.0 + 30.0)
    vector_score = float(np.mean(vector_scores)) if vector_scores else position_score
    coarse_score = coarse_pose_score(reference, user)

    score = (
        config.SCORE_WEIGHTS["coarse"] * coarse_score
        + config.SCORE_WEIGHTS["position"] * position_score
        + config.SCORE_WEIGHTS["angle"] * angle_score
        + config.SCORE_WEIGHTS["vector"] * vector_score
    )
    feedback = feedback_for_score(score, user.valid_count)
    return {
        "score": float(np.clip(score, 0.0, 100.0)),
        "feedback": feedback,
        "coarse": coarse_score,
        "position": position_score,
        "angle": angle_score,
        "vector": vector_score,
        "error_keypoints": sorted(keypoint_errors),
        "error_joints": sorted(angle_errors),
        "error_summary": error_summary(keypoint_errors, angle_errors),
    }


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


def empty_result(feedback: str):
    return {
        "score": 0.0,
        "feedback": feedback,
        "coarse": 0.0,
        "position": 0.0,
        "angle": 0.0,
        "vector": 0.0,
        "error_keypoints": [],
        "error_joints": [],
        "error_summary": "",
    }


def error_summary(keypoint_errors: Dict[int, float], angle_errors: Dict[str, float]) -> str:
    labels = []
    for idx in sorted(keypoint_errors, key=keypoint_errors.get, reverse=True)[:3]:
        labels.append(config.COCO_KEYPOINT_NAMES[idx])
    labels.extend(name.replace("_", " ") for name in sorted(angle_errors, key=angle_errors.get, reverse=True)[:2])
    return ", ".join(labels[:4])


def coarse_pose_score(reference: PoseFrame, user: PoseFrame) -> float:
    """Reward approximate body-part placement, not exact pixel-level matching."""
    if reference.normalized_keypoints is None or user.normalized_keypoints is None:
        return 0.0

    checks = []
    # Hands, elbows, knees, ankles carry most visible dance motion.
    important_points = [7, 8, 9, 10, 13, 14, 15, 16]
    for idx in important_points:
        if not (reference.valid_mask[idx] and user.valid_mask[idx]):
            continue
        ref = reference.normalized_keypoints[idx]
        usr = user.normalized_keypoints[idx]
        checks.append(axis_bucket_score(ref[0], usr[0], tolerance=0.28))
        checks.append(axis_bucket_score(ref[1], usr[1], tolerance=0.28))

    # Limb extension: bent/extended similarity is easier to match than exact angle.
    for name, ref_angle in reference.joint_angles.items():
        if name not in user.joint_angles:
            continue
        ref_bucket = angle_bucket(ref_angle)
        user_bucket = angle_bucket(user.joint_angles[name])
        checks.append(100.0 if ref_bucket == user_bucket else 55.0 if abs(ref_bucket - user_bucket) == 1 else 20.0)

    if not checks:
        return 0.0
    return float(np.mean(checks))


def axis_bucket_score(ref_value: float, user_value: float, tolerance: float) -> float:
    diff = abs(ref_value - user_value)
    if diff <= tolerance:
        return 100.0
    if diff <= tolerance * 2.0:
        return 70.0
    if np.sign(ref_value) == np.sign(user_value):
        return 45.0
    return 15.0


def angle_bucket(angle: float) -> int:
    if angle < 70:
        return 0
    if angle < 130:
        return 1
    return 2
