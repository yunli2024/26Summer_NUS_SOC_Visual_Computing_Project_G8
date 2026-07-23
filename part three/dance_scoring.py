"""Pose normalization and temporal matching for Bonus Task 2.

The scorer deliberately ignores absolute image position and body size.  It
compares the twelve body joints (shoulders through ankles) with a mixture of
normalized joint distances and limb-angle similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


BODY_JOINTS = np.arange(5, 17)
LEFT_RIGHT_PAIRS = (
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
)
ANGLE_TRIPLES = (
    (5, 7, 9),    # left elbow
    (6, 8, 10),   # right elbow
    (7, 5, 11),   # left shoulder
    (8, 6, 12),   # right shoulder
    (5, 11, 13),  # left hip
    (6, 12, 14),  # right hip
    (11, 13, 15), # left knee
    (12, 14, 16), # right knee
)


@dataclass(frozen=True)
class PoseScore:
    score: float
    position_score: float
    angle_score: float
    coverage: float
    mirrored: bool = False


@dataclass(frozen=True)
class MatchResult(PoseScore):
    reference_index: int = -1
    lag_frames: int = 0


def _center(points: np.ndarray, valid: np.ndarray, indices: Iterable[int]) -> Optional[np.ndarray]:
    idx = np.asarray(list(indices), dtype=np.int32)
    idx = idx[valid[idx]]
    if idx.size == 0:
        return None
    return np.mean(points[idx], axis=0)


def normalize_pose(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a translation- and scale-normalized 17-joint pose."""
    points = np.asarray(points, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool).copy()
    if points.shape != (17, 2) or valid.shape != (17,):
        raise ValueError("Expected points (17, 2) and valid (17,).")

    valid &= np.isfinite(points).all(axis=1)
    if np.count_nonzero(valid[BODY_JOINTS]) < 4:
        raise ValueError("Too few visible body joints to score this pose.")

    hip_center = _center(points, valid, (11, 12))
    shoulder_center = _center(points, valid, (5, 6))
    root = hip_center if hip_center is not None else shoulder_center
    if root is None:
        root = np.mean(points[BODY_JOINTS[valid[BODY_JOINTS]]], axis=0)

    scale_candidates: list[float] = []
    for a, b in ((5, 6), (11, 12)):
        if valid[a] and valid[b]:
            scale_candidates.append(float(np.linalg.norm(points[a] - points[b])))
    if hip_center is not None and shoulder_center is not None:
        scale_candidates.append(float(np.linalg.norm(hip_center - shoulder_center)) * 1.35)

    body = points[BODY_JOINTS[valid[BODY_JOINTS]]]
    if body.shape[0] >= 2:
        span = np.ptp(body, axis=0)
        scale_candidates.append(float(np.linalg.norm(span)) * 0.35)

    scale = max(scale_candidates, default=0.0)
    if scale < 1e-6:
        raise ValueError("Pose scale is too small to score.")

    normalized = np.full((17, 2), np.nan, dtype=np.float32)
    normalized[valid] = (points[valid] - root) / scale
    return normalized, valid


def mirror_pose(points: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mirror a pose horizontally and exchange anatomical left/right joints."""
    mirrored = np.asarray(points, dtype=np.float32).copy()
    mirrored[:, 0] *= -1.0
    mirrored_valid = np.asarray(valid, dtype=bool).copy()
    for left, right in LEFT_RIGHT_PAIRS:
        mirrored[[left, right]] = mirrored[[right, left]]
        mirrored_valid[[left, right]] = mirrored_valid[[right, left]]
    return mirrored, mirrored_valid


def _joint_angle(points: np.ndarray, a: int, b: int, c: int) -> float:
    ba = points[a] - points[b]
    bc = points[c] - points[b]
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-8:
        return float("nan")
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _score_normalized(
    player: np.ndarray,
    player_valid: np.ndarray,
    reference: np.ndarray,
    reference_valid: np.ndarray,
) -> PoseScore:
    common = player_valid & reference_valid
    body_common = common[BODY_JOINTS]
    common_indices = BODY_JOINTS[body_common]
    coverage = float(common_indices.size / BODY_JOINTS.size)
    if common_indices.size < 4:
        return PoseScore(0.0, 0.0, 0.0, coverage)

    # Wrists/ankles carry more dance information than shoulders/hips.
    joint_weights = np.ones(common_indices.size, dtype=np.float32)
    joint_weights[np.isin(common_indices, (9, 10, 15, 16))] = 1.35
    distances = np.linalg.norm(player[common_indices] - reference[common_indices], axis=1)
    position_similarity = np.exp(-0.5 * (distances / 0.55) ** 2)
    position_score = float(np.average(position_similarity, weights=joint_weights))

    angle_similarities: list[float] = []
    for a, b, c in ANGLE_TRIPLES:
        if common[a] and common[b] and common[c]:
            player_angle = _joint_angle(player, a, b, c)
            reference_angle = _joint_angle(reference, a, b, c)
            if np.isfinite(player_angle) and np.isfinite(reference_angle):
                difference = abs(player_angle - reference_angle)
                angle_similarities.append(float(np.exp(-0.5 * (difference / 32.0) ** 2)))

    angle_score = float(np.mean(angle_similarities)) if angle_similarities else position_score
    combined = 0.65 * angle_score + 0.35 * position_score
    # Missing joints reduce confidence without making partial detections useless.
    combined *= 0.60 + 0.40 * coverage
    return PoseScore(
        score=float(np.clip(100.0 * combined, 0.0, 100.0)),
        position_score=100.0 * position_score,
        angle_score=100.0 * angle_score,
        coverage=coverage,
    )


def pose_similarity(
    player_points: np.ndarray,
    player_valid: np.ndarray,
    reference_points: np.ndarray,
    reference_valid: np.ndarray,
    allow_mirror: bool = True,
) -> PoseScore:
    """Compare two poses, optionally accepting an anatomically mirrored move."""
    player, player_ok = normalize_pose(player_points, player_valid)
    reference, reference_ok = normalize_pose(reference_points, reference_valid)
    direct = _score_normalized(player, player_ok, reference, reference_ok)
    if not allow_mirror:
        return direct

    mirrored_reference, mirrored_ok = mirror_pose(reference, reference_ok)
    mirrored = _score_normalized(player, player_ok, mirrored_reference, mirrored_ok)
    if mirrored.score > direct.score:
        return PoseScore(
            mirrored.score,
            mirrored.position_score,
            mirrored.angle_score,
            mirrored.coverage,
            True,
        )
    return direct


def best_reference_match(
    player_points: np.ndarray,
    player_valid: np.ndarray,
    reference_points: np.ndarray,
    reference_valid: np.ndarray,
    current_index: int,
    max_lag_frames: int,
    allow_mirror: bool = True,
) -> Optional[MatchResult]:
    """Find the best recent reference pose so normal human reaction delay is tolerated."""
    count = len(reference_points)
    if count == 0:
        return None
    current_index = int(np.clip(current_index, 0, count - 1))
    start = max(0, current_index - max(0, int(max_lag_frames)))
    best: Optional[MatchResult] = None
    for index in range(start, current_index + 1):
        try:
            candidate = pose_similarity(
                player_points,
                player_valid,
                reference_points[index],
                reference_valid[index],
                allow_mirror=allow_mirror,
            )
        except ValueError:
            continue
        result = MatchResult(
            score=candidate.score,
            position_score=candidate.position_score,
            angle_score=candidate.angle_score,
            coverage=candidate.coverage,
            mirrored=candidate.mirrored,
            reference_index=index,
            lag_frames=current_index - index,
        )
        if best is None or result.score > best.score:
            best = result
    return best


def feedback_for_score(score: float) -> tuple[str, int, tuple[int, int, int]]:
    """Return label, game points and BGR display colour."""
    if score >= 85.0:
        return "PERFECT", 1000, (80, 230, 80)
    if score >= 70.0:
        return "GREAT", 700, (0, 210, 255)
    if score >= 55.0:
        return "GOOD", 400, (0, 150, 255)
    return "MISS", 0, (80, 80, 255)
