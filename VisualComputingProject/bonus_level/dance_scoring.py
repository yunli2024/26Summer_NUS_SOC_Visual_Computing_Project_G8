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
# A reference move is considered a genuine hold only after the UI-side
# HoldStateFilter confirms that its smoothed activity is below this value.
# The higher exit threshold adds hysteresis, preventing HOLD from flickering
# when a slow movement sits close to the boundary.
MOTION_ACTIVE_THRESHOLD = 0.07
MOTION_HOLD_EXIT_THRESHOLD = 0.10
MOTION_HOLD_CONFIRM_SAMPLES = 2
MOTION_HOLD_EMA_ALPHA = 0.35
MOTION_NOISE_FLOOR = 0.04
MOTION_VECTOR_SIGMA = 0.22


@dataclass(frozen=True)
class PoseScore:
    score: float
    position_score: float
    angle_score: float
    coverage: float
    mirrored: bool = False
    motion_score: float = 100.0
    player_motion: float = 0.0
    reference_motion: float = 0.0
    motion_used: bool = False


@dataclass(frozen=True)
class MatchResult(PoseScore):
    reference_index: int = -1
    lag_frames: int = 0


class HoldStateFilter:
    """Debounce reference hold detection with smoothing and hysteresis."""

    def __init__(
        self,
        enter_threshold: float = MOTION_ACTIVE_THRESHOLD,
        exit_threshold: float = MOTION_HOLD_EXIT_THRESHOLD,
        confirm_samples: int = MOTION_HOLD_CONFIRM_SAMPLES,
        ema_alpha: float = MOTION_HOLD_EMA_ALPHA,
    ) -> None:
        if not 0.0 <= enter_threshold < exit_threshold:
            raise ValueError("Hold thresholds must satisfy 0 <= enter < exit.")
        if confirm_samples < 1:
            raise ValueError("confirm_samples must be at least 1.")
        if not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in (0, 1].")
        self.enter_threshold = float(enter_threshold)
        self.exit_threshold = float(exit_threshold)
        self.confirm_samples = int(confirm_samples)
        self.ema_alpha = float(ema_alpha)
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.low_samples = 0
        self.smoothed_motion: Optional[float] = None

    def update(self, reference_motion: float, motion_used: bool = True) -> bool:
        if not motion_used or not np.isfinite(reference_motion):
            self.reset()
            return False

        motion = max(0.0, float(reference_motion))
        if self.smoothed_motion is None:
            self.smoothed_motion = motion
        else:
            self.smoothed_motion = (
                self.ema_alpha * motion
                + (1.0 - self.ema_alpha) * self.smoothed_motion
            )

        if self.active:
            if self.smoothed_motion >= self.exit_threshold:
                self.active = False
                self.low_samples = 0
        elif self.smoothed_motion < self.enter_threshold:
            self.low_samples += 1
            if self.low_samples >= self.confirm_samples:
                self.active = True
        else:
            self.low_samples = 0

        return self.active


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


def _motion_aware_score(
    pose_score: PoseScore,
    player_previous_points: np.ndarray,
    player_previous_valid: np.ndarray,
    player_points: np.ndarray,
    player_valid: np.ndarray,
    reference_previous_points: np.ndarray,
    reference_previous_valid: np.ndarray,
    reference_points: np.ndarray,
    reference_valid: np.ndarray,
) -> PoseScore:
    """Combine pose similarity with relative joint motion over a short window."""
    player_previous, player_previous_ok = normalize_pose(
        player_previous_points, player_previous_valid
    )
    player_current, player_current_ok = normalize_pose(player_points, player_valid)
    reference_previous, reference_previous_ok = normalize_pose(
        reference_previous_points, reference_previous_valid
    )
    reference_current, reference_current_ok = normalize_pose(
        reference_points, reference_valid
    )
    if pose_score.mirrored:
        reference_previous, reference_previous_ok = mirror_pose(
            reference_previous, reference_previous_ok
        )
        reference_current, reference_current_ok = mirror_pose(
            reference_current, reference_current_ok
        )

    common = (
        player_previous_ok
        & player_current_ok
        & reference_previous_ok
        & reference_current_ok
    )
    common_indices = BODY_JOINTS[common[BODY_JOINTS]]
    if common_indices.size < 4:
        return pose_score

    player_delta = (
        player_current[common_indices] - player_previous[common_indices]
    )
    reference_delta = (
        reference_current[common_indices] - reference_previous[common_indices]
    )
    weights = np.ones(common_indices.size, dtype=np.float32)
    weights[np.isin(common_indices, (9, 10, 15, 16))] = 1.35

    player_squared = np.sum(player_delta * player_delta, axis=1)
    reference_squared = np.sum(reference_delta * reference_delta, axis=1)
    player_motion = float(np.sqrt(np.average(player_squared, weights=weights)))
    reference_motion = float(
        np.sqrt(np.average(reference_squared, weights=weights))
    )

    # During a genuine hold, pose quality alone should decide the score.
    if reference_motion < MOTION_ACTIVE_THRESHOLD:
        return PoseScore(
            score=pose_score.score,
            position_score=pose_score.position_score,
            angle_score=pose_score.angle_score,
            coverage=pose_score.coverage,
            mirrored=pose_score.mirrored,
            motion_score=100.0,
            player_motion=player_motion,
            reference_motion=reference_motion,
            motion_used=True,
        )

    motion_errors = np.linalg.norm(player_delta - reference_delta, axis=1)
    vector_similarities = np.exp(
        -0.5 * (motion_errors / MOTION_VECTOR_SIGMA) ** 2
    )
    vector_similarity = float(
        np.average(vector_similarities, weights=weights)
    )

    effective_player = max(0.0, player_motion - MOTION_NOISE_FLOOR)
    effective_reference = max(
        1e-6, reference_motion - MOTION_NOISE_FLOOR
    )
    activity_ratio = effective_player / effective_reference
    activity_similarity = float(
        np.exp(
            -0.5
            * (
                np.log(
                    (effective_player + 0.02)
                    / (effective_reference + 0.02)
                )
                / 0.70
            )
            ** 2
        )
    )
    motion_similarity = 0.70 * vector_similarity + 0.30 * activity_similarity

    # A static player must not receive a high score by matching a broadly
    # similar pose while the reference dancer is visibly moving.
    activity_progress = float(
        np.clip((activity_ratio - 0.20) / 0.55, 0.0, 1.0)
    )
    anti_static_factor = 0.45 + 0.55 * activity_progress
    combined = (
        0.55 * pose_score.score + 0.45 * (100.0 * motion_similarity)
    )
    combined *= anti_static_factor
    return PoseScore(
        score=float(np.clip(combined, 0.0, 100.0)),
        position_score=pose_score.position_score,
        angle_score=pose_score.angle_score,
        coverage=pose_score.coverage,
        mirrored=pose_score.mirrored,
        motion_score=100.0 * motion_similarity,
        player_motion=player_motion,
        reference_motion=reference_motion,
        motion_used=True,
    )


def best_reference_match(
    player_points: np.ndarray,
    player_valid: np.ndarray,
    reference_points: np.ndarray,
    reference_valid: np.ndarray,
    current_index: int,
    max_lag_frames: int,
    allow_mirror: bool = True,
    player_previous_points: Optional[np.ndarray] = None,
    player_previous_valid: Optional[np.ndarray] = None,
    motion_delta_frames: int = 1,
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
            previous_reference_index = index - max(
                1, int(motion_delta_frames)
            )
            if (
                player_previous_points is not None
                and player_previous_valid is not None
                and previous_reference_index >= 0
            ):
                candidate = _motion_aware_score(
                    candidate,
                    player_previous_points,
                    player_previous_valid,
                    player_points,
                    player_valid,
                    reference_points[previous_reference_index],
                    reference_valid[previous_reference_index],
                    reference_points[index],
                    reference_valid[index],
                )
        except ValueError:
            continue
        result = MatchResult(
            score=candidate.score,
            position_score=candidate.position_score,
            angle_score=candidate.angle_score,
            coverage=candidate.coverage,
            mirrored=candidate.mirrored,
            motion_score=candidate.motion_score,
            player_motion=candidate.player_motion,
            reference_motion=candidate.reference_motion,
            motion_used=candidate.motion_used,
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
