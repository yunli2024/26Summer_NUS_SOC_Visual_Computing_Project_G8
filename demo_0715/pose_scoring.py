from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Optional, Sequence, Tuple

import numpy as np

from pose_pipeline import FramePoseResult, PoseDetection


CORE_KEYPOINTS = (5, 6, 11, 12)
BODY_KEYPOINTS = tuple(range(5, 17))
KEYPOINT_WEIGHTS = np.array(
    [0, 0, 0, 0, 0, 1.0, 1.0, 1.25, 1.25, 1.5, 1.5, 1.0, 1.0, 1.25, 1.25, 1.5, 1.5],
    dtype=np.float32,
)
MIRROR_INDEX = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15])
ANGLE_TRIPLES = (
    (5, 7, 9),
    (6, 8, 10),
    (7, 5, 11),
    (8, 6, 12),
    (5, 11, 13),
    (6, 12, 14),
    (11, 13, 15),
    (12, 14, 16),
)
LIMB_PAIRS = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 11),
    (6, 12),
)


@dataclass(frozen=True)
class NormalizedPose:
    frame_index: int
    points: np.ndarray
    visible: np.ndarray
    confidence: np.ndarray
    scale: float
    center: Tuple[float, float]


@dataclass(frozen=True)
class PoseMatch:
    score: float
    feedback: str
    matched_ref_frame: Optional[int]
    lag_frames: Optional[int]
    common_keypoints: int
    distance_score: float
    angle_score: float
    message: str
    limb_score: float = 0.0
    quality: float = 0.0
    mirror_used: bool = False


def normalize_detection(
    detection: PoseDetection,
    *,
    frame_index: int,
    keypoint_conf: float = 0.25,
) -> Optional[NormalizedPose]:
    if len(detection.keypoints) < 17:
        return None

    raw_points = np.array([(x, y) for x, y, _ in detection.keypoints], dtype=np.float32)
    conf = np.array([c for _, _, c in detection.keypoints], dtype=np.float32)
    visible = conf >= keypoint_conf
    if int(visible.sum()) < 4:
        return None

    core_visible = [idx for idx in CORE_KEYPOINTS if visible[idx]]
    if core_visible:
        center_xy = raw_points[core_visible].mean(axis=0)
    else:
        center_xy = raw_points[visible].mean(axis=0)

    shoulder_width = joint_distance(raw_points, visible, 5, 6)
    hip_width = joint_distance(raw_points, visible, 11, 12)
    torso_height = average_distances(raw_points, visible, ((5, 11), (6, 12)))
    bbox_scale = bbox_diagonal(raw_points[visible])
    scale = max(shoulder_width, hip_width, torso_height, bbox_scale * 0.35, 1.0)

    points = (raw_points - center_xy) / scale
    return NormalizedPose(
        frame_index=frame_index,
        points=points,
        visible=visible,
        confidence=conf,
        scale=float(scale),
        center=(float(center_xy[0]), float(center_xy[1])),
    )


def normalize_main_pose(result: FramePoseResult, *, keypoint_conf: float = 0.25) -> Optional[NormalizedPose]:
    detection = result.main_detection
    if detection is None:
        return None
    return normalize_detection(detection, frame_index=result.frame_index, keypoint_conf=keypoint_conf)


def compare_poses(reference: NormalizedPose, user: NormalizedPose, *, allow_mirror: bool = True) -> PoseMatch:
    direct = _compare_aligned(reference, user, mirror_used=False)
    if not allow_mirror:
        return direct
    mirrored = _compare_aligned(reference, mirror_pose(user), mirror_used=True)
    return mirrored if mirrored.score > direct.score else direct


def _compare_aligned(reference: NormalizedPose, user: NormalizedPose, *, mirror_used: bool) -> PoseMatch:
    common = reference.visible & user.visible
    body_common = common.copy()
    body_common[:5] = False
    common = body_common
    common_count = int(common.sum())
    if common_count < 5:
        return PoseMatch(
            score=0.0,
            feedback="No Pose",
            matched_ref_frame=reference.frame_index,
            lag_frames=None,
            common_keypoints=common_count,
            distance_score=0.0,
            angle_score=0.0,
            message="Not enough shared visible keypoints.",
            mirror_used=mirror_used,
        )

    diffs = reference.points[common] - user.points[common]
    distances = np.linalg.norm(diffs, axis=1)
    confidence = np.minimum(reference.confidence[common], user.confidence[common])
    weights = KEYPOINT_WEIGHTS[common] * np.clip(confidence, 0.15, 1.0)
    mean_distance = weighted_mean(distances, weights)
    distance_score = float(np.exp(-((mean_distance / 0.42) ** 2)))

    angle_values = []
    angle_weights = []
    for a, b, c in ANGLE_TRIPLES:
        if reference.visible[a] and reference.visible[b] and reference.visible[c] and user.visible[a] and user.visible[b] and user.visible[c]:
            ref_angle = joint_angle(reference.points[a], reference.points[b], reference.points[c])
            user_angle = joint_angle(user.points[a], user.points[b], user.points[c])
            angle_values.append(abs(ref_angle - user_angle))
            angle_weights.append(min(reference.confidence[a], reference.confidence[b], reference.confidence[c], user.confidence[a], user.confidence[b], user.confidence[c]))
    if angle_values:
        angle_error = weighted_mean(np.asarray(angle_values), np.asarray(angle_weights))
        angle_score = float(np.exp(-((angle_error / 0.75) ** 2)))
    else:
        angle_score = distance_score

    limb_values = []
    limb_weights = []
    for a, b in LIMB_PAIRS:
        if reference.visible[a] and reference.visible[b] and user.visible[a] and user.visible[b]:
            ref_vector = reference.points[b] - reference.points[a]
            user_vector = user.points[b] - user.points[a]
            limb_values.append(vector_similarity(ref_vector, user_vector))
            limb_weights.append(min(reference.confidence[a], reference.confidence[b], user.confidence[a], user.confidence[b]))
    limb_score = weighted_mean(np.asarray(limb_values), np.asarray(limb_weights)) if limb_values else distance_score

    coverage = float(KEYPOINT_WEIGHTS[common].sum() / max(KEYPOINT_WEIGHTS[list(BODY_KEYPOINTS)].sum(), 1e-6))
    confidence_quality = float(np.clip(weighted_mean(confidence, KEYPOINT_WEIGHTS[common]), 0.0, 1.0))
    quality = float(np.clip((0.35 + 0.65 * coverage) * (0.85 + 0.15 * confidence_quality), 0.0, 1.0))
    base_similarity = (0.48 * distance_score + 0.30 * angle_score + 0.22 * limb_score) * quality
    # A squared calibration preserves genuinely close matches while preventing
    # generic upright poses from receiving inflated dance-game scores.
    score = 100.0 * (base_similarity**2)
    score = float(np.clip(score, 0.0, 100.0))
    feedback = feedback_label(score, common_count)
    return PoseMatch(
        score=score,
        feedback=feedback,
        matched_ref_frame=reference.frame_index,
        lag_frames=None,
        common_keypoints=common_count,
        distance_score=distance_score,
        angle_score=angle_score,
        message="OK",
        limb_score=float(limb_score),
        quality=quality,
        mirror_used=mirror_used,
    )


class TemporalPoseMatcher:
    def __init__(
        self,
        *,
        max_lag_frames: int = 12,
        keypoint_conf: float = 0.25,
        allow_mirror: bool = True,
        lag_jump_penalty: float = 1.25,
    ) -> None:
        self.max_lag_frames = max_lag_frames
        self.keypoint_conf = keypoint_conf
        self.allow_mirror = allow_mirror
        self.lag_jump_penalty = lag_jump_penalty
        self.reference_buffer: Deque[NormalizedPose] = deque(maxlen=max_lag_frames + 1)
        self.last_match: PoseMatch = missing_match("Waiting")
        self.last_lag: Optional[int] = None

    def reset(self) -> None:
        self.reference_buffer.clear()
        self.last_match = missing_match("Waiting")
        self.last_lag = None

    def set_max_lag_frames(self, max_lag_frames: int, existing: Sequence[NormalizedPose] = ()) -> None:
        self.max_lag_frames = max(0, max_lag_frames)
        self.reference_buffer = deque(existing, maxlen=self.max_lag_frames + 1)

    def push_reference(self, result: FramePoseResult) -> Optional[NormalizedPose]:
        pose = normalize_main_pose(result, keypoint_conf=self.keypoint_conf)
        if pose is not None:
            self.reference_buffer.append(pose)
        return pose

    def match_user(self, user_result: FramePoseResult) -> PoseMatch:
        user_pose = normalize_main_pose(user_result, keypoint_conf=self.keypoint_conf)
        if user_pose is None:
            self.last_match = missing_match("Find User")
            return self.last_match
        if not self.reference_buffer:
            self.last_match = missing_match("Find Ref")
            return self.last_match

        latest_ref_frame = self.reference_buffer[-1].frame_index
        matches = [compare_poses(ref_pose, user_pose, allow_mirror=self.allow_mirror) for ref_pose in self.reference_buffer]

        def selection_score(item: PoseMatch) -> float:
            if item.matched_ref_frame is None:
                return item.score
            lag = latest_ref_frame - item.matched_ref_frame
            if self.last_lag is None:
                return item.score
            return item.score - self.lag_jump_penalty * abs(lag - self.last_lag)

        best = max(matches, key=selection_score)
        lag = latest_ref_frame - best.matched_ref_frame if best.matched_ref_frame is not None else None
        if lag is not None:
            self.last_lag = lag
        self.last_match = PoseMatch(
            score=best.score,
            feedback=best.feedback,
            matched_ref_frame=best.matched_ref_frame,
            lag_frames=lag,
            common_keypoints=best.common_keypoints,
            distance_score=best.distance_score,
            angle_score=best.angle_score,
            message=best.message,
            limb_score=best.limb_score,
            quality=best.quality,
            mirror_used=best.mirror_used,
        )
        return self.last_match


def feedback_label(score: float, common_keypoints: int) -> str:
    if common_keypoints < 7:
        return "Partial"
    if score >= 94:
        return "Perfect"
    if score >= 85:
        return "Super"
    if score >= 70:
        return "Good"
    if score >= 50:
        return "Keep Going"
    return "Miss"


def missing_match(label: str) -> PoseMatch:
    return PoseMatch(
        score=0.0,
        feedback=label,
        matched_ref_frame=None,
        lag_frames=None,
        common_keypoints=0,
        distance_score=0.0,
        angle_score=0.0,
        message=label,
    )


def joint_distance(points: np.ndarray, visible: np.ndarray, a: int, b: int) -> float:
    if not (visible[a] and visible[b]):
        return 0.0
    return float(np.linalg.norm(points[a] - points[b]))


def average_distances(points: np.ndarray, visible: np.ndarray, pairs: Iterable[Tuple[int, int]]) -> float:
    values = [joint_distance(points, visible, a, b) for a, b in pairs]
    values = [value for value in values if value > 0]
    if not values:
        return 0.0
    return float(np.mean(values))


def bbox_diagonal(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    span = points.max(axis=0) - points.min(axis=0)
    return float(np.linalg.norm(span))


def joint_angle(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    ba = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    bc = np.asarray(c, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    denom = max(float(np.linalg.norm(ba) * np.linalg.norm(bc)), 1e-6)
    cosine = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
    return float(np.arccos(cosine))


def mirror_pose(pose: NormalizedPose) -> NormalizedPose:
    points = pose.points[MIRROR_INDEX].copy()
    points[:, 0] *= -1.0
    return NormalizedPose(
        frame_index=pose.frame_index,
        points=points,
        visible=pose.visible[MIRROR_INDEX].copy(),
        confidence=pose.confidence[MIRROR_INDEX].copy(),
        scale=pose.scale,
        center=pose.center,
    )


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    total = float(np.asarray(weights, dtype=np.float64).sum())
    if total <= 1e-8:
        return float(np.asarray(values, dtype=np.float64).mean())
    return float(np.dot(np.asarray(values, dtype=np.float64), np.asarray(weights, dtype=np.float64)) / total)


def vector_similarity(first: np.ndarray, second: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(first) * np.linalg.norm(second)), 1e-6)
    cosine = float(np.clip(np.dot(first, second) / denom, -1.0, 1.0))
    return 0.5 * (cosine + 1.0)
