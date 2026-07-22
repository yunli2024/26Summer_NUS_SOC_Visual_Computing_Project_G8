from __future__ import annotations

from pose_pipeline import FramePoseResult, PoseDetection, PoseStreamTracker
from pose_scoring import TemporalPoseMatcher, compare_poses, normalize_detection


def make_detection(offset_x: float = 0.0, offset_y: float = 0.0, scale: float = 1.0, arm_delta: float = 0.0) -> PoseDetection:
    base = [
        (50, 20),
        (44, 16),
        (56, 16),
        (40, 18),
        (60, 18),
        (34, 54),
        (66, 54),
        (24, 86),
        (76, 86),
        (18, 118),
        (82, 118),
        (40, 124),
        (60, 124),
        (36, 170),
        (64, 170),
        (32, 216),
        (68, 216),
    ]
    shaped = list(base)
    shaped[9] = (base[9][0] - arm_delta, base[9][1] + arm_delta * 0.3)
    shaped[10] = (base[10][0] + arm_delta, base[10][1] - arm_delta * 0.2)
    keypoints = [(int(x * scale + offset_x), int(y * scale + offset_y), 0.95) for x, y in shaped]
    return PoseDetection(
        keypoints=keypoints,
        box=(int(15 * scale + offset_x), int(10 * scale + offset_y), int(85 * scale + offset_x), int(225 * scale + offset_y)),
        box_confidence=0.99,
        visible_count=17,
        main_score=1.0,
    )


def make_result(frame_index: int, detection: PoseDetection) -> FramePoseResult:
    return FramePoseResult(
        frame_index=frame_index,
        detections=[detection],
        main_index=0,
        inference_ms=0.0,
        draw_ms=0.0,
    )


def test_spatial_normalization() -> None:
    reference = normalize_detection(make_detection(), frame_index=10)
    shifted_scaled = normalize_detection(make_detection(offset_x=300, offset_y=120, scale=1.8), frame_index=11)
    assert reference is not None
    assert shifted_scaled is not None
    match = compare_poses(reference, shifted_scaled)
    assert match.score > 98.0, match
    assert match.common_keypoints == 12


def test_wrong_pose_is_not_perfect() -> None:
    reference = normalize_detection(make_detection(), frame_index=10)
    wrong = make_detection()
    wrong_points = list(wrong.keypoints)
    wrong_points[9] = (82, 45, 0.95)
    wrong_points[10] = (18, 45, 0.95)
    wrong_points[15] = (78, 145, 0.95)
    wrong_points[16] = (22, 145, 0.95)
    wrong = PoseDetection(
        keypoints=wrong_points,
        box=wrong.box,
        box_confidence=wrong.box_confidence,
        visible_count=wrong.visible_count,
        main_score=wrong.main_score,
    )
    wrong_pose = normalize_detection(wrong, frame_index=11)
    assert reference is not None and wrong_pose is not None
    match = compare_poses(reference, wrong_pose)
    assert match.score < 70.0, match


def test_partial_pose_is_quality_gated() -> None:
    reference = normalize_detection(make_detection(), frame_index=10)
    partial = make_detection()
    keypoints = [(x, y, c if idx in (5, 6, 7, 8, 11, 12) else 0.05) for idx, (x, y, c) in enumerate(partial.keypoints)]
    partial = PoseDetection(
        keypoints=keypoints,
        box=partial.box,
        box_confidence=partial.box_confidence,
        visible_count=6,
        main_score=partial.main_score,
    )
    partial_pose = normalize_detection(partial, frame_index=11)
    assert reference is not None and partial_pose is not None
    match = compare_poses(reference, partial_pose)
    assert match.feedback == "Partial", match
    assert match.score < 70.0, match


def test_temporal_lag_matching() -> None:
    matcher = TemporalPoseMatcher(max_lag_frames=5, keypoint_conf=0.2)
    for frame in range(20, 26):
        matcher.push_reference(make_result(frame, make_detection(arm_delta=(frame - 20) * 5)))
    user_result = make_result(99, make_detection(arm_delta=10))
    match = matcher.match_user(user_result)
    assert match.score > 98.0, match
    assert match.lag_frames == 3, match


def test_main_dancer_tracking_prefers_continuity() -> None:
    tracker = PoseStreamTracker(smoothing=0.0)
    first = make_detection(offset_x=0)
    distractor = make_detection(offset_x=300)
    first_result = FramePoseResult(0, [first, distractor], 0, 0.0, 0.0)
    tracked = tracker.update(first_result)
    assert tracked.main_index == 0

    moved_main = make_detection(offset_x=8)
    tempting_distractor = PoseDetection(
        keypoints=distractor.keypoints,
        box=distractor.box,
        box_confidence=1.0,
        visible_count=17,
        main_score=10.0,
    )
    second_result = FramePoseResult(1, [moved_main, tempting_distractor], 1, 0.0, 0.0)
    tracked = tracker.update(second_result)
    assert tracked.main_index == 0, tracked


def main() -> int:
    test_spatial_normalization()
    test_wrong_pose_is_not_perfect()
    test_partial_pose_is_quality_gated()
    test_temporal_lag_matching()
    test_main_dancer_tracking_prefers_continuity()
    print("pose_scoring_tests_ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
