"""Pure-logic checks for the integrated Part Three scoring system."""

from __future__ import annotations

import unittest

import numpy as np

from VisualComputingProject.bonus_level.src import config
from VisualComputingProject.bonus_level.src.pose_features import add_joint_angles
from VisualComputingProject.bonus_level.src.pose_similarity import (
    HoldStateFilter,
    compare_poses,
)
from VisualComputingProject.bonus_level.src.pose_normalizer import normalize_pose
from VisualComputingProject.bonus_level.src.pose_types import PoseFrame
from VisualComputingProject.bonus_level.src.temporal_alignment import TemporalAligner


def make_frame(
    timestamp: float,
    frame_index: int,
    wrist_shift: float = 0.0,
) -> PoseFrame:
    keypoints = np.zeros((17, 2), dtype=np.float32)
    keypoints[5] = (-0.2, -0.3)
    keypoints[6] = (0.2, -0.3)
    keypoints[7] = (-0.45, -0.1)
    keypoints[8] = (0.45, -0.1)
    keypoints[9] = (-0.65 + wrist_shift, 0.1)
    keypoints[10] = (0.65, 0.1)
    keypoints[11] = (-0.15, 0.25)
    keypoints[12] = (0.15, 0.25)
    keypoints[13] = (-0.2, 0.65)
    keypoints[14] = (0.2, 0.65)
    keypoints[15] = (-0.25, 1.0)
    keypoints[16] = (0.25, 1.0)

    frame = PoseFrame(
        timestamp=timestamp,
        frame_index=frame_index,
        source="test",
        bbox=(-1.0, -1.0, 1.0, 1.0),
        keypoints=keypoints.copy(),
        confidences=np.ones(17, dtype=np.float32),
        valid_mask=np.ones(17, dtype=bool),
        normalized_keypoints=keypoints.copy(),
    )
    add_joint_angles(frame)
    return frame


class TemporalAlignmentTests(unittest.TestCase):
    def test_identical_pose_is_near_100(self):
        result = compare_poses(make_frame(0.0, 1), make_frame(0.0, 2))
        self.assertGreater(result["score"], 99.9)

    def test_compare_poses_returns_error_details(self):
        result = compare_poses(
            make_frame(0.0, 1),
            make_frame(0.0, 2, wrist_shift=0.8),
            allow_mirror=False,
        )
        self.assertIn(9, result["error_keypoints"])
        self.assertTrue(result["error_summary"])
        self.assertLess(result["score"], 100.0)

    def test_translation_and_scale_are_removed(self):
        reference = make_frame(0.0, 1)
        user = make_frame(0.0, 2)
        reference.keypoints = reference.keypoints * 120.0 + 300.0
        user.keypoints = user.keypoints * 408.0 + np.array(
            [721.0, 477.0], dtype=np.float32
        )
        reference.normalized_keypoints = None
        user.normalized_keypoints = None
        normalize_pose(reference)
        normalize_pose(user)
        result = compare_poses(reference, user, allow_mirror=False)
        self.assertGreater(result["score"], 99.9)

    def test_partial_pose_receives_coverage_penalty(self):
        reference = make_frame(0.0, 1)
        user = make_frame(0.0, 2)
        mask = np.zeros(17, dtype=bool)
        mask[[0, 1, 2, 3, 4, 5, 6, 7, 9]] = True
        reference.valid_mask = mask.copy()
        user.valid_mask = mask.copy()
        result = compare_poses(reference, user, allow_mirror=False)
        self.assertAlmostEqual(result["coverage"], 4.0 / 12.0)
        self.assertGreater(result["score"], 70.0)
        self.assertLess(result["score"], 75.0)

    def test_user_buffer_keeps_recent_frames_only(self):
        aligner = TemporalAligner()
        aligner.add_reference(make_frame(0.0, 1))
        for index in range(config.USER_BUFFER_FRAME_COUNT + 5):
            aligner.match(make_frame(index * 0.01, index + 1))
        self.assertEqual(len(aligner.user_frames), config.USER_BUFFER_FRAME_COUNT)
        self.assertEqual(aligner.user_frames[0].frame_index, 6)

    def test_delay_search_uses_past_reference_not_future_reference(self):
        aligner = TemporalAligner()
        aligner.add_reference(make_frame(0.0, 1))
        aligner.add_reference(make_frame(0.2, 2, wrist_shift=0.8))
        aligner.add_reference(make_frame(0.7, 3, wrist_shift=-0.8))

        matched, result, lag = aligner.match(
            make_frame(0.6, 10, wrist_shift=0.8)
        )
        self.assertIsNotNone(matched)
        self.assertEqual(result["matched_reference_frame"], 2)
        self.assertAlmostEqual(lag, 0.4, places=6)
        self.assertGreater(result["score"], 99.9)

    def test_matching_motion_is_rewarded(self):
        aligner = TemporalAligner()
        aligner.add_reference(make_frame(0.0, 1))
        aligner.match(make_frame(0.4, 10))
        aligner.add_reference(make_frame(0.4, 2, wrist_shift=-0.8))

        _, result, lag = aligner.match(
            make_frame(0.8, 11, wrist_shift=-0.8)
        )
        self.assertTrue(result["motion_used"])
        self.assertGreater(result["motion"], 99.0)
        self.assertGreater(result["score"], 99.0)
        self.assertAlmostEqual(lag, 0.4, places=6)

    def test_static_player_is_penalized_while_reference_moves(self):
        aligner = TemporalAligner()
        aligner.add_reference(make_frame(0.0, 1))
        static_pose = make_frame(0.4, 10, wrist_shift=-0.8)
        aligner.match(static_pose)
        aligner.add_reference(make_frame(0.4, 2, wrist_shift=-0.8))

        _, result, _ = aligner.match(
            make_frame(0.8, 11, wrist_shift=-0.8)
        )
        self.assertTrue(result["motion_used"])
        self.assertLess(result["player_motion"], 1e-6)
        self.assertLess(result["score"], 55.0)
        self.assertEqual(result["feedback"], "Move!")

    def test_hold_filter_requires_confirmation_and_uses_hysteresis(self):
        hold_filter = HoldStateFilter()
        self.assertFalse(hold_filter.update(0.05, True))
        self.assertTrue(hold_filter.update(0.05, True))
        for _ in range(5):
            self.assertTrue(hold_filter.update(0.08, True))
        for _ in range(5):
            hold_filter.update(0.12, True)
        self.assertFalse(hold_filter.active)


if __name__ == "__main__":
    unittest.main()
