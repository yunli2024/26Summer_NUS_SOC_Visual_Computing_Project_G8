"""Pure-logic checks for Bonus Level scoring and temporal alignment."""

from __future__ import annotations

import unittest

import numpy as np

from VisualComputingProject.bonus_level.src import config
from VisualComputingProject.bonus_level.src.pose_features import add_joint_angles
from VisualComputingProject.bonus_level.src.pose_similarity import compare_poses
from VisualComputingProject.bonus_level.src.pose_types import PoseFrame
from VisualComputingProject.bonus_level.src.temporal_alignment import TemporalAligner


def make_frame(timestamp: float, frame_index: int, wrist_shift: float = 0.0) -> PoseFrame:
    keypoints = np.zeros((17, 2), dtype=np.float32)
    for idx in range(17):
        keypoints[idx] = (idx * 0.03, idx * 0.02)
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
    def test_compare_poses_returns_error_details(self):
        result = compare_poses(make_frame(0.0, 1), make_frame(0.0, 2, wrist_shift=0.8))

        self.assertIn(9, result["error_keypoints"])
        self.assertTrue(result["error_summary"])
        self.assertLess(result["score"], 100.0)

    def test_user_buffer_keeps_recent_frames_only(self):
        aligner = TemporalAligner()
        aligner.add_reference(make_frame(0.0, 1))

        for idx in range(config.USER_BUFFER_FRAME_COUNT + 5):
            aligner.match(make_frame(idx * 0.01, idx + 1))

        self.assertEqual(len(aligner.user_frames), config.USER_BUFFER_FRAME_COUNT)
        self.assertEqual(aligner.user_frames[0].frame_index, 6)

    def test_match_can_select_best_recent_user_frame(self):
        aligner = TemporalAligner()
        aligner.add_reference(make_frame(0.0, 1))

        aligner.match(make_frame(0.05, 10))
        _, result, _ = aligner.match(make_frame(0.06, 11, wrist_shift=0.8))

        self.assertEqual(result["matched_user_frame"], 10)
        self.assertGreater(result["score"], 90.0)


if __name__ == "__main__":
    unittest.main()
