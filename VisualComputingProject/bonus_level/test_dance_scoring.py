"""Small deterministic tests for the Bonus Task 2 scoring algorithm."""

import unittest

import numpy as np

from dance_scoring import (
    HoldStateFilter,
    best_reference_match,
    mirror_pose,
    pose_similarity,
)


def sample_pose() -> np.ndarray:
    pose = np.zeros((17, 2), dtype=np.float32)
    pose[0] = (0.0, -2.5)
    pose[1:5] = (0.0, -2.3)
    pose[5], pose[6] = (-0.6, -1.6), (0.6, -1.6)
    pose[7], pose[8] = (-1.1, -0.9), (1.0, -1.1)
    pose[9], pose[10] = (-1.5, -1.7), (1.5, -0.5)
    pose[11], pose[12] = (-0.45, 0.0), (0.45, 0.0)
    pose[13], pose[14] = (-0.8, 1.2), (0.8, 1.0)
    pose[15], pose[16] = (-0.9, 2.4), (1.25, 2.1)
    return pose


class DanceScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pose = sample_pose()
        self.valid = np.ones(17, dtype=bool)

    def test_identical_pose_is_near_100(self) -> None:
        score = pose_similarity(self.pose, self.valid, self.pose, self.valid)
        self.assertGreater(score.score, 99.9)

    def test_translation_and_scale_do_not_change_score(self) -> None:
        transformed = self.pose * 3.4 + np.array([421.0, 177.0], dtype=np.float32)
        score = pose_similarity(transformed, self.valid, self.pose, self.valid)
        self.assertGreater(score.score, 99.9)

    def test_wrong_arm_is_penalized(self) -> None:
        changed = self.pose.copy()
        changed[9] = (0.1, 0.5)
        score = pose_similarity(changed, self.valid, self.pose, self.valid, allow_mirror=False)
        self.assertLess(score.score, 93.0)

    def test_mirrored_move_can_be_accepted(self) -> None:
        mirrored, mirrored_valid = mirror_pose(self.pose, self.valid)
        score = pose_similarity(mirrored, mirrored_valid, self.pose, self.valid, allow_mirror=True)
        self.assertGreater(score.score, 99.9)
        self.assertTrue(score.mirrored)

    def test_temporal_search_finds_delayed_pose(self) -> None:
        references = np.repeat(self.pose[None, ...], 6, axis=0)
        references[3, 9] = (-2.2, -0.2)
        references[4, 9] = (-2.2, 0.7)
        references[5, 9] = (-0.2, 1.0)
        valid = np.ones((6, 17), dtype=bool)
        player = references[3] * 2.0 + 100.0
        result = best_reference_match(player, self.valid, references, valid, 5, 3, False)
        self.assertIsNotNone(result)
        self.assertEqual(result.reference_index, 3)
        self.assertEqual(result.lag_frames, 2)
        self.assertGreater(result.score, 99.9)

    def test_matching_motion_scores_near_100(self) -> None:
        previous = self.pose.copy()
        current = self.pose.copy()
        current[7] = (-1.7, -1.2)
        current[9] = (-2.3, -2.1)
        references = np.stack([previous, current])
        valid = np.ones((2, 17), dtype=bool)
        result = best_reference_match(
            current,
            self.valid,
            references,
            valid,
            current_index=1,
            max_lag_frames=0,
            allow_mirror=False,
            player_previous_points=previous,
            player_previous_valid=self.valid,
            motion_delta_frames=1,
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.motion_used)
        self.assertGreater(result.motion_score, 99.9)
        self.assertGreater(result.score, 99.9)

    def test_static_player_is_penalized_while_reference_moves(self) -> None:
        previous = self.pose.copy()
        current = self.pose.copy()
        current[7] = (-1.7, -1.2)
        current[9] = (-2.3, -2.1)
        references = np.stack([previous, current])
        valid = np.ones((2, 17), dtype=bool)
        result = best_reference_match(
            current,
            self.valid,
            references,
            valid,
            current_index=1,
            max_lag_frames=0,
            allow_mirror=False,
            player_previous_points=current,
            player_previous_valid=self.valid,
            motion_delta_frames=1,
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.motion_used)
        self.assertGreater(result.reference_motion, 0.09)
        self.assertLess(result.player_motion, 1e-6)
        self.assertLess(result.score, 55.0)

    def test_static_hold_is_not_penalized(self) -> None:
        references = np.repeat(self.pose[None, ...], 2, axis=0)
        valid = np.ones((2, 17), dtype=bool)
        result = best_reference_match(
            self.pose,
            self.valid,
            references,
            valid,
            current_index=1,
            max_lag_frames=0,
            allow_mirror=False,
            player_previous_points=self.pose,
            player_previous_valid=self.valid,
            motion_delta_frames=1,
        )
        self.assertIsNotNone(result)
        self.assertTrue(result.motion_used)
        self.assertGreater(result.score, 99.9)

    def test_hold_filter_rejects_a_single_low_motion_sample(self) -> None:
        hold_filter = HoldStateFilter(ema_alpha=1.0)
        self.assertFalse(hold_filter.update(0.05))
        self.assertFalse(hold_filter.update(0.12))

    def test_hold_filter_uses_confirmation_and_hysteresis(self) -> None:
        hold_filter = HoldStateFilter(ema_alpha=1.0)
        self.assertFalse(hold_filter.update(0.05))
        self.assertTrue(hold_filter.update(0.05))
        self.assertTrue(hold_filter.update(0.08))
        self.assertFalse(hold_filter.update(0.11))


if __name__ == "__main__":
    unittest.main()
