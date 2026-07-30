"""Unit tests for primary-dancer selection, continuity, and smoothing."""

from __future__ import annotations

import unittest

import numpy as np

from pose_analyzer import PoseDetection, PrimaryDancerTracker


FRAME_SHAPE = (1000, 1000, 3)


def detection(
    box: tuple[float, float, float, float],
    point_value: float = 10.0,
    confidence: float = 0.9,
) -> PoseDetection:
    return PoseDetection(
        box=np.asarray(box, dtype=np.float32),
        box_confidence=confidence,
        points=np.full((17, 2), point_value, dtype=np.float32),
        keypoint_confidence=np.full(17, 0.95, dtype=np.float32),
    )


class PrimaryDancerTrackerTests(unittest.TestCase):
    def test_initial_selection_prefers_central_large_person(self) -> None:
        tracker = PrimaryDancerTracker()
        edge_person = detection((20, 250, 220, 650))
        central_person = detection((330, 180, 670, 820))

        chosen, _, _, scores = tracker.select(
            [edge_person, central_person], FRAME_SHAPE
        )

        self.assertIs(chosen, central_person)
        self.assertGreater(scores[1], scores[0])

    def test_identity_continuity_beats_new_distant_competitor(self) -> None:
        tracker = PrimaryDancerTracker()
        original = detection((300, 200, 600, 800), point_value=10.0)
        tracker.select([original], FRAME_SHAPE)
        continued = detection((320, 200, 620, 800), point_value=12.0)
        distant_competitor = detection((600, 100, 1000, 900), point_value=30.0)

        chosen, _, _, scores = tracker.select(
            [continued, distant_competitor], FRAME_SHAPE
        )

        self.assertIs(chosen, continued)
        self.assertGreater(scores[0], scores[1])

    def test_landmark_smoothing_reuses_valid_previous_points(self) -> None:
        tracker = PrimaryDancerTracker(smoothing_alpha=0.6)
        tracker.select([detection((300, 200, 600, 800), 10.0)], FRAME_SHAPE)

        _, points, valid, _ = tracker.select(
            [detection((300, 200, 600, 800), 20.0)], FRAME_SHAPE
        )

        self.assertTrue(np.all(valid))
        np.testing.assert_allclose(points, 16.0)

    def test_tracker_resets_after_nine_missed_frames(self) -> None:
        tracker = PrimaryDancerTracker()
        tracker.select([detection((300, 200, 600, 800))], FRAME_SHAPE)

        for _ in range(9):
            tracker.select([], FRAME_SHAPE)

        self.assertIsNone(tracker.previous_box)
        self.assertIsNone(tracker.previous_points)
        self.assertEqual(tracker.missed_frames, 0)


if __name__ == "__main__":
    unittest.main()
