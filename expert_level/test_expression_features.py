"""Deterministic tests for explicit facial-landmark geometry features."""

import unittest

import numpy as np

from expression_features import (
    GEOMETRY_FEATURE_GROUPS,
    GEOMETRY_FEATURE_NAMES,
    append_landmark_geometry,
    append_landmark_geometry_groups,
    landmark_geometry_features,
)


class LandmarkGeometryFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.features = rng.normal(0.0, 0.4, size=(4, 136)).astype(np.float32)

    def test_geometry_shape_and_finiteness(self) -> None:
        geometry = landmark_geometry_features(self.features)
        self.assertEqual(geometry.shape, (4, len(GEOMETRY_FEATURE_NAMES)))
        self.assertTrue(np.isfinite(geometry).all())

    def test_single_row_matches_batch_row(self) -> None:
        single = landmark_geometry_features(self.features[0])
        batch = landmark_geometry_features(self.features[:1])
        np.testing.assert_allclose(single, batch)

    def test_append_preserves_original_coordinates(self) -> None:
        combined = append_landmark_geometry(self.features)
        self.assertEqual(
            combined.shape,
            (4, 136 + len(GEOMETRY_FEATURE_NAMES)),
        )
        np.testing.assert_array_equal(combined[:, :136], self.features)

    def test_grouped_append_uses_only_requested_features(self) -> None:
        groups = ("brow", "mouth")
        combined = append_landmark_geometry_groups(self.features, groups)
        expected_count = sum(len(GEOMETRY_FEATURE_GROUPS[group]) for group in groups)
        self.assertEqual(combined.shape, (4, 136 + expected_count))
        np.testing.assert_array_equal(combined[:, :136], self.features)

    def test_all_groups_match_full_geometry_order(self) -> None:
        grouped = append_landmark_geometry_groups(
            self.features,
            tuple(GEOMETRY_FEATURE_GROUPS),
        )
        full = append_landmark_geometry(self.features)
        np.testing.assert_allclose(grouped, full)

    def test_unknown_or_repeated_group_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            append_landmark_geometry_groups(self.features, ("unknown",))
        with self.assertRaises(ValueError):
            append_landmark_geometry_groups(self.features, ("eyes", "eyes"))


if __name__ == "__main__":
    unittest.main()
