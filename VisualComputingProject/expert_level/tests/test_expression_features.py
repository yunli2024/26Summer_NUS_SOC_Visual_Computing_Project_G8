"""Deterministic checks for the integrated Zhangyx geometry feature set."""

from __future__ import annotations

import unittest

import numpy as np

from VisualComputingProject.expert_level.src.expression_features import (
    GEOMETRY_FEATURE_GROUPS,
    GEOMETRY_FEATURE_NAMES,
    append_landmark_geometry,
    append_landmark_geometry_groups,
    landmark_geometry_features,
)


class ExpressionGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.features = rng.normal(0.0, 0.4, size=(4, 136)).astype(np.float32)

    def test_geometry_shape_and_finiteness(self) -> None:
        geometry = landmark_geometry_features(self.features)
        self.assertEqual(geometry.shape, (4, len(GEOMETRY_FEATURE_NAMES)))
        self.assertTrue(np.isfinite(geometry).all())

    def test_append_preserves_coordinates(self) -> None:
        combined = append_landmark_geometry(self.features)
        self.assertEqual(combined.shape, (4, 136 + len(GEOMETRY_FEATURE_NAMES)))
        np.testing.assert_array_equal(combined[:, :136], self.features)

    def test_group_selection_is_deterministic(self) -> None:
        groups = ("brow", "mouth")
        combined = append_landmark_geometry_groups(self.features, groups)
        expected = sum(len(GEOMETRY_FEATURE_GROUPS[group]) for group in groups)
        self.assertEqual(combined.shape, (4, 136 + expected))

    def test_unknown_and_repeated_groups_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            append_landmark_geometry_groups(self.features, ("unknown",))
        with self.assertRaises(ValueError):
            append_landmark_geometry_groups(self.features, ("eyes", "eyes"))


if __name__ == "__main__":
    unittest.main()
