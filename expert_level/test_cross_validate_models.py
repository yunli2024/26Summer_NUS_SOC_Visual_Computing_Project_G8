"""Deterministic tests for train-only expression cross-validation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cross_validate_models import (
    Candidate,
    cross_validate,
    default_candidates,
    load_train_split,
    summarize,
    validate_training_labels,
)


class CrossValidationTests(unittest.TestCase):
    def test_default_candidates_include_pca_and_geometry(self) -> None:
        candidates = default_candidates()
        self.assertTrue(any(item.pca_variance == 0.95 for item in candidates))
        self.assertTrue(any(item.geometry for item in candidates))

    def test_feature_loader_never_requires_test_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train_only.npz"
            np.savez_compressed(
                path,
                train_X=np.zeros((14, 136), dtype=np.float32),
                train_y=np.repeat(np.asarray(["a", "b"]), 7),
                metadata=json.dumps({"split": "train"}),
            )
            train_x, train_y, metadata = load_train_split(path)
        self.assertEqual(train_x.shape, (14, 136))
        self.assertEqual(train_y.shape, (14,))
        self.assertEqual(metadata["split"], "train")

    def test_cross_validation_covers_every_fold(self) -> None:
        rng = np.random.default_rng(7)
        labels = np.repeat(np.asarray(["a", "b", "c"]), 9)
        features = rng.normal(size=(len(labels), 136)).astype(np.float32)
        features[:, 0] += np.repeat(np.asarray([-3.0, 0.0, 3.0]), 9)
        candidate = Candidate("smoke", False, None, 1.0, 1.0)
        results = cross_validate(
            features,
            labels,
            (candidate,),
            folds=3,
            cache_mb=64.0,
            seed=11,
        )
        self.assertEqual([row.fold for row in results], [1, 2, 3])
        self.assertEqual(sum(row.validation_samples for row in results), 27)
        self.assertTrue(all(np.isfinite(row.macro_f1) for row in results))

    def test_summary_records_stability_and_class_balance(self) -> None:
        candidate = Candidate("smoke", False, None, 1.0, 1.0)
        rng = np.random.default_rng(9)
        labels = np.repeat(np.asarray(["a", "b"]), 8)
        features = rng.normal(size=(16, 136)).astype(np.float32)
        results = cross_validate(
            features,
            labels,
            (candidate,),
            folds=2,
            cache_mb=64.0,
            seed=4,
        )
        summary = summarize(results, (candidate,))
        self.assertEqual(summary[0]["rank"], 1)
        self.assertIn("std", summary[0]["macro_f1"])
        self.assertIn("worst_class_recall", summary[0])

    def test_formal_run_requires_all_seven_classes(self) -> None:
        incomplete = np.repeat(np.asarray(["happy", "neutral"]), 5)
        with self.assertRaisesRegex(ValueError, "exactly the seven FER classes"):
            validate_training_labels(incomplete, folds=5)


if __name__ == "__main__":
    unittest.main()
