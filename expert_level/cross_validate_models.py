"""Cross-validate fixed expression pipelines using the official train split only.

This is the model-confirmation stage after candidate tuning. It never reads the
official test arrays, so the held-out test split remains reserved for the final
one-time evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from expression_features import (
    CLASS_NAMES,
    GEOMETRY_FEATURE_NAMES,
    append_landmark_geometry,
)


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DEFAULT_FEATURES = BASE_DIR / "artifacts" / "fer_landmark_features.npz"
DEFAULT_OUTPUT = BASE_DIR / "artifacts_cross_validation"


@dataclass(frozen=True)
class Candidate:
    name: str
    geometry: bool
    pca_variance: float | None
    c: float
    gamma_multiplier: float


@dataclass(frozen=True)
class FoldResult:
    candidate: str
    fold: int
    train_samples: int
    validation_samples: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    balanced_accuracy: float
    worst_class_recall: float
    fit_seconds: float
    prediction_ms_per_sample: float
    pca_components: int | None


def default_candidates() -> tuple[Candidate, ...]:
    """Return fixed candidates, including the course-required PCA comparison."""
    return (
        Candidate("coordinate_svm", False, None, 10.0, 2.0),
        Candidate("pca95_svm", False, 0.95, 10.0, 1.0),
        Candidate("geometry_svm", True, None, 10.0, 2.0),
    )


def build_pipeline(candidate: Candidate, cache_mb: float, seed: int) -> Pipeline:
    steps: list[tuple[str, object]] = []
    feature_count = 136
    if candidate.geometry:
        steps.append(
            (
                "geometry",
                FunctionTransformer(append_landmark_geometry, validate=False),
            )
        )
        feature_count += len(GEOMETRY_FEATURE_NAMES)
    steps.append(("scaler", StandardScaler()))
    if candidate.pca_variance is not None:
        steps.append(
            (
                "pca",
                PCA(
                    n_components=candidate.pca_variance,
                    svd_solver="full",
                    random_state=seed,
                ),
            )
        )
        # PCA changes dimensionality at fit time. "scale" computes gamma from
        # fold-local transformed training data without validation leakage.
        gamma: float | str = "scale"
    else:
        gamma = candidate.gamma_multiplier / feature_count
    steps.append(
        (
            "svm",
            SVC(
                C=candidate.c,
                kernel="rbf",
                gamma=gamma,
                class_weight="balanced",
                cache_size=cache_mb,
                decision_function_shape="ovr",
                random_state=seed,
            ),
        )
    )
    return Pipeline(steps)


def load_train_split(feature_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load only train arrays; test arrays are deliberately never accessed."""
    with np.load(feature_path, allow_pickle=False) as data:
        train_x = data["train_X"].astype(np.float32)
        train_y = data["train_y"]
        metadata = json.loads(str(data["metadata"]))
    if train_x.ndim != 2 or train_x.shape[1] != 136:
        raise ValueError(f"Expected train_X with shape (n, 136), got {train_x.shape}.")
    if train_y.ndim != 1 or len(train_y) != len(train_x):
        raise ValueError("train_y must be one-dimensional and match train_X.")
    return train_x, train_y, metadata


def portable_project_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_DIR.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def validate_training_labels(
    train_y: np.ndarray,
    folds: int,
) -> dict[str, int]:
    labels, counts = np.unique(train_y, return_counts=True)
    class_counts = {
        str(label): int(count)
        for label, count in zip(labels, counts)
    }
    expected = set(CLASS_NAMES)
    observed = set(class_counts)
    if observed != expected:
        missing = sorted(expected - observed)
        unexpected = sorted(observed - expected)
        raise ValueError(
            "Training labels must contain exactly the seven FER classes; "
            f"missing={missing}, unexpected={unexpected}."
        )
    too_small = {
        label: count
        for label, count in class_counts.items()
        if count < folds
    }
    if too_small:
        raise ValueError(
            f"Every class needs at least {folds} samples for {folds}-fold "
            f"stratification; too small={too_small}."
        )
    return class_counts


def cross_validate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    candidates: tuple[Candidate, ...],
    folds: int,
    cache_mb: float,
    seed: int,
) -> list[FoldResult]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(train_x, train_y))
    results: list[FoldResult] = []
    for candidate in candidates:
        for fold, (fit_indices, validation_indices) in enumerate(splits, start=1):
            model = build_pipeline(candidate, cache_mb, seed + fold)
            started = time.perf_counter()
            model.fit(train_x[fit_indices], train_y[fit_indices])
            fit_seconds = time.perf_counter() - started

            started = time.perf_counter()
            predictions = model.predict(train_x[validation_indices])
            prediction_ms = (
                (time.perf_counter() - started)
                * 1000.0
                / len(validation_indices)
            )
            recalls = recall_score(
                train_y[validation_indices],
                predictions,
                labels=np.unique(train_y),
                average=None,
                zero_division=0,
            )
            pca = model.named_steps.get("pca")
            result = FoldResult(
                candidate=candidate.name,
                fold=fold,
                train_samples=int(len(fit_indices)),
                validation_samples=int(len(validation_indices)),
                accuracy=float(
                    accuracy_score(train_y[validation_indices], predictions)
                ),
                macro_f1=float(
                    f1_score(
                        train_y[validation_indices],
                        predictions,
                        average="macro",
                    )
                ),
                weighted_f1=float(
                    f1_score(
                        train_y[validation_indices],
                        predictions,
                        average="weighted",
                    )
                ),
                balanced_accuracy=float(
                    balanced_accuracy_score(
                        train_y[validation_indices],
                        predictions,
                    )
                ),
                worst_class_recall=float(np.min(recalls)),
                fit_seconds=float(fit_seconds),
                prediction_ms_per_sample=float(prediction_ms),
                pca_components=(
                    int(pca.n_components_) if pca is not None else None
                ),
            )
            results.append(result)
            print(
                f"{candidate.name} fold {fold}/{folds}: "
                f"Macro-F1={result.macro_f1:.4f}, "
                f"balanced accuracy={result.balanced_accuracy:.4f}, "
                f"prediction={result.prediction_ms_per_sample:.3f} ms",
                flush=True,
            )
    return results


def summarize(
    results: list[FoldResult],
    candidates: tuple[Candidate, ...],
) -> list[dict]:
    summaries: list[dict] = []
    for candidate in candidates:
        rows = [row for row in results if row.candidate == candidate.name]
        if not rows:
            raise ValueError(f"No fold results for candidate {candidate.name}.")

        def stats(field: str) -> dict[str, float]:
            values = np.asarray(
                [getattr(row, field) for row in rows],
                dtype=np.float64,
            )
            return {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }

        summaries.append(
            {
                "candidate": candidate.name,
                "configuration": asdict(candidate),
                "folds": len(rows),
                "accuracy": stats("accuracy"),
                "macro_f1": stats("macro_f1"),
                "weighted_f1": stats("weighted_f1"),
                "balanced_accuracy": stats("balanced_accuracy"),
                "worst_class_recall": stats("worst_class_recall"),
                "fit_seconds": stats("fit_seconds"),
                "prediction_ms_per_sample": stats(
                    "prediction_ms_per_sample"
                ),
                "pca_components": sorted(
                    {
                        row.pca_components
                        for row in rows
                        if row.pca_components is not None
                    }
                ),
            }
        )
    # Primary: Macro-F1. Tie-breakers reflect stability, class balance, then
    # latency; accuracy is reported but intentionally not a selection key.
    summaries.sort(
        key=lambda row: (
            -row["macro_f1"]["mean"],
            row["macro_f1"]["std"],
            -row["balanced_accuracy"]["mean"],
            -row["worst_class_recall"]["mean"],
            row["prediction_ms_per_sample"]["mean"],
        )
    )
    for rank, row in enumerate(summaries, start=1):
        row["rank"] = rank
    return summaries


def save_outputs(
    output: Path,
    results: list[FoldResult],
    summaries: list[dict],
    feature_path: Path,
    metadata: dict,
    class_counts: dict[str, int],
    folds: int,
    seed: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fold_path = output / "cross_validation_folds.csv"
    with fold_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(row) for row in results)
    report = {
        "protocol": {
            "split": "official train only",
            "official_test_accessed": False,
            "splitter": "StratifiedKFold",
            "folds": folds,
            "shuffle": True,
            "seed": seed,
            "training_samples": int(sum(class_counts.values())),
            "class_counts": class_counts,
            "selection_order": [
                "macro_f1_mean_desc",
                "macro_f1_std_asc",
                "balanced_accuracy_mean_desc",
                "worst_class_recall_mean_desc",
                "prediction_ms_per_sample_mean_asc",
            ],
        },
        "feature_cache": portable_project_path(feature_path),
        "extraction": metadata,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
        "candidates": summaries,
    }
    (output / "cross_validation.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--cache-mb", type=float, default=2048.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.folds < 2:
            raise ValueError("--folds must be at least 2.")
        if not args.features.is_file():
            raise FileNotFoundError(
                f"Feature cache not found: {args.features}. Restore the FER "
                "dataset and run task1_pipeline.py --stage extract first."
            )
        train_x, train_y, metadata = load_train_split(args.features)
        class_counts = validate_training_labels(train_y, args.folds)
        candidates = default_candidates()
        results = cross_validate(
            train_x,
            train_y,
            candidates,
            args.folds,
            args.cache_mb,
            args.seed,
        )
        summaries = summarize(results, candidates)
        save_outputs(
            args.output,
            results,
            summaries,
            args.features,
            metadata,
            class_counts,
            args.folds,
            args.seed,
        )
        print(
            f"Selected by train-only cross-validation: "
            f"{summaries[0]['candidate']} "
            f"(Macro-F1 {summaries[0]['macro_f1']['mean']:.4f} "
            f"+/- {summaries[0]['macro_f1']['std']:.4f})",
            flush=True,
        )
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
