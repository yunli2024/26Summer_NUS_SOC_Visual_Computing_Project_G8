"""Tune RBF-SVM C/gamma without using the official test set for selection."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

from expression_features import (
    CLASS_NAMES,
    FEATURE_VERSION,
    GEOMETRY_FEATURE_NAMES,
    append_landmark_geometry,
)
from task1_pipeline import save_confusion_matrices, save_failure_cases


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES = BASE_DIR / "artifacts" / "fer_landmark_features.npz"
DEFAULT_OUTPUT = BASE_DIR / "artifacts_svm_tuned"


@dataclass(frozen=True)
class TrialResult:
    stage: str
    c: float
    gamma: float
    gamma_multiplier: float
    train_samples: int
    validation_samples: int
    accuracy: float
    macro_f1: float
    weighted_f1: float
    training_seconds: float
    prediction_seconds: float
    support_vectors: int


def stratified_indices(labels: np.ndarray, count: int, seed: int) -> np.ndarray:
    indices = np.arange(len(labels))
    if count <= 0 or count >= len(indices):
        return indices
    selected, _ = train_test_split(
        indices,
        train_size=count,
        stratify=labels,
        random_state=seed,
    )
    return np.asarray(selected, dtype=np.int64)


def fit_trial(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    stage: str,
    c: float,
    gamma: float,
    gamma_multiplier: float,
    cache_mb: float,
    seed: int,
) -> TrialResult:
    model = SVC(
        C=c,
        kernel="rbf",
        gamma=gamma,
        class_weight="balanced",
        cache_size=cache_mb,
        decision_function_shape="ovr",
        random_state=seed,
    )
    started = time.perf_counter()
    model.fit(train_x, train_y)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions = model.predict(validation_x)
    prediction_seconds = time.perf_counter() - started
    return TrialResult(
        stage=stage,
        c=float(c),
        gamma=float(gamma),
        gamma_multiplier=float(gamma_multiplier),
        train_samples=int(len(train_x)),
        validation_samples=int(len(validation_x)),
        accuracy=float(accuracy_score(validation_y, predictions)),
        macro_f1=float(f1_score(validation_y, predictions, average="macro")),
        weighted_f1=float(f1_score(validation_y, predictions, average="weighted")),
        training_seconds=float(training_seconds),
        prediction_seconds=float(prediction_seconds),
        support_vectors=int(np.sum(model.n_support_)),
    )


def run_trials(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    candidates: list[tuple[float, float, float]],
    stage: str,
    cache_mb: float,
    seed: int,
    n_jobs: int,
) -> list[TrialResult]:
    def submit_candidate(candidate: tuple[float, float, float]) -> TrialResult:
        c, gamma, multiplier = candidate
        return fit_trial(
            train_x,
            train_y,
            validation_x,
            validation_y,
            stage,
            c,
            gamma,
            multiplier,
            cache_mb,
            seed,
        )

    if n_jobs <= 1:
        results = [submit_candidate(candidate) for candidate in candidates]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(submit_candidate, candidate): candidate
                for candidate in candidates
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"[{stage}] completed C={result.c:g}, "
                    f"gamma={result.gamma_multiplier:g}x, "
                    f"macro_f1={result.macro_f1:.4f}",
                    flush=True,
                )
    for result in sorted(results, key=lambda item: (-item.macro_f1, -item.accuracy)):
        print(
            f"[{stage}] C={result.c:g}, gamma={result.gamma:.7f} "
            f"({result.gamma_multiplier:g}x): accuracy={result.accuracy:.4f}, "
            f"macro_f1={result.macro_f1:.4f}, train={result.training_seconds:.1f}s, "
            f"SV={result.support_vectors}",
            flush=True,
        )
    return results


def save_results_csv(results: list[TrialResult], path: Path) -> None:
    fieldnames = list(asdict(results[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def save_heatmap(
    coarse_results: list[TrialResult],
    c_values: list[float],
    gamma_multipliers: list[float],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = np.full((len(c_values), len(gamma_multipliers)), np.nan, dtype=np.float32)
    for result in coarse_results:
        row = c_values.index(result.c)
        column = gamma_multipliers.index(result.gamma_multiplier)
        values[row, column] = result.macro_f1

    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    image = axis.imshow(values, cmap="YlGnBu", vmin=float(np.nanmin(values)), vmax=float(np.nanmax(values)))
    axis.set_xticks(range(len(gamma_multipliers)), [f"{value:g}x" for value in gamma_multipliers])
    axis.set_yticks(range(len(c_values)), [f"{value:g}" for value in c_values])
    axis.set_xlabel("Gamma multiplier (base = 1 / feature count)")
    axis.set_ylabel("C")
    axis.set_title("Coarse validation Macro F1")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center")
    figure.colorbar(image, ax=axis, label="Macro F1")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    if not args.features.is_file():
        raise FileNotFoundError(f"Feature cache not found: {args.features}")
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5.")
    if args.top_k < 1:
        raise ValueError("--top-k must be positive.")

    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.features, allow_pickle=False) as data:
        raw_train_x = data["train_X"].astype(np.float32)
        train_y = data["train_y"]
        raw_test_x = data["test_X"].astype(np.float32)
        test_y = data["test_y"]
        test_paths = data["test_paths"]
        metadata = json.loads(str(data["metadata"]))

    search_train_x = (
        append_landmark_geometry(raw_train_x) if args.geometry else raw_train_x
    )
    fit_x, validation_x, fit_y, validation_y = train_test_split(
        search_train_x,
        train_y,
        test_size=args.validation_fraction,
        stratify=train_y,
        random_state=args.seed,
    )
    coarse_indices = stratified_indices(fit_y, args.coarse_samples, args.seed)
    coarse_x = fit_x[coarse_indices]
    coarse_y = fit_y[coarse_indices]

    coarse_scaler = StandardScaler()
    coarse_scaled = coarse_scaler.fit_transform(coarse_x)
    coarse_validation_scaled = coarse_scaler.transform(validation_x)
    feature_count = coarse_scaled.shape[1]
    base_gamma = 1.0 / feature_count
    candidates = [
        (c, base_gamma * multiplier, multiplier)
        for c in args.c_values
        for multiplier in args.gamma_multipliers
    ]

    print(
        f"Official train/test: {len(raw_train_x)}/{len(raw_test_x)}; "
        f"internal fit/validation: {len(fit_x)}/{len(validation_x)}",
        flush=True,
    )
    print(
        f"Features: {raw_train_x.shape[1]} landmark coordinates"
        + (
            f" + {len(GEOMETRY_FEATURE_NAMES)} explicit geometry"
            if args.geometry
            else ""
        )
        + f" = {feature_count}",
        flush=True,
    )
    print(
        f"Coarse search: {len(coarse_x)} samples, {len(candidates)} candidates, "
        f"{args.n_jobs} parallel workers",
        flush=True,
    )
    coarse_results = run_trials(
        coarse_scaled,
        coarse_y,
        coarse_validation_scaled,
        validation_y,
        candidates,
        "coarse",
        args.cache_mb,
        args.seed,
        args.n_jobs,
    )
    ranked_coarse = sorted(
        coarse_results,
        key=lambda item: (-item.macro_f1, -item.accuracy, item.training_seconds),
    )
    finalist_candidates = [
        (result.c, result.gamma, result.gamma_multiplier)
        for result in ranked_coarse[: min(args.top_k, len(ranked_coarse))]
    ]

    full_scaler = StandardScaler()
    full_fit_scaled = full_scaler.fit_transform(fit_x)
    full_validation_scaled = full_scaler.transform(validation_x)
    print(
        f"Full validation: {len(full_fit_scaled)} samples, "
        f"{len(finalist_candidates)} finalists",
        flush=True,
    )
    finalist_results = run_trials(
        full_fit_scaled,
        fit_y,
        full_validation_scaled,
        validation_y,
        finalist_candidates,
        "full_validation",
        args.cache_mb,
        args.seed,
        args.n_jobs,
    )
    best = sorted(
        finalist_results,
        key=lambda item: (-item.macro_f1, -item.accuracy, item.training_seconds),
    )[0]
    print(
        f"Selected C={best.c:g}, gamma={best.gamma:.7f} "
        f"from validation Macro F1={best.macro_f1:.4f}",
        flush=True,
    )

    final_steps = []
    if args.geometry:
        final_steps.append(
            (
                "geometry",
                FunctionTransformer(append_landmark_geometry, validate=False),
            )
        )
    final_steps.extend(
        [
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=best.c,
                    kernel="rbf",
                    gamma=best.gamma,
                    class_weight="balanced",
                    cache_size=args.final_cache_mb,
                    decision_function_shape="ovr",
                    random_state=args.seed,
                ),
            ),
        ]
    )
    final_model = Pipeline(final_steps)
    started = time.perf_counter()
    final_model.fit(raw_train_x, train_y)
    final_training_seconds = time.perf_counter() - started

    started = time.perf_counter()
    predictions = final_model.predict(raw_test_x)
    batch_ms = (time.perf_counter() - started) * 1000.0 / len(raw_test_x)
    timing_x = raw_test_x[: min(1000, len(raw_test_x))]
    started = time.perf_counter()
    for row in timing_x:
        final_model.predict(row.reshape(1, -1))
    single_ms = (time.perf_counter() - started) * 1000.0 / len(timing_x)

    report_dict = classification_report(
        test_y,
        predictions,
        labels=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        test_y,
        predictions,
        labels=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    all_results = coarse_results + finalist_results
    metrics = {
        "accuracy": float(accuracy_score(test_y, predictions)),
        "macro_f1": float(f1_score(test_y, predictions, average="macro")),
        "weighted_f1": float(f1_score(test_y, predictions, average="weighted")),
        "training_seconds": float(final_training_seconds),
        "batch_prediction_ms_per_image": float(batch_ms),
        "single_prediction_ms_per_image": float(single_ms),
        "meets_30ms_requirement": bool(single_ms < 30.0),
        "train_samples": int(len(raw_train_x)),
        "test_samples": int(len(raw_test_x)),
        "input_feature_count": int(raw_train_x.shape[1]),
        "feature_count": int(feature_count),
        "geometry_enabled": bool(args.geometry),
        "geometry_feature_count": (
            int(len(GEOMETRY_FEATURE_NAMES)) if args.geometry else 0
        ),
        "geometry_feature_names": (
            list(GEOMETRY_FEATURE_NAMES) if args.geometry else []
        ),
        "classifier": "svm_tuned_geometry" if args.geometry else "svm_tuned",
        "svm_c": float(best.c),
        "svm_gamma": float(best.gamma),
        "svm_gamma_multiplier": float(best.gamma_multiplier),
        "selection_metric": "validation_macro_f1",
        "validation_fraction": float(args.validation_fraction),
        "validation_samples": int(len(validation_x)),
        "coarse_samples": int(len(coarse_x)),
        "coarse_candidate_count": int(len(candidates)),
        "full_validation_candidate_count": int(len(finalist_candidates)),
        "best_validation": asdict(best),
        "versions": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
        "classification_report": report_dict,
        "tuning_results": [asdict(result) for result in all_results],
        "extraction": metadata,
    }
    model_bundle = {
        "model": final_model,
        "classes": list(CLASS_NAMES),
        "feature_version": FEATURE_VERSION,
        "feature_mode": (
            "coordinates_plus_geometry" if args.geometry else "coordinates"
        ),
        "geometry_feature_names": (
            list(GEOMETRY_FEATURE_NAMES) if args.geometry else []
        ),
        "image_size": metadata["image_size"],
        "center_inset": metadata["center_inset"],
    }
    joblib.dump(model_bundle, args.output / "expression_classifier.joblib")
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    (args.output / "classification_report.txt").write_text(report_text, encoding="utf-8")
    save_results_csv(all_results, args.output / "svm_tuning_results.csv")
    save_heatmap(
        coarse_results,
        list(args.c_values),
        list(args.gamma_multipliers),
        args.output / "tuning_heatmap.png",
    )
    save_confusion_matrices(test_y, predictions, args.output / "confusion_matrix.png")
    decision_scores = final_model.decision_function(raw_test_x)
    save_failure_cases(
        test_paths,
        test_y,
        predictions,
        decision_scores,
        args.output / "failure_cases.png",
    )

    print(report_text)
    print(f"Final Accuracy: {metrics['accuracy']:.4f}")
    print(f"Final Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Single-image prediction: {single_ms:.3f} ms")
    print(f"Saved tuned SVM artifacts to: {args.output}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--coarse-samples", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--geometry",
        action="store_true",
        help=(
            "Append 38 explicit facial-geometry descriptors to the 136 normalized "
            "landmark coordinates. The saved Pipeline still accepts 136-D input."
        ),
    )
    parser.add_argument("--c-values", type=float, nargs="+", default=(1.0, 3.0, 10.0, 30.0))
    parser.add_argument(
        "--gamma-multipliers",
        type=float,
        nargs="+",
        default=(0.5, 1.0, 2.0),
    )
    parser.add_argument("--cache-mb", type=float, default=768.0)
    parser.add_argument("--final-cache-mb", type=float, default=2048.0)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
