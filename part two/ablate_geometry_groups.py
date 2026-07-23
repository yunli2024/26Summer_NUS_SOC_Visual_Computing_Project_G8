"""Grouped ablation study for the 38 explicit facial-geometry features.

The official test set is evaluated only once, after the best group subset has
been selected using an internal split of the official training set.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import asdict, dataclass
import json
import platform
from pathlib import Path
import time

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
    GEOMETRY_FEATURE_GROUPS,
    GEOMETRY_FEATURE_NAMES,
    append_landmark_geometry_groups,
    landmark_geometry_features,
)
from task1_pipeline import save_confusion_matrices, save_failure_cases


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES = BASE_DIR / "artifacts" / "fer_landmark_features.npz"
DEFAULT_OUTPUT = BASE_DIR / "artifacts_svm_geometry_ablation"
GROUP_ORDER = tuple(GEOMETRY_FEATURE_GROUPS)


@dataclass(frozen=True)
class Variant:
    name: str
    groups: tuple[str, ...]
    family: str


@dataclass(frozen=True)
class AblationResult:
    stage: str
    variant: str
    family: str
    groups: str
    geometry_features: int
    feature_count: int
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


def build_variants() -> list[Variant]:
    variants = [Variant("coordinates_only", (), "baseline")]
    variants.extend(
        Variant(f"add_{group}", (group,), "add_one")
        for group in GROUP_ORDER
    )
    variants.append(Variant("all_geometry", GROUP_ORDER, "full"))
    variants.extend(
        Variant(
            f"drop_{group}",
            tuple(candidate for candidate in GROUP_ORDER if candidate != group),
            "drop_one",
        )
        for group in GROUP_ORDER
    )
    return variants


def group_indices(groups: tuple[str, ...]) -> np.ndarray:
    name_to_index = {
        name: index for index, name in enumerate(GEOMETRY_FEATURE_NAMES)
    }
    names = [
        feature_name
        for group in groups
        for feature_name in GEOMETRY_FEATURE_GROUPS[group]
    ]
    return np.asarray([name_to_index[name] for name in names], dtype=np.int64)


def combine_features(
    coordinates: np.ndarray,
    all_geometry: np.ndarray,
    groups: tuple[str, ...],
) -> np.ndarray:
    indices = group_indices(groups)
    if indices.size == 0:
        return coordinates
    return np.concatenate((coordinates, all_geometry[:, indices]), axis=1)


def fit_variant(
    stage: str,
    variant: Variant,
    train_coordinates: np.ndarray,
    train_geometry: np.ndarray,
    train_labels: np.ndarray,
    validation_coordinates: np.ndarray,
    validation_geometry: np.ndarray,
    validation_labels: np.ndarray,
    c: float,
    gamma_multiplier: float,
    cache_mb: float,
    seed: int,
) -> AblationResult:
    train_x = combine_features(
        train_coordinates, train_geometry, variant.groups
    )
    validation_x = combine_features(
        validation_coordinates, validation_geometry, variant.groups
    )
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_x)
    validation_scaled = scaler.transform(validation_x)
    feature_count = int(train_scaled.shape[1])
    gamma = float(gamma_multiplier / feature_count)
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
    model.fit(train_scaled, train_labels)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions = model.predict(validation_scaled)
    prediction_seconds = time.perf_counter() - started
    return AblationResult(
        stage=stage,
        variant=variant.name,
        family=variant.family,
        groups=";".join(variant.groups) if variant.groups else "none",
        geometry_features=feature_count - 136,
        feature_count=feature_count,
        c=float(c),
        gamma=gamma,
        gamma_multiplier=float(gamma_multiplier),
        train_samples=int(len(train_scaled)),
        validation_samples=int(len(validation_scaled)),
        accuracy=float(accuracy_score(validation_labels, predictions)),
        macro_f1=float(f1_score(validation_labels, predictions, average="macro")),
        weighted_f1=float(
            f1_score(validation_labels, predictions, average="weighted")
        ),
        training_seconds=float(training_seconds),
        prediction_seconds=float(prediction_seconds),
        support_vectors=int(np.sum(model.n_support_)),
    )


def save_results(results: list[AblationResult], path: Path) -> None:
    ordered = sorted(
        results,
        key=lambda result: (
            0 if result.stage == "coarse" else 1,
            result.variant,
        ),
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=list(asdict(ordered[0]).keys()),
        )
        writer.writeheader()
        writer.writerows(asdict(result) for result in ordered)


def load_results(path: Path) -> list[AblationResult]:
    if not path.is_file():
        return []
    results: list[AblationResult] = []
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            results.append(
                AblationResult(
                    stage=row["stage"],
                    variant=row["variant"],
                    family=row["family"],
                    groups=row["groups"],
                    geometry_features=int(row["geometry_features"]),
                    feature_count=int(row["feature_count"]),
                    c=float(row["c"]),
                    gamma=float(row["gamma"]),
                    gamma_multiplier=float(row["gamma_multiplier"]),
                    train_samples=int(row["train_samples"]),
                    validation_samples=int(row["validation_samples"]),
                    accuracy=float(row["accuracy"]),
                    macro_f1=float(row["macro_f1"]),
                    weighted_f1=float(row["weighted_f1"]),
                    training_seconds=float(row["training_seconds"]),
                    prediction_seconds=float(row["prediction_seconds"]),
                    support_vectors=int(row["support_vectors"]),
                )
            )
    return results


def run_stage(
    stage: str,
    variants: list[Variant],
    train_coordinates: np.ndarray,
    train_geometry: np.ndarray,
    train_labels: np.ndarray,
    validation_coordinates: np.ndarray,
    validation_geometry: np.ndarray,
    validation_labels: np.ndarray,
    args: argparse.Namespace,
    all_results: list[AblationResult],
    result_path: Path,
) -> list[AblationResult]:
    completed = {
        result.variant for result in all_results if result.stage == stage
    }
    pending = [variant for variant in variants if variant.name not in completed]
    if completed:
        print(
            f"[{stage}] resuming with {len(completed)} completed, "
            f"{len(pending)} pending",
            flush=True,
        )

    def run(variant: Variant) -> AblationResult:
        return fit_variant(
            stage,
            variant,
            train_coordinates,
            train_geometry,
            train_labels,
            validation_coordinates,
            validation_geometry,
            validation_labels,
            args.c,
            args.gamma_multiplier,
            args.cache_mb,
            args.seed,
        )

    if args.n_jobs <= 1:
        for variant in pending:
            result = run(variant)
            all_results.append(result)
            save_results(all_results, result_path)
            print_result(result)
    else:
        with ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = {
                executor.submit(run, variant): variant for variant in pending
            }
            for future in as_completed(futures):
                result = future.result()
                all_results.append(result)
                save_results(all_results, result_path)
                print_result(result)
    return [result for result in all_results if result.stage == stage]


def print_result(result: AblationResult) -> None:
    print(
        f"[{result.stage}] {result.variant:<18} "
        f"d={result.feature_count:<3} "
        f"acc={result.accuracy:.4f} macro_f1={result.macro_f1:.4f} "
        f"train={result.training_seconds:.1f}s SV={result.support_vectors}",
        flush=True,
    )


def save_ablation_plot(
    coarse_results: list[AblationResult],
    full_results: list[AblationResult],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered_coarse = sorted(
        coarse_results,
        key=lambda result: result.macro_f1,
        reverse=True,
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(15, 6),
        constrained_layout=True,
        gridspec_kw={"width_ratios": (2.2, 1.0)},
    )
    colors = [
        "#2a9d8f" if result.family in {"full", "drop_one"} else "#457b9d"
        for result in ordered_coarse
    ]
    coarse_names = [result.variant for result in ordered_coarse][::-1]
    coarse_scores = [100.0 * result.macro_f1 for result in ordered_coarse][::-1]
    coarse_bars = axes[0].barh(
        coarse_names,
        coarse_scores,
        color=colors[::-1],
    )
    coarse_min = min(coarse_scores)
    coarse_max = max(coarse_scores)
    coarse_padding = max(0.15, 0.18 * (coarse_max - coarse_min))
    axes[0].set_xlim(coarse_min - coarse_padding, coarse_max + 2.0 * coarse_padding)
    axes[0].bar_label(coarse_bars, fmt="%.2f", padding=3, fontsize=8)
    axes[0].set_xlabel("Validation Macro-F1 (%)")
    axes[0].set_title("10k coarse grouped ablation")
    axes[0].grid(axis="x", alpha=0.25)

    ordered_full = sorted(
        full_results,
        key=lambda result: result.macro_f1,
        reverse=True,
    )
    full_names = [result.variant for result in ordered_full]
    full_scores = [100.0 * result.macro_f1 for result in ordered_full]
    full_bars = axes[1].bar(
        full_names,
        full_scores,
        color="#e76f51",
    )
    full_min = min(full_scores)
    full_max = max(full_scores)
    full_padding = max(0.10, 0.25 * (full_max - full_min))
    axes[1].set_ylim(full_min - full_padding, full_max + 2.0 * full_padding)
    axes[1].bar_label(full_bars, fmt="%.2f", padding=3, fontsize=9)
    axes[1].set_ylabel("Validation Macro-F1 (%)")
    axes[1].set_title("22,967-sample finalist validation")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle(
        "Facial-geometry group ablation (official test set not used)",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def train_final(
    best_variant: Variant,
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    test_paths: np.ndarray,
    metadata: dict,
    args: argparse.Namespace,
    all_results: list[AblationResult],
) -> dict:
    steps = []
    if best_variant.groups:
        steps.append(
            (
                "geometry",
                FunctionTransformer(
                    append_landmark_geometry_groups,
                    kw_args={"groups": best_variant.groups},
                    validate=False,
                ),
            )
        )
    feature_count = 136 + sum(
        len(GEOMETRY_FEATURE_GROUPS[group])
        for group in best_variant.groups
    )
    gamma = float(args.gamma_multiplier / feature_count)
    steps.extend(
        (
            ("scaler", StandardScaler()),
            (
                "svm",
                SVC(
                    C=args.c,
                    kernel="rbf",
                    gamma=gamma,
                    class_weight="balanced",
                    cache_size=args.final_cache_mb,
                    decision_function_shape="ovr",
                    random_state=args.seed,
                ),
            ),
        )
    )
    model = Pipeline(steps)
    print(
        f"Final fit: {best_variant.name}, groups={best_variant.groups}, "
        f"features={feature_count}",
        flush=True,
    )
    started = time.perf_counter()
    model.fit(train_x, train_y)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions = model.predict(test_x)
    batch_ms = (time.perf_counter() - started) * 1000.0 / len(test_x)
    timing_x = test_x[: min(1000, len(test_x))]
    started = time.perf_counter()
    for row in timing_x:
        model.predict(row.reshape(1, -1))
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
    metrics = {
        "accuracy": float(accuracy_score(test_y, predictions)),
        "macro_f1": float(f1_score(test_y, predictions, average="macro")),
        "weighted_f1": float(
            f1_score(test_y, predictions, average="weighted")
        ),
        "training_seconds": float(training_seconds),
        "batch_prediction_ms_per_image": float(batch_ms),
        "single_prediction_ms_per_image": float(single_ms),
        "meets_30ms_requirement": bool(single_ms < 30.0),
        "train_samples": int(len(train_x)),
        "test_samples": int(len(test_x)),
        "input_feature_count": 136,
        "feature_count": int(feature_count),
        "selected_variant": best_variant.name,
        "selected_groups": list(best_variant.groups),
        "selected_geometry_feature_names": [
            name
            for group in best_variant.groups
            for name in GEOMETRY_FEATURE_GROUPS[group]
        ],
        "geometry_feature_count": int(feature_count - 136),
        "classifier": "svm_group_ablation",
        "svm_c": float(args.c),
        "svm_gamma": gamma,
        "svm_gamma_multiplier": float(args.gamma_multiplier),
        "selection_metric": "validation_macro_f1",
        "validation_fraction": float(args.validation_fraction),
        "coarse_samples": int(args.coarse_samples),
        "coarse_variant_count": len(build_variants()),
        "full_validation_candidate_count": int(args.top_k),
        "ablation_results": [asdict(result) for result in all_results],
        "classification_report": report_dict,
        "versions": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "extraction": metadata,
    }
    bundle = {
        "model": model,
        "classes": list(CLASS_NAMES),
        "feature_version": FEATURE_VERSION,
        "feature_mode": "coordinates_plus_selected_geometry",
        "geometry_groups": list(best_variant.groups),
        "geometry_feature_names": metrics["selected_geometry_feature_names"],
        "image_size": metadata["image_size"],
        "center_inset": metadata["center_inset"],
    }
    joblib.dump(bundle, args.output / "expression_classifier.joblib")
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (args.output / "classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )
    save_confusion_matrices(
        test_y, predictions, args.output / "confusion_matrix.png"
    )
    decision_scores = model.decision_function(test_x)
    save_failure_cases(
        test_paths,
        test_y,
        predictions,
        decision_scores,
        args.output / "failure_cases.png",
    )
    print(report_text, flush=True)
    print(
        f"Final test: accuracy={metrics['accuracy']:.4f}, "
        f"macro_f1={metrics['macro_f1']:.4f}, single={single_ms:.2f}ms",
        flush=True,
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--coarse-samples", type=int, default=10_000)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--c", type=float, default=10.0)
    parser.add_argument("--gamma-multiplier", type=float, default=2.0)
    parser.add_argument("--cache-mb", type=float, default=768.0)
    parser.add_argument("--final-cache-mb", type=float, default=2048.0)
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate group definitions and cached features without training.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.features, allow_pickle=False) as cache:
        train_x = cache["train_X"].astype(np.float32)
        train_y = cache["train_y"]
        test_x = cache["test_X"].astype(np.float32)
        test_y = cache["test_y"]
        test_paths = cache["test_paths"]
        metadata = json.loads(str(cache["metadata"]))

    variants = build_variants()
    if args.check:
        all_names = [
            name
            for group in GROUP_ORDER
            for name in GEOMETRY_FEATURE_GROUPS[group]
        ]
        if tuple(all_names) != GEOMETRY_FEATURE_NAMES:
            raise RuntimeError("Geometry groups do not preserve feature order.")
        if len(variants) != 12:
            raise RuntimeError("Expected 12 ablation variants.")
        print("Geometry ablation validation passed")
        print(f"  groups: {GROUP_ORDER}")
        print(f"  group sizes: {[len(GEOMETRY_FEATURE_GROUPS[g]) for g in GROUP_ORDER]}")
        print(f"  variants: {[variant.name for variant in variants]}")
        print(f"  train/test: {len(train_x)}/{len(test_x)}")
        return 0

    indices = np.arange(len(train_x))
    fit_indices, validation_indices = train_test_split(
        indices,
        test_size=args.validation_fraction,
        stratify=train_y,
        random_state=args.seed,
    )
    if args.coarse_samples < len(fit_indices):
        coarse_indices, _ = train_test_split(
            fit_indices,
            train_size=args.coarse_samples,
            stratify=train_y[fit_indices],
            random_state=args.seed,
        )
    else:
        coarse_indices = fit_indices

    print("Computing geometry once for all ablation variants", flush=True)
    train_geometry = landmark_geometry_features(train_x)
    result_path = args.output / "group_ablation_results.csv"
    all_results = load_results(result_path)

    coarse_results = run_stage(
        "coarse",
        variants,
        train_x[coarse_indices],
        train_geometry[coarse_indices],
        train_y[coarse_indices],
        train_x[validation_indices],
        train_geometry[validation_indices],
        train_y[validation_indices],
        args,
        all_results,
        result_path,
    )
    ranked_coarse = sorted(
        coarse_results,
        key=lambda result: (-result.macro_f1, -result.accuracy),
    )
    variant_by_name = {variant.name: variant for variant in variants}
    finalists = [
        variant_by_name[result.variant]
        for result in ranked_coarse[: args.top_k]
    ]
    print(
        "Full-validation finalists: "
        + ", ".join(variant.name for variant in finalists),
        flush=True,
    )
    full_results = run_stage(
        "full_validation",
        finalists,
        train_x[fit_indices],
        train_geometry[fit_indices],
        train_y[fit_indices],
        train_x[validation_indices],
        train_geometry[validation_indices],
        train_y[validation_indices],
        args,
        all_results,
        result_path,
    )
    best_result = sorted(
        full_results,
        key=lambda result: (-result.macro_f1, -result.accuracy),
    )[0]
    best_variant = variant_by_name[best_result.variant]
    save_ablation_plot(
        coarse_results,
        full_results,
        args.output / "group_ablation_validation.png",
    )
    train_final(
        best_variant,
        train_x,
        train_y,
        test_x,
        test_y,
        test_paths,
        metadata,
        args,
        all_results,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
