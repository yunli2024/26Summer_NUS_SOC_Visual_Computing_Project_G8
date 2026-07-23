"""Part Two Task 1: extract FER landmarks, train a classifier, and evaluate it."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from expression_features import CLASS_NAMES, FEATURE_VERSION, ExpressionFeatureExtractor


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BASE_DIR / "facial_expression_dataset" / "facial_expression_dataset"
DEFAULT_OUTPUT = BASE_DIR / "artifacts"


def collect_samples(
    dataset_root: Path,
    split: str,
    max_per_class: int,
    seed: int,
) -> list[tuple[Path, str]]:
    rng = np.random.default_rng(seed)
    samples: list[tuple[Path, str]] = []
    for class_name in CLASS_NAMES:
        class_dir = dataset_root / split / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Dataset class directory not found: {class_dir}")
        paths = sorted(
            path for path in class_dir.iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if max_per_class > 0 and len(paths) > max_per_class:
            selected = rng.choice(len(paths), size=max_per_class, replace=False)
            paths = [paths[index] for index in sorted(selected)]
        samples.extend((path.resolve(), class_name) for path in paths)
    return samples


def extract_split(
    extractor: ExpressionFeatureExtractor,
    samples: list[tuple[Path, str]],
    face_mode: str,
    progress_every: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, list[str]]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    paths: list[str] = []
    failures: list[str] = []
    methods: Counter[str] = Counter()
    started = time.perf_counter()

    for index, (image_path, label) in enumerate(samples, start=1):
        try:
            feature, _, method = extractor.extract_path(image_path, face_mode)
            features.append(feature)
            labels.append(label)
            paths.append(str(image_path))
            methods[method] += 1
        except (ValueError, cv2.error) as error:
            failures.append(f"{image_path}\t{type(error).__name__}: {error}")

        if index % progress_every == 0 or index == len(samples):
            elapsed = time.perf_counter() - started
            rate = index / max(elapsed, 1e-6)
            print(
                f"  {index:>6}/{len(samples)} | success={len(features):>6} "
                f"failed={len(failures):>4} | {rate:.1f} images/s",
                flush=True,
            )

    if not features:
        raise RuntimeError("No landmark features were extracted.")

    stats = {
        "requested": len(samples),
        "successful": len(features),
        "failed": len(failures),
        "success_rate": len(features) / len(samples),
        "methods": dict(methods),
        "seconds": time.perf_counter() - started,
    }
    return (
        np.stack(features).astype(np.float32),
        np.asarray(labels),
        np.asarray(paths),
        stats,
        failures,
    )


def extract_features(args: argparse.Namespace) -> Path:
    dataset_root = args.dataset.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    args.output.mkdir(parents=True, exist_ok=True)

    extractor = ExpressionFeatureExtractor(args.image_size, args.center_inset)
    arrays: dict[str, np.ndarray] = {}
    metadata = {
        "feature_version": FEATURE_VERSION,
        "dataset": str(dataset_root),
        "face_mode": args.face_mode,
        "image_size": args.image_size,
        "center_inset": args.center_inset,
        "classes": list(CLASS_NAMES),
        "splits": {},
    }
    all_failures: list[str] = []

    for split, limit in (
        ("train", args.max_train_per_class),
        ("test", args.max_test_per_class),
    ):
        samples = collect_samples(dataset_root, split, limit, args.seed)
        print(f"Extracting {split}: {len(samples)} images", flush=True)
        X, y, paths, stats, failures = extract_split(
            extractor, samples, args.face_mode, args.progress_every
        )
        arrays[f"{split}_X"] = X
        arrays[f"{split}_y"] = y
        arrays[f"{split}_paths"] = paths
        metadata["splits"][split] = stats
        all_failures.extend(f"{split}\t{failure}" for failure in failures)

    feature_path = args.output / "fer_landmark_features.npz"
    np.savez_compressed(feature_path, metadata=json.dumps(metadata), **arrays)
    (args.output / "extraction_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    (args.output / "extraction_failures.tsv").write_text(
        "split\tpath\terror\n" + "\n".join(all_failures), encoding="utf-8"
    )
    print(f"Saved features: {feature_path}", flush=True)
    return feature_path


def save_confusion_matrices(y_true, y_pred, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for axis, normalize, title in (
        (axes[0], None, "Confusion matrix (counts)"),
        (axes[1], "true", "Confusion matrix (row-normalized)"),
    ):
        matrix = confusion_matrix(y_true, y_pred, labels=CLASS_NAMES, normalize=normalize)
        display = ConfusionMatrixDisplay(matrix, display_labels=CLASS_NAMES)
        display.plot(ax=axis, cmap="Blues", colorbar=False, values_format=".2f" if normalize else "d")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=35)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def save_failure_cases(
    paths: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    decision_scores: np.ndarray,
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wrong = np.flatnonzero(y_true != y_pred)
    if len(wrong) == 0:
        return
    sorted_scores = np.sort(decision_scores, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]
    selected = wrong[np.argsort(margins[wrong])[::-1][:20]]

    figure, axes = plt.subplots(4, 5, figsize=(12, 10), constrained_layout=True)
    for axis in axes.flat:
        axis.axis("off")
    for axis, index in zip(axes.flat, selected):
        encoded = np.fromfile(Path(paths[index]), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        axis.imshow(image, cmap="gray", vmin=0, vmax=255)
        axis.set_title(
            f"true: {y_true[index]}\npred: {y_pred[index]}",
            fontsize=9,
            color="crimson",
        )
    figure.suptitle("High-confidence failure cases", fontsize=15)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def train_and_evaluate(args: argparse.Namespace, feature_path: Path) -> None:
    import joblib
    import sklearn
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(feature_path, allow_pickle=False) as data:
        train_X = data["train_X"]
        train_y = data["train_y"]
        test_X = data["test_X"]
        test_y = data["test_y"]
        test_paths = data["test_paths"]
        metadata = json.loads(str(data["metadata"]))

    print(
        f"Training {args.classifier} on {len(train_X)} samples "
        f"with {train_X.shape[1]} features",
        flush=True,
    )
    if args.classifier == "svm":
        classifier = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        C=args.svm_c,
                        kernel="rbf",
                        gamma="scale",
                        class_weight="balanced",
                        cache_size=args.svm_cache_mb,
                        decision_function_shape="ovr",
                        random_state=args.seed,
                    ),
                ),
            ]
        )
    else:
        classifier = HistGradientBoostingClassifier(
            learning_rate=args.hgb_learning_rate,
            max_iter=args.hgb_iterations,
            max_leaf_nodes=args.hgb_max_leaf_nodes,
            l2_regularization=1.0,
            class_weight="balanced",
            early_stopping=len(train_X) >= 100,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=args.seed,
        )
    train_started = time.perf_counter()
    classifier.fit(train_X, train_y)
    training_seconds = time.perf_counter() - train_started

    batch_started = time.perf_counter()
    predictions = classifier.predict(test_X)
    batch_ms = (time.perf_counter() - batch_started) * 1000.0 / len(test_X)

    timing_X = test_X[: min(1000, len(test_X))]
    single_started = time.perf_counter()
    for row in timing_X:
        classifier.predict(row.reshape(1, -1))
    single_ms = (time.perf_counter() - single_started) * 1000.0 / len(timing_X)

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
        "weighted_f1": float(f1_score(test_y, predictions, average="weighted")),
        "training_seconds": training_seconds,
        "batch_prediction_ms_per_image": batch_ms,
        "single_prediction_ms_per_image": single_ms,
        "meets_30ms_requirement": single_ms < 30.0,
        "train_samples": int(len(train_X)),
        "test_samples": int(len(test_X)),
        "feature_count": int(train_X.shape[1]),
        "classifier": args.classifier,
        "svm_c": args.svm_c,
        "versions": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "classification_report": report_dict,
        "extraction": metadata,
    }

    model_bundle = {
        "model": classifier,
        "classes": list(CLASS_NAMES),
        "feature_version": FEATURE_VERSION,
        "image_size": metadata["image_size"],
        "center_inset": metadata["center_inset"],
    }
    joblib.dump(model_bundle, args.output / "expression_classifier.joblib")
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output / "classification_report.txt").write_text(report_text, encoding="utf-8")
    save_confusion_matrices(test_y, predictions, args.output / "confusion_matrix.png")
    decision_scores = (
        classifier.decision_function(test_X)
        if hasattr(classifier, "decision_function")
        else classifier.predict_proba(test_X)
    )
    save_failure_cases(
        test_paths,
        test_y,
        predictions,
        decision_scores,
        args.output / "failure_cases.png",
    )

    print(report_text)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Single-image prediction: {single_ms:.3f} ms")
    print(f"Saved evaluation artifacts to: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "extract", "train"), default="all")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--features", type=Path, default=None, help="NPZ cache for train stage")
    parser.add_argument("--face-mode", choices=("center", "haar", "haar-fallback"), default="center")
    parser.add_argument("--image-size", type=int, default=192)
    parser.add_argument("--center-inset", type=float, default=0.08)
    parser.add_argument("--max-train-per-class", type=int, default=0, help="0 uses every image")
    parser.add_argument("--max-test-per-class", type=int, default=0, help="0 uses every image")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--classifier", choices=("hgb", "svm"), default="hgb")
    parser.add_argument("--svm-c", type=float, default=10.0)
    parser.add_argument("--svm-cache-mb", type=float, default=2048.0)
    parser.add_argument("--hgb-iterations", type=int, default=250)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.08)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        feature_path = args.features or (args.output / "fer_landmark_features.npz")
        if args.stage in {"all", "extract"}:
            feature_path = extract_features(args)
        if args.stage in {"all", "train"}:
            if not feature_path.is_file():
                raise FileNotFoundError(f"Feature cache not found: {feature_path}")
            train_and_evaluate(args, feature_path)
    except (FileNotFoundError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
