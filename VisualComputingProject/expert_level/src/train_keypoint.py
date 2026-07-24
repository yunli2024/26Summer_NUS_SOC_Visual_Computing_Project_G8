from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from keypoint_features import (
    EMOTION_CLASSES,
    UNIFIED_FEATURE_VERSION,
    ExpertFeatureExtractor,
    prepare_fer_image,
)
from face_pipeline import HaarFaceDetector, LbfLandmarkEstimator
from fer_dataset import FerZipDataset, ImageRecord


SRC_DIR = Path(__file__).resolve().parent
EXPERT_DIR = SRC_DIR.parent
PROJECT_DIR = EXPERT_DIR.parent
RESOURCES_DIR = PROJECT_DIR / "resources"
DEFAULT_DATA_ZIP = RESOURCES_DIR / "expression_data" / "facial_expression_dataset.zip"
DEFAULT_CASCADE = RESOURCES_DIR / "face_models" / "haarcascade_frontalface_default.xml"
DEFAULT_LBF_MODEL = RESOURCES_DIR / "face_models" / "lbfmodel.yaml"
DEFAULT_MODEL_PATH = EXPERT_DIR / "models" / "keypoint" / "current" / "expression_classifier.joblib"
DEFAULT_REPORT_DIR = EXPERT_DIR / "results" / "keypoint_cv"
DEFAULT_CACHE_DIR = EXPERT_DIR / "data" / "cache"
CACHE_VERSION = 2
_WORKER_DATASET: Optional[FerZipDataset] = None
_WORKER_EXTRACTOR: Optional[ExpertFeatureExtractor] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Expert Level expression classifier from facial landmarks.")
    parser.add_argument("--data-zip", type=Path, default=DEFAULT_DATA_ZIP, help="FER-style dataset zip path.")
    parser.add_argument("--cascade", type=Path, default=DEFAULT_CASCADE, help="Haar cascade XML path.")
    parser.add_argument("--lbf-model", type=Path, default=DEFAULT_LBF_MODEL, help="OpenCV LBF landmark model path.")
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH, help="Output classifier path.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR, help="Directory for metrics and CSV reports.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Directory for cached landmark features.")
    parser.add_argument("--rebuild-cache", action="store_true", help="Ignore cached features and extract again.")
    parser.add_argument("--max-train-per-class", type=int, default=0, help="Use 0 for all training images.")
    parser.add_argument("--max-test-per-class", type=int, default=0, help="Use 0 for all test images.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--min-neighbors", type=int, default=5, help="Haar strictness for FER images.")
    parser.add_argument("--min-face-size", type=int, default=24, help="Minimum face size for upscaled FER images.")
    parser.add_argument("--no-center-fallback", action="store_true", help="Disable centered-box fallback for cropped FER faces.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for landmark extraction. Use 1 for serial.")
    parser.add_argument("--cv-folds", type=int, default=5, help="Stratified K-fold count on the training split only.")
    parser.add_argument(
        "--models",
        default="linear_svm_pca,logreg_pca,hgb",
        help="Comma-separated candidates: linear_svm_pca, logreg_pca, hgb.",
    )
    parser.add_argument(
        "--pca-variance",
        type=float,
        default=0.98,
        help="Variance retained by PCA candidates; must be in (0, 1).",
    )
    return parser.parse_args()


def label_to_index(label: str) -> int:
    return EMOTION_CLASSES.index(label)


def build_extractor(args: argparse.Namespace) -> ExpertFeatureExtractor:
    return build_extractor_from_values(
        args.cascade,
        args.lbf_model,
        min_neighbors=args.min_neighbors,
        min_face_size=args.min_face_size,
        use_center_fallback=not args.no_center_fallback,
    )


def build_extractor_from_values(
    cascade_path: Path,
    lbf_model_path: Path,
    *,
    min_neighbors: int,
    min_face_size: int,
    use_center_fallback: bool,
) -> ExpertFeatureExtractor:
    detector = HaarFaceDetector(
        cascade_path=cascade_path,
        min_neighbors=min_neighbors,
        min_face_size=min_face_size,
        preprocess="clahe",
    )
    landmark_estimator = LbfLandmarkEstimator(lbf_model_path)
    return ExpertFeatureExtractor(
        detector,
        landmark_estimator,
        max_faces=1,
        use_center_fallback=use_center_fallback,
        feature_version=UNIFIED_FEATURE_VERSION,
    )


def init_worker(
    data_zip: str,
    cascade: str,
    lbf_model: str,
    min_neighbors: int,
    min_face_size: int,
    use_center_fallback: bool,
) -> None:
    global _WORKER_DATASET, _WORKER_EXTRACTOR
    _WORKER_DATASET = FerZipDataset(Path(data_zip))
    _WORKER_EXTRACTOR = build_extractor_from_values(
        Path(cascade),
        Path(lbf_model),
        min_neighbors=min_neighbors,
        min_face_size=min_face_size,
        use_center_fallback=use_center_fallback,
    )


def extract_one_worker(payload):
    split, record = payload
    if _WORKER_DATASET is None or _WORKER_EXTRACTOR is None:
        raise RuntimeError("Worker was not initialized.")
    try:
        image = _WORKER_DATASET.read_image(record)
        image = prepare_fer_image(image)
        features = _WORKER_EXTRACTOR.extract_primary(image)
        if features is None:
            return {
                "ok": False,
                "split": split,
                "label": record.label,
                "path": record.path,
                "reason": "no_landmarks",
            }
        return {
            "ok": True,
            "label_index": label_to_index(record.label),
            "vector": features.vector,
            "source": features.source,
            "path": record.path,
        }
    except Exception as exc:
        return {
            "ok": False,
            "split": split,
            "label": record.label,
            "path": record.path,
            "reason": str(exc)[:160],
        }


def extract_split(
    dataset: FerZipDataset,
    records: Sequence[ImageRecord],
    extractor: ExpertFeatureExtractor,
    *,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, str]], Dict[str, int]]:
    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    paths: List[str] = []
    failures: List[Dict[str, str]] = []
    stats = {"total": 0, "ok": 0, "failed": 0, "center_fallback": 0, "haar": 0}
    start = time.perf_counter()

    for idx, record in enumerate(records, 1):
        stats["total"] += 1
        try:
            image = dataset.read_image(record)
            image = prepare_fer_image(image)
            features = extractor.extract_primary(image)
            if features is None:
                stats["failed"] += 1
                failures.append({"split": split, "label": record.label, "path": record.path, "reason": "no_landmarks"})
                continue

            x_rows.append(features.vector)
            y_rows.append(label_to_index(record.label))
            paths.append(record.path)
            stats["ok"] += 1
            stats[features.source] = stats.get(features.source, 0) + 1
        except Exception as exc:
            stats["failed"] += 1
            failures.append({"split": split, "label": record.label, "path": record.path, "reason": str(exc)[:160]})

        if idx == 1 or idx % 250 == 0 or idx == len(records):
            elapsed = time.perf_counter() - start
            rate = idx / max(elapsed, 1e-9)
            print(f"[{split}] {idx}/{len(records)} processed | ok={stats['ok']} failed={stats['failed']} | {rate:.1f} img/s")

    if not x_rows:
        raise RuntimeError(f"No usable landmark features extracted for split={split}.")
    return (
        np.vstack(x_rows).astype(np.float32),
        np.asarray(y_rows, dtype=np.int64),
        np.asarray(paths),
        failures,
        stats,
    )


def extract_split_parallel(
    args: argparse.Namespace,
    records: Sequence[ImageRecord],
    *,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, str]], Dict[str, int]]:
    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []
    paths: List[str] = []
    failures: List[Dict[str, str]] = []
    stats = {"total": 0, "ok": 0, "failed": 0, "center_fallback": 0, "haar": 0}
    start = time.perf_counter()
    workers = max(1, int(args.workers))
    tasks = [(split, record) for record in records]

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=init_worker,
        initargs=(
            str(args.data_zip),
            str(args.cascade),
            str(args.lbf_model),
            args.min_neighbors,
            args.min_face_size,
            not args.no_center_fallback,
        ),
    ) as executor:
        for idx, result in enumerate(executor.map(extract_one_worker, tasks, chunksize=8), 1):
            stats["total"] += 1
            if result["ok"]:
                x_rows.append(result["vector"])
                y_rows.append(int(result["label_index"]))
                paths.append(str(result["path"]))
                stats["ok"] += 1
                source = str(result["source"])
                stats[source] = stats.get(source, 0) + 1
            else:
                stats["failed"] += 1
                failures.append(
                    {
                        "split": str(result["split"]),
                        "label": str(result["label"]),
                        "path": str(result["path"]),
                        "reason": str(result["reason"]),
                    }
                )

            if idx == 1 or idx % 250 == 0 or idx == len(records):
                elapsed = time.perf_counter() - start
                rate = idx / max(elapsed, 1e-9)
                print(
                    f"[{split}] {idx}/{len(records)} processed with {workers} workers | "
                    f"ok={stats['ok']} failed={stats['failed']} | {rate:.1f} img/s",
                    flush=True,
                )

    if not x_rows:
        raise RuntimeError(f"No usable landmark features extracted for split={split}.")
    return (
        np.vstack(x_rows).astype(np.float32),
        np.asarray(y_rows, dtype=np.int64),
        np.asarray(paths),
        failures,
        stats,
    )


def cache_path(args: argparse.Namespace, split: str, max_per_class: int) -> Path:
    fallback = "fallback" if not args.no_center_fallback else "nofallback"
    name = f"{split}_m{max_per_class}_seed{args.seed}_{fallback}_v{CACHE_VERSION}.npz"
    return args.cache_dir / name


def load_feature_cache(path: Path):
    data = np.load(path, allow_pickle=False)
    x_rows = data["x"].astype(np.float32)
    y_rows = data["y"].astype(np.int64)
    paths = data["paths"].astype(str)
    failures = json.loads(str(data["failures_json"]))
    stats = json.loads(str(data["stats_json"]))
    return x_rows, y_rows, paths, failures, stats


def save_feature_cache(
    path: Path,
    x_rows: np.ndarray,
    y_rows: np.ndarray,
    paths: np.ndarray,
    failures,
    stats,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=x_rows.astype(np.float32),
        y=y_rows.astype(np.int64),
        paths=np.asarray(paths),
        failures_json=json.dumps(failures),
        stats_json=json.dumps(stats),
    )


def load_or_extract_split(
    args: argparse.Namespace,
    dataset: FerZipDataset,
    records: Sequence[ImageRecord],
    *,
    split: str,
    max_per_class: int,
):
    path = cache_path(args, split, max_per_class)
    if path.exists() and not args.rebuild_cache:
        print(f"[{split}] loading cached features from {path}")
        return load_feature_cache(path)
    if args.workers > 1:
        result = extract_split_parallel(args, records, split=split)
    else:
        extractor = build_extractor(args)
        result = extract_split(dataset, records, extractor, split=split)
    save_feature_cache(path, *result)
    print(f"[{split}] cached features to {path}")
    return result


def write_confusion_csv(path: Path, matrix: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual\\predicted", *EMOTION_CLASSES])
        for label, row in zip(EMOTION_CLASSES, matrix):
            writer.writerow([label, *[int(value) for value in row]])


def write_failures(path: Path, failures: Sequence[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["split", "label", "path", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failures)


def build_candidate_models(args: argparse.Namespace) -> Dict[str, Pipeline]:
    if not 0.0 < args.pca_variance < 1.0:
        raise ValueError("--pca-variance must be in (0, 1).")
    requested = [name.strip() for name in args.models.split(",") if name.strip()]
    supported = {"linear_svm_pca", "logreg_pca", "hgb"}
    unknown = sorted(set(requested) - supported)
    if unknown:
        raise ValueError(f"Unknown model candidates: {', '.join(unknown)}")
    if not requested:
        raise ValueError("At least one model candidate is required.")

    def pca_steps():
        return [
            ("scaler", StandardScaler()),
            (
                "pca",
                PCA(
                    n_components=args.pca_variance,
                    whiten=True,
                    svd_solver="full",
                    random_state=args.seed,
                ),
            ),
        ]

    models: Dict[str, Pipeline] = {}
    for name in requested:
        if name == "linear_svm_pca":
            models[name] = Pipeline(
                [
                    *pca_steps(),
                    (
                        "classifier",
                        LinearSVC(
                            C=1.0,
                            class_weight="balanced",
                            dual="auto",
                            random_state=args.seed,
                        ),
                    ),
                ]
            )
        elif name == "logreg_pca":
            models[name] = Pipeline(
                [
                    *pca_steps(),
                    (
                        "classifier",
                        LogisticRegression(
                            C=1.0,
                            class_weight="balanced",
                            max_iter=2000,
                            random_state=args.seed,
                        ),
                    ),
                ]
            )
        elif name == "hgb":
            models[name] = Pipeline(
                [
                    (
                        "classifier",
                        HistGradientBoostingClassifier(
                            learning_rate=0.08,
                            max_iter=250,
                            max_leaf_nodes=31,
                            l2_regularization=1.0,
                            class_weight="balanced",
                            # Outer StratifiedKFold already estimates
                            # generalization; disabling the internal split also
                            # keeps small smoke tests and rare classes valid.
                            early_stopping=False,
                            random_state=args.seed,
                        ),
                    )
                ]
            )
    return models


def benchmark_single_prediction(model: Pipeline, rows: np.ndarray, limit: int = 250) -> float:
    sample = rows[: min(limit, len(rows))]
    if len(sample) == 0:
        return float("inf")
    started = time.perf_counter()
    for row in sample:
        model.predict(row.reshape(1, -1))
    return (time.perf_counter() - started) * 1000.0 / len(sample)


def cross_validate_candidates(
    args: argparse.Namespace,
    models: Dict[str, Pipeline],
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[str, Pipeline, List[Dict[str, float | int | str | bool]]]:
    folds = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    scoring = {"macro_f1": "f1_macro", "balanced_accuracy": "balanced_accuracy"}
    summaries: List[Dict[str, float | int | str | bool]] = []
    fitted_models: Dict[str, Pipeline] = {}

    for name, model in models.items():
        print(f"[cv] {name}: {args.cv_folds}-fold StratifiedKFold", flush=True)
        result = cross_validate(
            model,
            x_train,
            y_train,
            cv=folds,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        fitted = clone(model)
        fit_started = time.perf_counter()
        fitted.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - fit_started
        latency_ms = benchmark_single_prediction(fitted, x_train)
        pca = fitted.named_steps.get("pca")
        macro_mean = float(np.mean(result["test_macro_f1"]))
        macro_std = float(np.std(result["test_macro_f1"]))
        balanced_mean = float(np.mean(result["test_balanced_accuracy"]))
        # Primary objective is robust class-wise behavior, not raw accuracy.
        selection_score = macro_mean - 0.25 * macro_std + 0.02 * balanced_mean
        summary: Dict[str, float | int | str | bool] = {
            "model": name,
            "cv_folds": int(args.cv_folds),
            "macro_f1_mean": macro_mean,
            "macro_f1_std": macro_std,
            "balanced_accuracy_mean": balanced_mean,
            "selection_score": selection_score,
            "fit_seconds_full_train": fit_seconds,
            "single_prediction_ms": latency_ms,
            "meets_30ms_requirement": latency_ms < 30.0,
            "pca_components": int(pca.n_components_) if pca is not None else int(x_train.shape[1]),
            "input_features": int(x_train.shape[1]),
        }
        summaries.append(summary)
        fitted_models[name] = fitted
        print(
            f"[cv] {name}: macro_f1={macro_mean:.4f}+/-{macro_std:.4f}, "
            f"single={latency_ms:.3f} ms, pca_components={summary['pca_components']}",
            flush=True,
        )

    realtime_candidates = [item for item in summaries if bool(item["meets_30ms_requirement"])]
    selection_pool = realtime_candidates or summaries
    selected = max(selection_pool, key=lambda item: float(item["selection_score"]))
    selected_name = str(selected["model"])
    return selected_name, fitted_models[selected_name], summaries


def write_cv_summary(path: Path, summaries: Sequence[Dict[str, object]]) -> None:
    if not summaries:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)


def write_prediction_failures(
    path: Path,
    model: Pipeline,
    x_test: np.ndarray,
    paths: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(x_test), dtype=np.float32)
    elif hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(x_test), dtype=np.float32)
    else:
        scores = np.zeros((len(x_test), len(EMOTION_CLASSES)), dtype=np.float32)
        scores[np.arange(len(x_test)), y_pred] = 1.0
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    ordered = np.sort(scores, axis=1)
    margins = ordered[:, -1] - ordered[:, -2]
    wrong = np.flatnonzero(y_true != y_pred)
    wrong = wrong[np.argsort(margins[wrong])[::-1]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["feature_row", "path", "actual", "predicted", "decision_margin"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index in wrong:
            writer.writerow(
                {
                    "feature_row": int(index),
                    "path": str(paths[index]),
                    "actual": EMOTION_CLASSES[int(y_true[index])],
                    "predicted": EMOTION_CLASSES[int(y_pred[index])],
                    "decision_margin": float(margins[index]),
                }
            )

def main() -> int:
    args = parse_args()
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = FerZipDataset(args.data_zip)
    dataset.validate()
    dataset.write_inventory(args.report_dir / "dataset_inventory.json")

    train_records = dataset.sample_records("train", max_per_class=args.max_train_per_class, seed=args.seed)
    test_records = dataset.sample_records("test", max_per_class=args.max_test_per_class, seed=args.seed)
    print(f"train records: {len(train_records)} | test records: {len(test_records)}")

    x_train, y_train, train_paths, train_failures, train_stats = load_or_extract_split(
        args,
        dataset,
        train_records,
        split="train",
        max_per_class=args.max_train_per_class,
    )
    x_test, y_test, test_paths, test_failures, test_stats = load_or_extract_split(
        args,
        dataset,
        test_records,
        split="test",
        max_per_class=args.max_test_per_class,
    )

    del train_paths
    candidates = build_candidate_models(args)
    selected_name, classifier, cv_summaries = cross_validate_candidates(
        args,
        candidates,
        x_train,
        y_train,
    )
    write_cv_summary(args.report_dir / "cross_validation.csv", cv_summaries)
    (args.report_dir / "cross_validation.json").write_text(
        json.dumps(cv_summaries, indent=2),
        encoding="utf-8",
    )

    t0 = time.perf_counter()
    y_pred = classifier.predict(x_test)
    batch_prediction_ms = (time.perf_counter() - t0) * 1000.0 / max(len(x_test), 1)
    single_prediction_ms = benchmark_single_prediction(classifier, x_test, limit=1000)

    accuracy = float(accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    matrix = confusion_matrix(y_test, y_pred, labels=list(range(len(EMOTION_CLASSES))))
    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(EMOTION_CLASSES))),
        target_names=list(EMOTION_CLASSES),
        output_dict=True,
        zero_division=0,
    )

    payload = {
        "classes": list(EMOTION_CLASSES),
        "train_records_selected": len(train_records),
        "test_records_selected": len(test_records),
        "train_feature_stats": train_stats,
        "test_feature_stats": test_stats,
        "feature_version": UNIFIED_FEATURE_VERSION,
        "input_features": int(x_train.shape[1]),
        "selection_metric": "macro_f1_mean - 0.25 * macro_f1_std + 0.02 * balanced_accuracy_mean",
        "selected_model": selected_name,
        "cross_validation": cv_summaries,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "batch_prediction_ms_per_image": batch_prediction_ms,
        "single_prediction_ms_per_image": single_prediction_ms,
        "meets_30ms_requirement": single_prediction_ms < 30.0,
        "classification_report": report,
        "model_path": str(args.model_out),
    }
    (args.report_dir / "expert_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_confusion_csv(args.report_dir / "confusion_matrix.csv", matrix)
    write_failures(args.report_dir / "extraction_failures.csv", [*train_failures, *test_failures])
    write_prediction_failures(
        args.report_dir / "misclassification_cases.csv",
        classifier,
        x_test,
        test_paths,
        y_test,
        y_pred,
    )

    joblib.dump(
        {
            "model": classifier,
            "classes": list(EMOTION_CLASSES),
            "feature_version": UNIFIED_FEATURE_VERSION,
            "feature_dim": int(x_train.shape[1]),
            "selected_model": selected_name,
            "cv_folds": int(args.cv_folds),
            "cv_summaries": cv_summaries,
            "train_stats": train_stats,
            "test_stats": test_stats,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "batch_prediction_ms_per_image": batch_prediction_ms,
            "single_prediction_ms_per_image": single_prediction_ms,
        },
        args.model_out,
    )

    print(
        f"selected={selected_name} | macro_f1={macro_f1:.4f} | "
        f"accuracy(reference only)={accuracy:.4f} | single_prediction={single_prediction_ms:.4f} ms"
    )
    print(f"model saved to: {args.model_out}")
    print(f"reports saved to: {args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
