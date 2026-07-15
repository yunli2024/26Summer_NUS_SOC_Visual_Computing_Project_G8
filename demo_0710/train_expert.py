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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from expression_features import EMOTION_CLASSES, ExpertFeatureExtractor, prepare_fer_image
from face_pipeline import HaarFaceDetector, LbfLandmarkEstimator
from fer_dataset import FerZipDataset, ImageRecord


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ZIP = PROJECT_DIR.parent / "expert" / "facial_expression_dataset.zip"
DEFAULT_CASCADE = PROJECT_DIR / "haarcascade_frontalface_default.xml"
DEFAULT_LBF_MODEL = PROJECT_DIR / "lbfmodel.yaml"
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "expression_classifier.joblib"
DEFAULT_REPORT_DIR = PROJECT_DIR / "reports"
DEFAULT_CACHE_DIR = PROJECT_DIR / "features"
CACHE_VERSION = 1
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
    return parser.parse_args()


def label_to_index(label: str) -> int:
    return EMOTION_CLASSES.index(label)


def build_extractor(args: argparse.Namespace) -> ExpertFeatureExtractor:
    detector = HaarFaceDetector(
        cascade_path=args.cascade,
        min_neighbors=args.min_neighbors,
        min_face_size=args.min_face_size,
        preprocess="clahe",
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
        }
    except Exception as exc:
        return {
            "ok": False,
            "split": split,
            "label": record.label,
            "path": record.path,
            "reason": str(exc)[:160],
        }
    landmark_estimator = LbfLandmarkEstimator(args.lbf_model)
    return ExpertFeatureExtractor(
        detector,
        landmark_estimator,
        max_faces=1,
        use_center_fallback=not args.no_center_fallback,
    )


def extract_split(
    dataset: FerZipDataset,
    records: Sequence[ImageRecord],
    extractor: ExpertFeatureExtractor,
    *,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]], Dict[str, int]]:
    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []
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
    return np.vstack(x_rows).astype(np.float32), np.asarray(y_rows, dtype=np.int64), failures, stats


def extract_split_parallel(
    args: argparse.Namespace,
    records: Sequence[ImageRecord],
    *,
    split: str,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, str]], Dict[str, int]]:
    x_rows: List[np.ndarray] = []
    y_rows: List[int] = []
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
    return np.vstack(x_rows).astype(np.float32), np.asarray(y_rows, dtype=np.int64), failures, stats


def cache_path(args: argparse.Namespace, split: str, max_per_class: int) -> Path:
    fallback = "fallback" if not args.no_center_fallback else "nofallback"
    name = f"{split}_m{max_per_class}_seed{args.seed}_{fallback}_v{CACHE_VERSION}.npz"
    return args.cache_dir / name


def load_feature_cache(path: Path):
    data = np.load(path, allow_pickle=False)
    x_rows = data["x"].astype(np.float32)
    y_rows = data["y"].astype(np.int64)
    failures = json.loads(str(data["failures_json"]))
    stats = json.loads(str(data["stats_json"]))
    return x_rows, y_rows, failures, stats


def save_feature_cache(path: Path, x_rows: np.ndarray, y_rows: np.ndarray, failures, stats) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=x_rows.astype(np.float32),
        y=y_rows.astype(np.int64),
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

    x_train, y_train, train_failures, train_stats = load_or_extract_split(
        args,
        dataset,
        train_records,
        split="train",
        max_per_class=args.max_train_per_class,
    )
    x_test, y_test, test_failures, test_stats = load_or_extract_split(
        args,
        dataset,
        test_records,
        split="test",
        max_per_class=args.max_test_per_class,
    )

    classifier = make_pipeline(
        StandardScaler(),
        SVC(C=3.0, gamma=0.01, class_weight="balanced", random_state=args.seed),
    )
    classifier.fit(x_train, y_train)

    t0 = time.perf_counter()
    y_pred = classifier.predict(x_test)
    prediction_ms = (time.perf_counter() - t0) * 1000.0 / max(len(x_test), 1)

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
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "avg_prediction_ms": prediction_ms,
        "classification_report": report,
        "model_path": str(args.model_out),
    }
    (args.report_dir / "expert_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_confusion_csv(args.report_dir / "confusion_matrix.csv", matrix)
    write_failures(args.report_dir / "failure_cases.csv", [*train_failures, *test_failures])

    joblib.dump(
        {
            "model": classifier,
            "classes": list(EMOTION_CLASSES),
            "feature_dim": int(x_train.shape[1]),
            "train_stats": train_stats,
            "test_stats": test_stats,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "avg_prediction_ms": prediction_ms,
        },
        args.model_out,
    )

    print(f"accuracy={accuracy:.4f} | macro_f1={macro_f1:.4f} | avg_prediction={prediction_ms:.4f} ms")
    print(f"model saved to: {args.model_out}")
    print(f"reports saved to: {args.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
