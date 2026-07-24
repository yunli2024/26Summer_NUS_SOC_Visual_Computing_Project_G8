"""Unified Expert Level entry point.

The default path is strictly facial-keypoint based. The historical ROI-CNN
files remain in ``src`` only as an analysis reference and are never selected by
this entry point.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib


EXPERT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXPERT_DIR.parent
SRC_DIR = EXPERT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fer_dataset import FerZipDataset  # noqa: E402
from keypoint_features import LEGACY_FEATURE_VERSION  # noqa: E402


MODEL_PATH = EXPERT_DIR / "models" / "keypoint" / "current" / "expression_classifier.joblib"
MODEL_METRICS_PATH = EXPERT_DIR / "results" / "zhangyx_part_two" / "geometry_svm_metrics.json"
DATASET_PATH = PROJECT_DIR / "resources" / "expression_data" / "facial_expression_dataset.zip"
FACE_MODELS_DIR = PROJECT_DIR / "resources" / "face_models"


def print_help() -> None:
    print(
        """Expert Level - keypoint-only facial-expression system

Usage:
  python VisualComputingProject/expert_level/main.py inspect
  python VisualComputingProject/expert_level/main.py demo [demo options]
  python VisualComputingProject/expert_level/main.py train [training options]

Examples:
  python VisualComputingProject/expert_level/main.py demo --mirror
  python VisualComputingProject/expert_level/main.py train --cv-folds 5 --workers 4

The train command uses eye-aligned 68-point geometry, PCA candidates, and
Stratified K-fold model selection. Run either command with --help for details.
"""
    )


def inspect() -> int:
    required = {
        "FER dataset": DATASET_PATH,
        "Haar cascade": FACE_MODELS_DIR / "haarcascade_frontalface_default.xml",
        "LBF 68-point model": FACE_MODELS_DIR / "lbfmodel.yaml",
        "YuNet detector": FACE_MODELS_DIR / "face_detection_yunet_2023mar.onnx",
        "current keypoint classifier": MODEL_PATH,
    }
    missing = False
    for label, path in required.items():
        exists = path.is_file()
        print(f"[{'OK' if exists else 'MISSING'}] {label}: {path}")
        missing |= not exists

    if DATASET_PATH.is_file():
        dataset = FerZipDataset(DATASET_PATH)
        dataset.validate()
        inventory = dataset.inventory()
        for split in ("train", "test"):
            counts = inventory[split]
            print(f"{split}: total={sum(counts.values())} classes={counts}")

    if MODEL_PATH.is_file():
        payload = joblib.load(MODEL_PATH)
        print(f"feature_version: {payload.get('feature_version', LEGACY_FEATURE_VERSION)}")
        print(f"selected_model: {payload.get('selected_model', 'zhangyx_geometry_rbf_svm')}")
        if MODEL_METRICS_PATH.is_file():
            metrics = json.loads(MODEL_METRICS_PATH.read_text(encoding="utf-8"))
            print(f"macro_f1: {metrics['macro_f1']:.4f}")
            print(
                "single_prediction_ms: "
                f"{metrics['single_prediction_ms_per_image']:.3f}"
            )
            print(f"feature_count: {metrics['feature_count']}")
    return 2 if missing else 0


def dispatch(command: str, forwarded: list[str]) -> int:
    sys.argv = [sys.argv[0], *forwarded]
    if command == "demo":
        from realtime_keypoint import main as demo_main

        return demo_main()
    if command == "train":
        from train_keypoint import main as train_main

        return train_main()
    raise ValueError(command)


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print_help()
        return 0
    command, forwarded = sys.argv[1], sys.argv[2:]
    if command == "inspect":
        if forwarded:
            print("inspect does not accept extra arguments.", file=sys.stderr)
            return 2
        return inspect()
    if command in {"demo", "train"}:
        return dispatch(command, forwarded)
    print(f"Unknown command: {command}", file=sys.stderr)
    print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
