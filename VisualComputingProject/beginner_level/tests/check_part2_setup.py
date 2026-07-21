"""Safe checks for the Beginner Level.

This script does not open the camera.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import cv2
import numpy as np


BEGINNER_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BEGINNER_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config
from face_detector import ImprovedFaceDetector
from landmark_detector import SmoothedLandmarkDetector, _validate_landmarks
from preprocessing import to_gray


def check(name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")
    return passed


def main() -> int:
    results = []
    for module_name in [
        "config",
        "face_detector",
        "landmark_detector",
        "preprocessing",
        "head_pose",
        "visualization",
        "run_face_landmarks",
    ]:
        try:
            importlib.import_module(module_name)
            results.append(check(f"module import: {module_name}", True))
        except Exception as exc:
            results.append(check(f"module import: {module_name}", False, str(exc)))

    results.append(check("cv2 import", True, cv2.__version__))
    results.append(check("cv2.face exists", hasattr(cv2, "face")))
    results.append(check("Haar file exists", config.HAAR_CASCADE_PATH.exists(), str(config.HAAR_CASCADE_PATH)))
    results.append(check("LBF file exists", config.LBF_MODEL_PATH.exists(), str(config.LBF_MODEL_PATH)))

    try:
        detector = ImprovedFaceDetector()
        results.append(check("Improved Haar detector loads", True))
        dummy = np.zeros((120, 120), dtype=np.uint8)
        result = detector.detect(dummy)
        results.append(check("Padded detection returns result object", hasattr(result, "failed_frames")))
        results.append(check("Detection status exists", result.status in {"DETECTED", "CACHED", "LOST"}))
        small_faces = np.asarray([(10, 10, 20, 20), (10, 10, 90, 30)], dtype=np.int32)
        filtered = detector._filter_faces(small_faces, (480, 640))
        results.append(check("Small/aspect false faces filtered", len(filtered) == 0))
        detector.last_faces = np.asarray([(100, 100, 180, 180)], dtype=np.int32)
        inner = np.asarray([(145, 145, 45, 45)], dtype=np.int32)
        selected = detector._select_face(inner)
        results.append(check("Inner shrunken false positive rejected", selected is None))
        fallback = detector._fallback(raw_count=0, filtered_count=0)
        results.append(check("Fallback status is cached", fallback.status == "CACHED"))
        detector.reject_current_detection()
        results.append(check("Rejected detection clears cache", len(detector.last_faces) == 0))
    except Exception as exc:
        results.append(check("Improved Haar detector loads", False, str(exc)))

    try:
        landmark_detector = SmoothedLandmarkDetector()
        results.append(check("Smoothed LBF detector loads", True))
        ok, landmarks, message = landmark_detector.fit(np.zeros((120, 120), dtype=np.uint8), np.asarray([(10, 10, 80, 80)]), "CACHED")
        results.append(check("Cached frame skips LBF update", not ok and landmarks == [] and "Cached" in message))
        fake_points = np.zeros((68, 2), dtype=np.float32)
        valid, _reason, _pose = _validate_landmarks(fake_points, np.asarray((10, 10, 80, 80)), (120, 120))
        results.append(check("Collapsed landmark geometry rejected", not valid))
        loose_points = np.column_stack(
            [
                np.linspace(10, 90, 68, dtype=np.float32),
                np.linspace(20, 85, 68, dtype=np.float32),
            ]
        )
        valid, _reason, _pose = _validate_landmarks(loose_points, np.asarray((10, 10, 80, 80)), (120, 120))
        results.append(check("Loose landmark geometry accepted", valid))
    except Exception as exc:
        results.append(check("Smoothed LBF detector loads", False, str(exc)))

    dummy_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
    gray, mode = to_gray(dummy_bgr, use_clahe=True, mode="clahe")
    results.append(check("CLAHE preprocessing", gray.shape == (20, 20) and mode == "clahe"))
    gray, mode = to_gray(dummy_bgr, mode="clahe-gamma")
    results.append(check("CLAHE gamma preprocessing", gray.shape == (20, 20) and mode == "clahe-gamma"))

    if all(results):
        print("Beginner setup check: PASS")
        return 0
    print("Beginner setup check: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
