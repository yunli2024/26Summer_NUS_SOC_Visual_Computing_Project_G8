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
from landmark_detector import (
    LandmarkSmoother,
    SmoothedLandmarkDetector,
    _pose_with_2d_yaw_fallback,
    _stabilize_open_mouth_lower_lip,
    _validate_landmarks,
)
from preprocessing import enhance_frame_for_detection, to_gray


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
        nested_faces = np.asarray([(100, 80, 240, 240), (160, 220, 100, 100)], dtype=np.int32)
        filtered_nested = detector._reject_nested_mouth_faces(nested_faces)
        results.append(check("Nested mouth false face rejected", len(filtered_nested) == 1))
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
        yaw_points = loose_points.copy()
        yaw_points[30] = (25, 50)
        pose = _pose_with_2d_yaw_fallback(yaw_points, np.asarray((10, 10, 80, 80)), (0.0, 160.0, 0.0))
        results.append(check("2D yaw fallback corrects underestimated yaw", abs(pose[0]) > 20 and abs(pose[1]) < 1))
        mouth_points = loose_points.copy()
        mouth_points[[50, 51, 52, 61, 62, 63], 1] = 45
        mouth_points[[55, 56, 57, 58, 59, 65, 66, 67], 1] = 48
        mouth_points[57, 1] = 60
        mouth_points[66, 1] = 60
        adjusted = _stabilize_open_mouth_lower_lip(mouth_points, np.asarray((10, 10, 80, 80)))
        results.append(check("Open-mouth lower lip constrained", adjusted[66, 1] >= 49.4))
        jaw_points = loose_points.copy()
        jaw_points[36:42] = (35, 35)
        jaw_points[42:48] = (65, 35)
        jaw_points[[27, 30, 33, 51, 57]] = (50, 55)
        jaw_points[16] = (120, 70)
        constrained = LandmarkSmoother()._constrain_jawline(jaw_points, np.asarray((10, 10, 80, 100)))
        results.append(check("Jawline side drift constrained", constrained[16, 0] < 95))
    except Exception as exc:
        results.append(check("Smoothed LBF detector loads", False, str(exc)))

    dummy_bgr = np.zeros((20, 20, 3), dtype=np.uint8)
    results.append(check("Video enhancement defaults off", not config.ENHANCE_VIDEO_FRAME))
    config.ENHANCE_VIDEO_FRAME = True
    enhanced, enhance_mode = enhance_frame_for_detection(dummy_bgr)
    results.append(check("Optional display video enhancement", enhanced.shape == dummy_bgr.shape and "lowlight" in enhance_mode))
    config.ENHANCE_VIDEO_FRAME = False
    gray, mode = to_gray(dummy_bgr, use_clahe=True, mode="clahe")
    results.append(check("CLAHE preprocessing", gray.shape == (20, 20) and mode == "clahe"))
    gray, mode = to_gray(dummy_bgr, mode="clahe-gamma")
    results.append(check("CLAHE gamma preprocessing", gray.shape == (20, 20) and mode == "clahe-gamma"))

    run_module = importlib.import_module("run_face_landmarks")
    detector = ImprovedFaceDetector()
    landmark_detector = SmoothedLandmarkDetector()
    changed = run_module._handle_runtime_key(ord("3"), detector, landmark_detector)
    results.append(check("Runtime preprocess key switch", changed and config.PREPROCESS_MODE == "gamma"))
    changed = run_module._handle_runtime_key(ord("v"), detector, landmark_detector)
    results.append(check("Runtime video enhancement toggle", changed))

    if all(results):
        print("Beginner setup check: PASS")
        return 0
    print("Beginner setup check: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
