"""Part II setup check.

This script does not open the webcam and does not run project inference.
It only verifies imports, model loading, and module availability.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import cv2


PART2_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PART2_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config
from face_detector import FaceDetector
from landmark_detector import LandmarkDetector


def check(name: str, passed: bool, detail: str = "") -> bool:
    status = "PASS" if passed else "FAIL"
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")
    return passed


def main() -> int:
    results = []

    results.append(check("cv2 import", True, cv2.__version__))
    results.append(check("cv2.face exists", hasattr(cv2, "face")))
    results.append(check("cv2.imshow exists", hasattr(cv2, "imshow")))

    results.append(check("Haar file exists", config.HAAR_CASCADE_PATH.exists(), str(config.HAAR_CASCADE_PATH)))
    results.append(check("LBF file exists", config.LBF_MODEL_PATH.exists(), str(config.LBF_MODEL_PATH)))

    try:
        FaceDetector()
        results.append(check("Haar model loads", True))
    except Exception as exc:
        results.append(check("Haar model loads", False, str(exc)))

    try:
        LandmarkDetector()
        results.append(check("LBF model loads", True))
    except Exception as exc:
        results.append(check("LBF model loads", False, str(exc)))

    for module_name in [
        "config",
        "face_detector",
        "landmark_detector",
        "visualization",
        "run_face_landmarks",
    ]:
        try:
            importlib.import_module(module_name)
            results.append(check(f"module import: {module_name}", True))
        except Exception as exc:
            results.append(check(f"module import: {module_name}", False, str(exc)))

    if all(results):
        print("Part II setup check: PASS")
        return 0

    print("Part II setup check: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
