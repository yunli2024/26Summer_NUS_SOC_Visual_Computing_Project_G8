"""Fail fast when the local visual-computing runtime is incomplete or conflicted."""

from __future__ import annotations

import importlib.metadata as metadata
import sys

import cv2
import ultralytics


def main() -> int:
    installed = {
        distribution.metadata["Name"].lower()
        for distribution in metadata.distributions()
        if distribution.metadata["Name"]
    }
    conflicts = sorted(
        name
        for name in ("opencv-python", "opencv-python-headless")
        if name in installed
    )
    if conflicts:
        joined = ", ".join(conflicts)
        raise SystemExit(
            f"Conflicting OpenCV wheel detected: {joined}.\n"
            "Remove the conflicting wheel, then reinstall opencv-contrib-python."
        )
    if not hasattr(cv2, "face"):
        raise SystemExit(
            "cv2.face is unavailable. Install opencv-contrib-python, not opencv-python."
        )
    print(f"Runtime check passed: Python {sys.version.split()[0]}")
    print(f"  OpenCV contrib: {cv2.__version__}")
    print(f"  Ultralytics: {ultralytics.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
