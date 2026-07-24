"""Bonus Level entry point.

Run this file to open the pose detection and dance scoring GUI.
"""

import argparse

try:
    from .src.app import run_app
    from .src import config
except ImportError:
    from src.app import run_app
    from src import config


def check_setup() -> int:
    required = {
        "YOLO pose model": config.MODEL_PATH,
        "reference video": config.DEFAULT_VIDEO_PATH,
    }
    missing = False
    for label, path in required.items():
        exists = path.is_file()
        print(f"[{'OK' if exists else 'MISSING'}] {label}: {path}")
        missing |= not exists
    print("Spatial normalization: torso-centered, body-scale normalized")
    print(f"Temporal alignment window: {config.ALIGNMENT_WINDOW_SECONDS:.2f} s")
    print(f"Pose weights: {config.POSE_SCORE_WEIGHTS}")
    print(f"Final pose/motion weights: {config.FINAL_SCORE_WEIGHTS}")
    print(f"Motion window: {config.MOTION_WINDOW_SECONDS:.2f} s")
    return 2 if missing else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Just Dance pose-keypoint application.")
    parser.add_argument("--check", action="store_true", help="Validate resources without opening the GUI.")
    args = parser.parse_args()
    if args.check:
        return check_setup()
    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
