"""Improved Beginner Level webcam program.

This file is not run automatically by tests. It opens the camera only when the
user runs the improved main command.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import config
from face_detector import ImprovedFaceDetector
from landmark_detector import SmoothedLandmarkDetector
from preprocessing import to_gray
import visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run improved face landmark detection.")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE preprocessing.")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Camera index.")
    parser.add_argument("--scale-factor", type=float, default=config.FACE_SCALE_FACTOR)
    parser.add_argument("--min-neighbors", type=int, default=config.FACE_MIN_NEIGHBORS)
    parser.add_argument("--min-size", type=int, default=config.FACE_MIN_SIZE[0])
    parser.add_argument("--max-size", type=int, default=config.FACE_MAX_SIZE[0])
    return parser.parse_args()


def open_camera(camera_index: int):
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def apply_runtime_config(args: argparse.Namespace) -> None:
    config.FACE_SCALE_FACTOR = args.scale_factor
    config.FACE_MIN_NEIGHBORS = args.min_neighbors
    config.FACE_MIN_SIZE = (args.min_size, args.min_size)
    config.FACE_MAX_SIZE = (args.max_size, args.max_size)


def main() -> int:
    args = parse_args()
    apply_runtime_config(args)
    use_clahe = args.clahe or config.USE_CLAHE

    try:
        face_detector = ImprovedFaceDetector()
        landmark_detector = SmoothedLandmarkDetector()
        cap = open_camera(args.camera)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Setup error: {exc}")
        return 1

    print("Improved camera started. Press q to quit.")
    last_time = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed. Exiting.")
                break

            gray, _mode = to_gray(frame, use_clahe)
            detection = face_detector.detect(gray)
            landmark_ok, landmarks, message = landmark_detector.fit(
                gray,
                detection.faces,
                detection.status,
            )

            visualization.draw_faces(frame, detection.faces, detection.status)
            if landmark_ok:
                visualization.draw_landmarks(frame, landmarks)

            now = time.perf_counter()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now

            visualization.draw_status(
                frame,
                fps=fps,
                status=detection.status,
                raw_count=detection.raw_count,
                filtered_count=detection.filtered_count,
                selected_size=detection.selected_size,
                clahe_enabled=use_clahe,
                failed_frames=detection.failed_frames,
                message=message,
            )
            cv2.imshow(config.WINDOW_NAME, frame)
            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user.")
                break
            if cv2.waitKey(1) & 0xFF == ord(config.QUIT_KEY):
                print("Quit key pressed.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
