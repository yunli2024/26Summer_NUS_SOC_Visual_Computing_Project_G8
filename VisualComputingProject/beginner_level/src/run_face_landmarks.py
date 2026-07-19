"""Main webcam program for Part II face landmark detection.

Run manually when ready:
    python part2_beginner/src/run_face_landmarks.py
    python part2_beginner/src/run_face_landmarks.py --clahe
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
from face_detector import FaceDetector
from landmark_detector import LandmarkDetector
import visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time face landmark detection.")
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="Enable CLAHE preprocessing for the grayscale frame.",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=config.CAMERA_INDEX,
        help="Camera index used by cv2.VideoCapture.",
    )
    return parser.parse_args()


def preprocess_frame(frame, use_clahe: bool):
    """Convert BGR frame to grayscale and optionally apply CLAHE."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not use_clahe:
        return gray, "gray"

    clahe = cv2.createCLAHE(
        clipLimit=config.CLAHE_CLIP_LIMIT,
        tileGridSize=config.CLAHE_TILE_GRID_SIZE,
    )
    return clahe.apply(gray), "CLAHE"


def open_camera(camera_index: int):
    """Open the selected camera and report a clear error if it fails."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open camera index {camera_index}. "
            "Check camera permission, camera index, or whether another app is using it."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def main() -> int:
    args = parse_args()
    use_clahe = args.clahe or config.USE_CLAHE

    try:
        face_detector = FaceDetector()
        landmark_detector = LandmarkDetector()
        cap = open_camera(args.camera)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Setup error: {exc}")
        return 1

    print("Camera started. Press q to quit.")
    last_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed. Exiting.")
                break

            gray, preprocessing_name = preprocess_frame(frame, use_clahe)
            faces = face_detector.detect(gray)
            landmark_ok, landmarks, message = landmark_detector.fit(gray, faces)

            visualization.draw_faces(frame, faces)
            if landmark_ok:
                visualization.draw_landmarks(frame, landmarks)

            now = time.perf_counter()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now

            if len(faces) == 0:
                message = "No face detected"

            visualization.draw_fps(frame, fps)
            visualization.draw_status(frame, message, preprocessing_name)
            cv2.imshow(config.WINDOW_NAME, frame)

            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user.")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord(config.QUIT_KEY):
                print("Quit key pressed.")
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
