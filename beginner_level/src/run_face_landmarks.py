"""Beginner Level webcam program.

This file is not run automatically by tests. It opens the camera only when the
user runs the main command.
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
import visualization
from face_detector import ImprovedFaceDetector
from landmark_detector import SmoothedLandmarkDetector
from preprocessing import PREPROCESS_MODES, enhance_frame_for_detection, to_gray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run face landmark detection.")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE preprocessing.")
    parser.add_argument(
        "--preprocess",
        choices=PREPROCESS_MODES,
        default=config.PREPROCESS_MODE,
        help="Realtime preprocessing mode.",
    )
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Camera index.")
    parser.add_argument(
        "--keep-mirror",
        action="store_true",
        help="Keep the camera's mirrored image instead of flipping it back.",
    )
    parser.add_argument("--scale-factor", type=float, default=config.FACE_SCALE_FACTOR)
    parser.add_argument("--min-neighbors", type=int, default=config.FACE_MIN_NEIGHBORS)
    parser.add_argument("--min-size", type=int, default=config.FACE_MIN_SIZE[0])
    parser.add_argument("--max-size", type=int, default=config.FACE_MAX_SIZE[0])
    parser.add_argument(
        "--single-face",
        action="store_true",
        help="Track only the most stable face instead of all detected faces.",
    )
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
    config.SINGLE_FACE_MODE = args.single_face
    config.UNMIRROR_CAMERA = not args.keep_mirror
    config.PREPROCESS_MODE = "clahe" if args.clahe else args.preprocess


def main() -> int:
    args = parse_args()
    apply_runtime_config(args)

    try:
        face_detector = ImprovedFaceDetector()
        landmark_detector = SmoothedLandmarkDetector()
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
            if config.UNMIRROR_CAMERA:
                frame = cv2.flip(frame, 1)
            detection_frame = frame
            display_frame, video_preprocess_name = (
                enhance_frame_for_detection(frame) if config.ENHANCE_VIDEO_FRAME else (frame.copy(), "off")
            )

            use_clahe = config.PREPROCESS_MODE in {"clahe", "clahe-gamma"}
            gray, preprocess_name = to_gray(detection_frame, use_clahe, mode=config.PREPROCESS_MODE)
            detection = face_detector.detect(gray)
            landmark_ok, landmarks, message = landmark_detector.fit(
                gray,
                detection.faces,
                detection.status,
            )
            faces_to_draw = detection.faces
            pose_labels = []
            if detection.detected_now:
                if landmark_ok:
                    faces_to_draw = landmark_detector.last_valid_faces
                    pose_labels = landmark_detector.last_pose_labels
                    face_detector.confirm_faces(faces_to_draw)
                else:
                    face_detector.reject_current_detection()
                    faces_to_draw = []
                    detection.status = "REJECTED"

            visualization.draw_faces(display_frame, faces_to_draw, detection.status, pose_labels)
            if landmark_ok:
                visualization.draw_landmarks(display_frame, landmarks)

            now = time.perf_counter()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now

            visualization.draw_status(
                display_frame,
                fps=fps,
                status=detection.status,
                raw_count=detection.raw_count,
                filtered_count=detection.filtered_count,
                selected_size=_selected_size(faces_to_draw, detection.selected_size),
                clahe_enabled=use_clahe,
                failed_frames=detection.failed_frames,
                message=message,
                preprocess_name=preprocess_name,
                video_preprocess_name=video_preprocess_name,
            )
            cv2.imshow(config.WINDOW_NAME, display_frame)
            if cv2.getWindowProperty(config.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("Window closed by user.")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord(config.QUIT_KEY):
                print("Quit key pressed.")
                break
            if _handle_runtime_key(key, face_detector, landmark_detector):
                print(
                    f"Runtime mode: preprocess={config.PREPROCESS_MODE}, "
                    f"video_enhance={config.ENHANCE_VIDEO_FRAME}, unmirror={config.UNMIRROR_CAMERA}"
                )
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def _selected_size(faces, fallback):
    if faces is None or len(faces) == 0:
        return fallback
    face = faces[0]
    return int(face[2]), int(face[3])


def _handle_runtime_key(key: int, face_detector, landmark_detector) -> bool:
    changed = False
    mode_by_key = {
        ord("1"): "raw",
        ord("2"): "clahe",
        ord("3"): "gamma",
        ord("4"): "clahe-gamma",
    }
    if key in mode_by_key:
        config.PREPROCESS_MODE = mode_by_key[key]
        changed = True
    elif key == ord("v"):
        config.ENHANCE_VIDEO_FRAME = not config.ENHANCE_VIDEO_FRAME
        changed = True
    elif key == ord("m"):
        config.UNMIRROR_CAMERA = not config.UNMIRROR_CAMERA
        changed = True
    elif key == ord("r"):
        changed = True

    if changed:
        face_detector.reject_current_detection()
        landmark_detector.reset()
    return changed


if __name__ == "__main__":
    raise SystemExit(main())
