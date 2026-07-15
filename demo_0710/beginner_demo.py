from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from face_pipeline import (
    DetectionResult,
    FpsMeter,
    HaarFaceDetector,
    LbfLandmarkEstimator,
    MediaPipeFaceDetector,
    draw_detections,
    draw_status,
    limit_boxes_by_area,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CASCADE = PROJECT_DIR / "haarcascade_frontalface_default.xml"
DEFAULT_LBF_MODEL = PROJECT_DIR / "lbfmodel.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Beginner demo: real-time face detection and 68-point LBF facial landmarks."
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index used by cv2.VideoCapture.")
    parser.add_argument("--width", type=int, default=960, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=540, help="Requested capture height.")
    parser.add_argument("--detector", choices=("haar", "mediapipe"), default="haar", help="Face detector backend.")
    parser.add_argument("--preprocess", choices=("none", "equalize", "clahe"), default="clahe", help="Haar grayscale preprocessing.")
    parser.add_argument("--scale-factor", type=float, default=1.1, help="Haar pyramid scale factor.")
    parser.add_argument("--min-neighbors", type=int, default=6, help="Haar detection confidence/strictness.")
    parser.add_argument("--min-face-size", type=int, default=60, help="Minimum face size in pixels.")
    parser.add_argument("--overlap-threshold", type=float, default=0.55, help="Suppress boxes that overlap this much over the smaller box.")
    parser.add_argument("--max-faces", type=int, default=4, help="Limit LBF fitting to the largest N faces. Use 0 for all.")
    parser.add_argument("--mirror", action="store_true", help="Mirror webcam frames for a selfie-style display.")
    parser.add_argument("--cascade", type=Path, default=DEFAULT_CASCADE, help="Path to Haar cascade XML.")
    parser.add_argument("--lbf-model", type=Path, default=DEFAULT_LBF_MODEL, help="Path to OpenCV LBF model YAML.")
    parser.add_argument("--window-name", default="Beginner Face Keypoints", help="OpenCV display window name.")
    return parser.parse_args()


def build_detector(args: argparse.Namespace):
    if args.detector == "haar":
        return HaarFaceDetector(
            cascade_path=args.cascade,
            scale_factor=args.scale_factor,
            min_neighbors=args.min_neighbors,
            min_face_size=args.min_face_size,
            preprocess=args.preprocess,
            overlap_threshold=args.overlap_threshold,
        )
    return MediaPipeFaceDetector()


def configure_camera(cap: cv2.VideoCapture, width: int, height: int) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def save_snapshot(frame, project_dir: Path) -> Path:
    snapshot_dir = project_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    output_path = snapshot_dir / f"beginner_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(output_path), frame)
    return output_path


def main() -> int:
    args = parse_args()

    try:
        detector = build_detector(args)
        landmark_estimator = LbfLandmarkEstimator(args.lbf_model)
    except Exception as exc:
        print(f"[startup error] {exc}", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(args.camera_index)
    configure_camera(cap, args.width, args.height)
    if not cap.isOpened():
        print(f"[camera error] Cannot open webcam index {args.camera_index}.", file=sys.stderr)
        return 3

    fps_meter = FpsMeter()
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window_name, args.width, args.height)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[camera warning] Failed to read a frame; exiting.", file=sys.stderr)
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            start = time.perf_counter()
            boxes = detector.detect(frame)
            boxes = limit_boxes_by_area(boxes, args.max_faces)
            landmarks = landmark_estimator.fit(frame, boxes)
            detections = [DetectionResult(box=box, landmarks=points) for box, points in zip(boxes, landmarks)]
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            fps = fps_meter.tick()
            draw_detections(frame, detections)
            draw_status(
                frame,
                fps=fps,
                detector_name=args.detector,
                preprocess=args.preprocess if args.detector == "haar" else "n/a",
                face_count=len(detections),
                elapsed_ms=elapsed_ms,
            )
            cv2.imshow(args.window_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                output_path = save_snapshot(frame, PROJECT_DIR)
                print(f"[snapshot] saved to {output_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
