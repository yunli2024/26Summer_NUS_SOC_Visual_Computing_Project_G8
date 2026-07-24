"""Bonus Task 1: analyze dance videos with robust primary-dancer pose tracking."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
YOLO_CONFIG_PATH = Path(tempfile.gettempdir()) / "visual-computing-yolo"
YOLO_CONFIG_PATH.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_PATH))
DEFAULT_MODEL = PROJECT_DIR / "resources" / "pose_models" / "yolov8n-pose.pt"
DEFAULT_VIDEO = PROJECT_DIR / "resources" / "videos" / "dance_example_1.mp4"
DEFAULT_OUTPUT = BASE_DIR / "task1_results"

# COCO 17-point skeleton. Nose is connected to both shoulders for a compact torso.
SKELETON = (
    (0, 5), (0, 6), (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
)


def portable_project_path(path: Path) -> str:
    """Prefer a repository-relative path in reports generated inside the project."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_DIR.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2] - first[0])) * max(0.0, float(first[3] - first[1]))
    second_area = max(0.0, float(second[2] - second[0])) * max(0.0, float(second[3] - second[1]))
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class PoseDetection:
    box: np.ndarray
    box_confidence: float
    points: np.ndarray
    keypoint_confidence: np.ndarray


class PrimaryDancerTracker:
    """Select the central/large person and preserve identity between frames."""

    def __init__(self, keypoint_threshold: float = 0.35, smoothing_alpha: float = 0.6) -> None:
        self.keypoint_threshold = keypoint_threshold
        self.smoothing_alpha = smoothing_alpha
        self.previous_box: np.ndarray | None = None
        self.previous_points: np.ndarray | None = None
        self.previous_valid: np.ndarray | None = None
        self.missed_frames = 0

    def reset(self) -> None:
        self.previous_box = None
        self.previous_points = None
        self.previous_valid = None
        self.missed_frames = 0

    def select(
        self,
        detections: list[PoseDetection],
        frame_shape: tuple[int, ...],
    ) -> tuple[PoseDetection | None, np.ndarray | None, np.ndarray | None, list[float]]:
        if not detections:
            self.missed_frames += 1
            if self.missed_frames > 8:
                self.reset()
            return None, None, None, []

        frame_height, frame_width = frame_shape[:2]
        frame_center = np.array([frame_width / 2.0, frame_height / 2.0])
        frame_diagonal = math.hypot(frame_width, frame_height)
        frame_area = max(float(frame_width * frame_height), 1.0)
        scores: list[float] = []

        for detection in detections:
            x1, y1, x2, y2 = detection.box
            area_score = min(((x2 - x1) * (y2 - y1)) / (frame_area * 0.35), 1.0)
            center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
            center_score = max(0.0, 1.0 - np.linalg.norm(center - frame_center) / (frame_diagonal * 0.55))

            if self.previous_box is None:
                score = 0.60 * area_score + 0.30 * center_score + 0.10 * detection.box_confidence
            else:
                previous_center = np.array(
                    [
                        (self.previous_box[0] + self.previous_box[2]) / 2.0,
                        (self.previous_box[1] + self.previous_box[3]) / 2.0,
                    ]
                )
                continuity_distance = np.linalg.norm(center - previous_center) / frame_diagonal
                continuity = max(box_iou(detection.box, self.previous_box), math.exp(-12.0 * continuity_distance))
                score = (
                    0.30 * area_score
                    + 0.15 * center_score
                    + 0.45 * continuity
                    + 0.10 * detection.box_confidence
                )
            scores.append(float(score))

        chosen = detections[int(np.argmax(scores))]
        current_valid = chosen.keypoint_confidence >= self.keypoint_threshold
        smoothed = chosen.points.astype(np.float32).copy()

        if self.previous_points is not None and self.previous_valid is not None:
            reusable = current_valid & self.previous_valid
            smoothed[reusable] = (
                self.smoothing_alpha * smoothed[reusable]
                + (1.0 - self.smoothing_alpha) * self.previous_points[reusable]
            )

        self.previous_box = chosen.box.copy()
        self.previous_points = smoothed.copy()
        self.previous_valid = current_valid.copy()
        self.missed_frames = 0
        return chosen, smoothed, current_valid, scores


def extract_detections(result) -> list[PoseDetection]:
    if result.boxes is None or result.keypoints is None:
        return []
    boxes = result.boxes.xyxy.cpu().numpy()
    box_confidences = result.boxes.conf.cpu().numpy()
    points = result.keypoints.xy.cpu().numpy()
    if result.keypoints.conf is None:
        confidences = np.ones(points.shape[:2], dtype=np.float32)
    else:
        confidences = result.keypoints.conf.cpu().numpy()

    count = min(len(boxes), len(points))
    return [
        PoseDetection(
            box=boxes[index].astype(np.float32),
            box_confidence=float(box_confidences[index]),
            points=points[index].astype(np.float32),
            keypoint_confidence=confidences[index].astype(np.float32),
        )
        for index in range(count)
    ]


def draw_pose(
    frame: np.ndarray,
    points: np.ndarray,
    valid: np.ndarray,
    box: np.ndarray,
    pose_confidence: float,
) -> None:
    for first, second in SKELETON:
        if valid[first] and valid[second]:
            p1 = tuple(np.rint(points[first]).astype(int))
            p2 = tuple(np.rint(points[second]).astype(int))
            cv2.line(frame, p1, p2, (40, 235, 90), 3, cv2.LINE_AA)
    for index, point in enumerate(points):
        if valid[index]:
            cv2.circle(frame, tuple(np.rint(point).astype(int)), 5, (40, 80, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(np.rint(point).astype(int)), 5, (255, 255, 255), 1, cv2.LINE_AA)

    x1, y1, x2, y2 = np.rint(box).astype(int)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 3, cv2.LINE_AA)
    label = f"PRIMARY DANCER  pose={pose_confidence:.2f}"
    cv2.rectangle(frame, (x1, max(0, y1 - 28)), (min(frame.shape[1] - 1, x1 + 260), y1), (0, 220, 255), -1)
    cv2.putText(frame, label, (x1 + 5, max(18, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA)


def draw_other_people(frame: np.ndarray, detections: list[PoseDetection], chosen: PoseDetection | None) -> None:
    for detection in detections:
        if detection is chosen:
            continue
        x1, y1, x2, y2 = np.rint(detection.box).astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (125, 125, 125), 1, cv2.LINE_AA)
        cv2.putText(frame, "other", (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (170, 170, 170), 1, cv2.LINE_AA)


def make_contact_sheet(frames: list[np.ndarray], output_path: Path) -> None:
    if not frames:
        return
    tile_width, tile_height = 480, 270
    columns = 3
    rows = math.ceil(len(frames) / columns)
    canvas = np.full((rows * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)
    for index, frame in enumerate(frames):
        resized = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        canvas[row * tile_height : (row + 1) * tile_height, column * tile_width : (column + 1) * tile_width] = resized
    success, encoded = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not success:
        raise RuntimeError("Could not encode contact sheet.")
    encoded.tofile(output_path)


def analyze_video(model, input_path: Path, output_root: Path, args: argparse.Namespace) -> dict:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    result_name = (
        input_path.stem
        if (
            input_path.parent.resolve() == BASE_DIR.resolve()
            or input_path.resolve() == DEFAULT_VIDEO.resolve()
        )
        else f"{input_path.parent.name}_{input_path.stem}"
    )
    result_dir = output_root / result_name
    result_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    writer = None
    annotated_path = result_dir / "annotated.mp4"
    if not args.no_video:
        writer = cv2.VideoWriter(
            str(annotated_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(source_fps / args.stride, 1.0),
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Could not create output video: {annotated_path}")

    tracker = PrimaryDancerTracker(args.keypoint_confidence, args.smoothing_alpha)
    processed = 0
    decoded = args.start_frame
    found_frames = 0
    multi_person_frames = 0
    visible_keypoints: list[int] = []
    pose_confidences: list[float] = []
    inference_times: list[float] = []
    contact_frames: list[np.ndarray] = []
    cached_points: list[np.ndarray] = []
    cached_valid: list[np.ndarray] = []
    cached_boxes: list[np.ndarray] = []
    cached_source_frames: list[int] = []
    started = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            relative_index = decoded - args.start_frame
            decoded += 1
            if relative_index % args.stride != 0:
                continue
            if args.max_frames > 0 and processed >= args.max_frames:
                break

            inference_started = time.perf_counter()
            result = model.predict(
                frame,
                conf=args.person_confidence,
                imgsz=args.image_size,
                device=args.device,
                verbose=False,
            )[0]
            inference_times.append((time.perf_counter() - inference_started) * 1000.0)
            detections = extract_detections(result)
            if len(detections) > 1:
                multi_person_frames += 1

            chosen, points, valid, _ = tracker.select(detections, frame.shape)
            draw_other_people(frame, detections, chosen)
            if chosen is not None and points is not None and valid is not None:
                found_frames += 1
                visible = int(valid.sum())
                visible_keypoints.append(visible)
                if visible:
                    pose_confidence = float(chosen.keypoint_confidence[valid].mean())
                else:
                    pose_confidence = 0.0
                pose_confidences.append(pose_confidence)
                draw_pose(frame, points, valid, chosen.box, pose_confidence)
                cached_points.append(points.astype(np.float32))
                cached_valid.append(valid.astype(bool))
                cached_boxes.append(chosen.box.astype(np.float32))
            else:
                cached_points.append(np.full((17, 2), np.nan, dtype=np.float32))
                cached_valid.append(np.zeros(17, dtype=bool))
                cached_boxes.append(np.full(4, np.nan, dtype=np.float32))
            cached_source_frames.append(decoded - 1)

            cv2.rectangle(frame, (0, 0), (width, 38), (20, 20, 20), -1)
            cv2.putText(
                frame,
                f"frame {processed + 1} | people {len(detections)} | visible joints {visible_keypoints[-1] if visible_keypoints else 0}/17",
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(frame)
            if len(contact_frames) < 6 and (processed == 0 or processed % args.contact_every == 0):
                contact_frames.append(frame.copy())
            if args.show:
                cv2.imshow("Bonus Task 1 - Pose Analysis", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            processed += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    statistics = {
        "input": portable_project_path(input_path),
        "source": {
            "width": width,
            "height": height,
            "fps": source_fps,
            "reported_frames": source_frames,
        },
        "settings": {
            "model": portable_project_path(args.model),
            "image_size": args.image_size,
            "person_confidence": args.person_confidence,
            "keypoint_confidence": args.keypoint_confidence,
            "smoothing_alpha": args.smoothing_alpha,
            "stride": args.stride,
            "start_frame": args.start_frame,
            "max_frames": args.max_frames,
            "device": args.device,
        },
        "results": {
            "processed_frames": processed,
            "primary_dancer_frames": found_frames,
            "primary_dancer_rate": found_frames / processed if processed else 0.0,
            "multi_person_frames": multi_person_frames,
            "multi_person_rate": multi_person_frames / processed if processed else 0.0,
            "average_visible_keypoints": float(np.mean(visible_keypoints)) if visible_keypoints else 0.0,
            "average_pose_confidence": float(np.mean(pose_confidences)) if pose_confidences else 0.0,
            "average_inference_ms": float(np.mean(inference_times)) if inference_times else 0.0,
            "processing_fps": processed / elapsed if elapsed > 0 else 0.0,
            "elapsed_seconds": elapsed,
        },
        "environment": {
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
    }
    (result_dir / "analysis.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")
    np.savez_compressed(
        result_dir / "pose_cache.npz",
        points=np.stack(cached_points).astype(np.float32),
        valid=np.stack(cached_valid).astype(bool),
        boxes=np.stack(cached_boxes).astype(np.float32),
        source_frames=np.asarray(cached_source_frames, dtype=np.int32),
        source_fps=np.asarray(source_fps, dtype=np.float32),
        playback_fps=np.asarray(max(source_fps / args.stride, 1.0), dtype=np.float32),
        input_path=np.asarray(str(input_path.resolve())),
    )
    make_contact_sheet(contact_frames, result_dir / "contact_sheet.jpg")
    print(json.dumps(statistics["results"], indent=2), flush=True)
    print(f"Saved results: {result_dir}", flush=True)
    return statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path, default=[DEFAULT_VIDEO])
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu", help="cpu, 0, 1, ...")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--person-confidence", type=float, default=0.30)
    parser.add_argument("--keypoint-confidence", type=float, default=0.35)
    parser.add_argument("--smoothing-alpha", type=float, default=0.60)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 processes the entire video")
    parser.add_argument("--contact-every", type=int, default=60)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if not args.model.is_file():
            raise FileNotFoundError(f"Pose model not found: {args.model}")
        if args.stride < 1:
            raise ValueError("stride must be at least 1.")
        from ultralytics import YOLO

        model = YOLO(str(args.model))
        args.output.mkdir(parents=True, exist_ok=True)
        summaries = [analyze_video(model, path, args.output, args) for path in args.inputs]
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    except (FileNotFoundError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
