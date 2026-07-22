from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2

from pose_pipeline import PoseEstimator, PoseStreamTracker, draw_pose_overlay


APP_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO = APP_DIR / "dance_example_1.mp4"
DEFAULT_MODEL = APP_DIR / "yolov8n-pose.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless Bonus Task 1 reference video pose analysis.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Reference video path.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLOv8 pose model path.")
    parser.add_argument("--frames", type=int, default=120, help="Maximum frames to process. Use 0 for the full video.")
    parser.add_argument("--start-frame", type=int, default=0, help="Frame index to start from.")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--warmup-frames", type=int, default=3, help="Processed frames to exclude from average timing.")
    parser.add_argument("--conf", type=float, default=0.30, help="YOLO detection confidence.")
    parser.add_argument("--keypoint-conf", type=float, default=0.25, help="Keypoint drawing/visibility confidence.")
    parser.add_argument("--main-only", action="store_true", help="Draw only the selected main dancer.")
    parser.add_argument("--save-samples", type=int, default=6, help="Number of annotated sample frames to save.")
    parser.add_argument("--output-dir", type=Path, default=APP_DIR / "outputs", help="Output directory.")
    parser.add_argument("--run-name", default="", help="Subdirectory name under output-dir. Defaults to start/frame settings.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    run_name = args.run_name or f"start{args.start_frame}_frames{args.frames}_conf{args.conf:.2f}".replace(".", "p")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    max_frames = total_frames if args.frames <= 0 else min(args.frames, total_frames or args.frames)
    sample_window = max(1, max_frames - max(0, args.warmup_frames))
    sample_interval = max(1, sample_window // max(1, args.save_samples))
    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    estimator = PoseEstimator(args.model)
    tracker = PoseStreamTracker()
    rows = []
    processed = 0
    frames_with_person = 0
    frames_with_multi_person = 0
    total_infer_ms = 0.0
    total_visible = 0
    measured_frames = 0
    measured_infer_ms = 0.0
    measured_draw_ms = 0.0
    measured_visible = 0
    saved_samples = 0
    sample_files: list[str] = []
    started = time.perf_counter()

    frame_index = max(0, args.start_frame)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % args.stride != 0:
            frame_index += 1
            continue
        if processed >= max_frames:
            break

        result = estimator.infer(
            frame,
            frame_index=frame_index,
            conf=args.conf,
            keypoint_conf=args.keypoint_conf,
            main_only=False,
        )
        result = tracker.update(result)
        annotated, drawn = draw_pose_overlay(
            frame,
            result,
            keypoint_conf=args.keypoint_conf,
            show_video=True,
            main_only=args.main_only,
        )

        main = drawn.main_detection
        main_visible = main.visible_count if main is not None else 0
        is_warmup = processed < args.warmup_frames
        if drawn.person_count > 0:
            frames_with_person += 1
        if drawn.person_count > 1:
            frames_with_multi_person += 1
        total_visible += main_visible
        total_infer_ms += drawn.inference_ms
        if not is_warmup:
            measured_frames += 1
            measured_infer_ms += drawn.inference_ms
            measured_draw_ms += drawn.draw_ms
            measured_visible += main_visible

        rows.append(
            {
                "frame": frame_index,
                "warmup": is_warmup,
                "persons": drawn.person_count,
                "main_visible": main_visible,
                "inference_ms": round(drawn.inference_ms, 3),
                "draw_ms": round(drawn.draw_ms, 3),
            }
        )

        sample_offset = max(0, processed - args.warmup_frames)
        should_save_sample = (
            args.save_samples > 0
            and not is_warmup
            and sample_offset % sample_interval == 0
            and saved_samples < args.save_samples
        )
        if should_save_sample:
            sample_name = f"sample_frame{frame_index:05d}.jpg"
            cv2.imwrite(str(run_dir / sample_name), annotated)
            sample_files.append(sample_name)
            saved_samples += 1

        processed += 1
        frame_index += 1

        if processed == 1 or processed % 30 == 0 or processed == max_frames:
            elapsed = time.perf_counter() - started
            print(f"processed {processed}/{max_frames} frames | {processed / max(elapsed, 1e-6):.1f} fps", flush=True)

    cap.release()
    elapsed = time.perf_counter() - started
    summary = {
        "video": str(args.video),
        "model": str(args.model),
        "video_frames": total_frames,
        "video_fps": fps,
        "video_size": [width, height],
        "processed_frames": processed,
        "measured_frames": measured_frames,
        "warmup_frames": min(args.warmup_frames, processed),
        "start_frame": args.start_frame,
        "stride": args.stride,
        "frames_with_person": frames_with_person,
        "frames_with_multi_person": frames_with_multi_person,
        "person_detection_rate": frames_with_person / max(processed, 1),
        "multi_person_rate": frames_with_multi_person / max(processed, 1),
        "avg_main_visible_keypoints": total_visible / max(processed, 1),
        "avg_inference_ms": total_infer_ms / max(processed, 1),
        "measured_avg_main_visible_keypoints": measured_visible / max(measured_frames, 1),
        "measured_avg_inference_ms": measured_infer_ms / max(measured_frames, 1),
        "measured_avg_draw_ms": measured_draw_ms / max(measured_frames, 1),
        "wall_fps": processed / max(elapsed, 1e-6),
        "sample_files": sample_files,
    }

    with (run_dir / "task1_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "warmup", "persons", "main_visible", "inference_ms", "draw_ms"])
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "task1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
