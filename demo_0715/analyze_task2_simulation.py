from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2

from bonus_task2_app import draw_match_overlay, make_combined_snapshot
from pose_pipeline import PoseEstimator, PoseStreamTracker, draw_pose_overlay
from pose_scoring import TemporalPoseMatcher


APP_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO = APP_DIR / "dance_example_1.mp4"
DEFAULT_MODEL = APP_DIR / "yolov8n-pose.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline Bonus Task 2 scoring simulation.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Reference video path.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLOv8 pose model path.")
    parser.add_argument("--ref-start", type=int, default=900, help="Reference start frame.")
    parser.add_argument("--simulated-user-lag", type=int, default=8, help="User motion lag in video frames.")
    parser.add_argument("--frames", type=int, default=90, help="Frames to process.")
    parser.add_argument("--lag-window", type=int, default=15, help="Temporal matching window in frames.")
    parser.add_argument("--conf", type=float, default=0.20, help="YOLO detection confidence.")
    parser.add_argument("--keypoint-conf", type=float, default=0.20, help="Keypoint visibility threshold.")
    parser.add_argument("--save-samples", type=int, default=4, help="Combined sample frames to save.")
    parser.add_argument("--output-dir", type=Path, default=APP_DIR / "outputs" / "task2_simulation", help="Output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Video not found: {args.video}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ref_cap = cv2.VideoCapture(str(args.video))
    user_cap = cv2.VideoCapture(str(args.video))
    if not ref_cap.isOpened() or not user_cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    user_start = max(0, args.ref_start - args.simulated_user_lag)
    ref_cap.set(cv2.CAP_PROP_POS_FRAMES, args.ref_start)
    user_cap.set(cv2.CAP_PROP_POS_FRAMES, user_start)

    estimator = PoseEstimator(args.model)
    matcher = TemporalPoseMatcher(max_lag_frames=args.lag_window, keypoint_conf=args.keypoint_conf)
    ref_tracker = PoseStreamTracker()
    user_tracker = PoseStreamTracker()
    sample_start = min(args.frames - 1, max(3, args.simulated_user_lag))
    sample_window = max(1, args.frames - sample_start)
    sample_interval = max(1, sample_window // max(1, args.save_samples))
    rows = []
    saved_samples = []
    score_sum = 0.0
    valid_scores = 0
    lag_hits = 0
    processed = 0
    started = time.perf_counter()

    for i in range(args.frames):
        ref_ok, ref_frame = ref_cap.read()
        user_ok, user_frame = user_cap.read()
        if not ref_ok or not user_ok:
            break

        ref_frame_index = args.ref_start + i
        user_frame_index = user_start + i
        ref_result, user_result = estimator.infer_batch(
            [ref_frame, user_frame],
            frame_indices=[ref_frame_index, user_frame_index],
            conf=args.conf,
            keypoint_conf=args.keypoint_conf,
        )
        ref_result = ref_tracker.update(ref_result)
        user_result = user_tracker.update(user_result)
        matcher.push_reference(ref_result)
        match = matcher.match_user(user_result)

        expected_ready = i >= args.simulated_user_lag
        if expected_ready and match.common_keypoints >= 6:
            score_sum += match.score
            valid_scores += 1
            if match.lag_frames is not None and abs(match.lag_frames - args.simulated_user_lag) <= 2:
                lag_hits += 1

        rows.append(
            {
                "step": i,
                "ref_frame": ref_frame_index,
                "user_frame": user_frame_index,
                "score": round(match.score, 3),
                "feedback": match.feedback,
                "matched_ref_frame": match.matched_ref_frame,
                "lag_frames": match.lag_frames,
                "common_keypoints": match.common_keypoints,
                "distance_score": round(match.distance_score, 4),
                "angle_score": round(match.angle_score, 4),
                "limb_score": round(match.limb_score, 4),
                "quality": round(match.quality, 4),
                "mirror_used": match.mirror_used,
            }
        )

        sample_offset = i - sample_start
        should_save_sample = (
            args.save_samples > 0
            and i >= sample_start
            and sample_offset % sample_interval == 0
            and len(saved_samples) < args.save_samples
        )
        if should_save_sample:
            ref_view, _ = draw_pose_overlay(ref_frame, ref_result, keypoint_conf=args.keypoint_conf, main_only=True)
            user_view, _ = draw_pose_overlay(user_frame, user_result, keypoint_conf=args.keypoint_conf, main_only=True)
            draw_match_overlay(user_view, match)
            sample_name = f"task2_sim_frame{i:04d}.jpg"
            cv2.imwrite(str(args.output_dir / sample_name), make_combined_snapshot(ref_view, user_view))
            saved_samples.append(sample_name)

        processed += 1
        if processed == 1 or processed % 30 == 0 or processed == args.frames:
            elapsed = time.perf_counter() - started
            print(f"processed {processed}/{args.frames} pairs | {processed / max(elapsed, 1e-6):.1f} pair-fps", flush=True)

    ref_cap.release()
    user_cap.release()
    elapsed = time.perf_counter() - started

    summary = {
        "video": str(args.video),
        "model": str(args.model),
        "ref_start": args.ref_start,
        "simulated_user_lag": args.simulated_user_lag,
        "lag_window": args.lag_window,
        "processed_pairs": processed,
        "valid_scored_pairs": valid_scores,
        "avg_score_after_lag": score_sum / max(valid_scores, 1),
        "lag_hit_rate_after_lag": lag_hits / max(valid_scores, 1),
        "pair_fps": processed / max(elapsed, 1e-6),
        "sample_files": saved_samples,
    }

    with (args.output_dir / "task2_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "ref_frame",
                "user_frame",
                "score",
                "feedback",
                "matched_ref_frame",
                "lag_frames",
                "common_keypoints",
                "distance_score",
                "angle_score",
                "limb_score",
                "quality",
                "mirror_used",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "task2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
