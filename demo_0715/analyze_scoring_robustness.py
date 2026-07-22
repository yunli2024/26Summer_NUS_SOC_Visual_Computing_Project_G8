from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pose_pipeline import PoseEstimator, PoseStreamTracker
from pose_scoring import compare_poses, normalize_main_pose


APP_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO = APP_DIR / "dance_example_1.mp4"
DEFAULT_MODEL = APP_DIR / "yolov8n-pose.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether the scorer separates nearby and clearly different dance poses.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--start", type=int, default=900)
    parser.add_argument("--near-offset", type=int, default=8)
    parser.add_argument("--wrong-offset", type=int, default=180)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--keypoint-conf", type=float, default=0.20)
    parser.add_argument("--output", type=Path, default=APP_DIR / "outputs" / "scoring_robustness_summary.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    caps = [cv2.VideoCapture(str(args.video)) for _ in range(3)]
    starts = [args.start, args.start + args.near_offset, args.start + args.wrong_offset]
    for cap, start in zip(caps, starts):
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {args.video}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    estimator = PoseEstimator(args.model)
    trackers = [PoseStreamTracker() for _ in range(3)]
    near_scores: list[float] = []
    wrong_scores: list[float] = []

    for sample_index in range(args.samples):
        frames = []
        valid = True
        for cap in caps:
            frame = None
            for _ in range(args.stride):
                ok, frame = cap.read()
                if not ok:
                    valid = False
                    break
            if not valid or frame is None:
                break
            frames.append(frame)
        if not valid:
            break

        base_frame = args.start + sample_index * args.stride
        indices = [base_frame, base_frame + args.near_offset, base_frame + args.wrong_offset]
        results = estimator.infer_batch(
            frames,
            frame_indices=indices,
            conf=args.conf,
            keypoint_conf=args.keypoint_conf,
        )
        results = [tracker.update(result) for tracker, result in zip(trackers, results)]
        poses = [normalize_main_pose(result, keypoint_conf=args.keypoint_conf) for result in results]
        if any(pose is None for pose in poses):
            continue
        reference, near, wrong = poses
        near_scores.append(compare_poses(reference, near).score)
        wrong_scores.append(compare_poses(reference, wrong).score)

    for cap in caps:
        cap.release()

    summary = {
        "model": str(args.model),
        "valid_samples": len(near_scores),
        "near_offset": args.near_offset,
        "wrong_offset": args.wrong_offset,
        "near_mean": mean(near_scores),
        "near_p10": percentile(near_scores, 10),
        "wrong_mean": mean(wrong_scores),
        "wrong_p90": percentile(wrong_scores, 90),
        "wrong_super_rate": float(np.mean(np.asarray(wrong_scores) >= 85.0)) if wrong_scores else 0.0,
        "mean_separation": mean(near_scores) - mean(wrong_scores),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(values, value)) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
