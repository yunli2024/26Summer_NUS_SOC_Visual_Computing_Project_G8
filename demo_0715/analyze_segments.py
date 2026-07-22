from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = APP_DIR / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Bonus Task 1 analysis on several reference-video segments.")
    parser.add_argument("--starts", default="0,450,900,1350,1800", help="Comma-separated segment start frames.")
    parser.add_argument("--frames", type=int, default=120, help="Frames per segment.")
    parser.add_argument("--conf", type=float, default=0.20, help="YOLO detection confidence.")
    parser.add_argument("--keypoint-conf", type=float, default=0.20, help="Keypoint visibility confidence.")
    parser.add_argument("--save-samples", type=int, default=3, help="Samples to save per segment.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    starts = [int(item.strip()) for item in args.starts.split(",") if item.strip()]
    segment_summaries = []

    for start in starts:
        run_name = f"segment_start{start}_frames{args.frames}"
        cmd = [
            sys.executable,
            str(APP_DIR / "analyze_reference.py"),
            "--start-frame",
            str(start),
            "--frames",
            str(args.frames),
            "--conf",
            str(args.conf),
            "--keypoint-conf",
            str(args.keypoint_conf),
            "--save-samples",
            str(args.save_samples),
            "--run-name",
            run_name,
            "--output-dir",
            str(args.output_dir),
        ]
        print("running", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
        summary_path = args.output_dir / run_name / "task1_summary.json"
        segment_summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))

    total_processed = sum(item["processed_frames"] for item in segment_summaries)
    total_measured = sum(item["measured_frames"] for item in segment_summaries)
    total_person = sum(item["frames_with_person"] for item in segment_summaries)
    total_multi = sum(item["frames_with_multi_person"] for item in segment_summaries)
    weighted_infer = sum(item["measured_avg_inference_ms"] * item["measured_frames"] for item in segment_summaries)
    weighted_visible = sum(item["measured_avg_main_visible_keypoints"] * item["measured_frames"] for item in segment_summaries)

    aggregate = {
        "segments": starts,
        "frames_per_segment": args.frames,
        "processed_frames": total_processed,
        "measured_frames": total_measured,
        "frames_with_person": total_person,
        "frames_with_multi_person": total_multi,
        "person_detection_rate": total_person / max(total_processed, 1),
        "multi_person_rate": total_multi / max(total_processed, 1),
        "measured_avg_inference_ms": weighted_infer / max(total_measured, 1),
        "measured_avg_main_visible_keypoints": weighted_visible / max(total_measured, 1),
        "segment_summaries": segment_summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "task1_segments_summary.json"
    output_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
