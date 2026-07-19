"""Minimal command entry for the final ROI Ensemble."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config
from dataset import inspect_dataset, print_counts, prepare_dataset_if_needed
from image_preprocessing import PREPROCESS_MODES
from realtime_demo import run_demo


def cmd_inspect(_args: argparse.Namespace) -> int:
    root = prepare_dataset_if_needed()
    print(f"Expert directory: {config.EXPERT_DIR}")
    print(f"Dataset root: {root}")
    print(f"Base checkpoint exists: {config.ROI_CNN_DEMO_MODEL_FILE.exists()} - {config.ROI_CNN_DEMO_MODEL_FILE}")
    print(f"Robust checkpoint exists: {config.ROBUST_ROI_CNN_MODEL_FILE.exists()} - {config.ROBUST_ROI_CNN_MODEL_FILE}")
    print(f"Haar model exists: {config.HAAR_CASCADE_PATH.exists()} - {config.HAAR_CASCADE_PATH}")
    print(f"LBF model exists: {config.LBF_MODEL_PATH.exists()} - {config.LBF_MODEL_PATH}")
    print_counts(inspect_dataset())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ROI Ensemble facial-expression inference.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect", help="Check the retained dataset, checkpoints, and face models.")

    demo_parser = subparsers.add_parser("demo", help="Run the realtime ROI Ensemble demo.")
    demo_parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Camera index.")
    demo_parser.add_argument(
        "--model-source",
        choices=["roi-ensemble"],
        default="roi-ensemble",
        help="The retained inference model.",
    )
    demo_parser.add_argument("--multi-face", action="store_true", help="Track and display up to five faces.")
    demo_parser.add_argument("--debug-hud", action="store_true", help="Show realtime tracking diagnostics.")
    demo_parser.add_argument("--show-pose", action="store_true", help="Show head-pose labels.")
    demo_parser.add_argument(
        "--preprocess",
        choices=PREPROCESS_MODES,
        default=config.REALTIME_PREPROCESS_MODE,
        help="Realtime grayscale preprocessing mode.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "inspect":
        return cmd_inspect(args)
    if args.command == "demo":
        return run_demo(
            camera_index=args.camera,
            model_source=args.model_source,
            debug_hud=args.debug_hud,
            preprocess_mode=args.preprocess,
            max_faces=5 if args.multi_face else config.REALTIME_DEFAULT_MAX_FACES,
            show_pose=args.show_pose,
        )
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
