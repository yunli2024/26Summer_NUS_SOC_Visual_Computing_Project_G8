from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import joblib
import numpy as np

from expression_effects import ExpressionPrediction, apply_expression_effect, draw_predictions, softmax
from keypoint_features import (
    LEGACY_FEATURE_VERSION,
    ZHANGYX_FEATURE_VERSION,
    ExpertFeatureExtractor,
)
from face_pipeline import (
    DetectionResult,
    FaceDetector,
    FpsMeter,
    HaarFaceDetector,
    LbfLandmarkEstimator,
    YuNetFaceDetector,
    draw_detections,
    draw_status,
)
from realtime_stability import MultiFaceStabilizer, ProbabilitySmoother, TrackedFace


SRC_DIR = Path(__file__).resolve().parent
EXPERT_DIR = SRC_DIR.parent
PROJECT_DIR = EXPERT_DIR.parent
FACE_MODELS_DIR = PROJECT_DIR / "resources" / "face_models"
DEFAULT_CASCADE = FACE_MODELS_DIR / "haarcascade_frontalface_default.xml"
DEFAULT_YUNET_MODEL = FACE_MODELS_DIR / "face_detection_yunet_2023mar.onnx"
DEFAULT_LBF_MODEL = FACE_MODELS_DIR / "lbfmodel.yaml"
DEFAULT_MODEL_PATH = EXPERT_DIR / "models" / "keypoint" / "current" / "expression_classifier.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expert demo: real-time landmark-based expression classification.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index used by cv2.VideoCapture.")
    parser.add_argument("--width", type=int, default=960, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=540, help="Requested capture height.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Trained expression classifier path.")
    parser.add_argument("--detector", choices=("yunet", "haar"), default="yunet", help="Face detector used before LBF fitting.")
    parser.add_argument("--yunet-model", type=Path, default=DEFAULT_YUNET_MODEL, help="YuNet ONNX face detector path.")
    parser.add_argument("--yunet-score-threshold", type=float, default=0.75, help="YuNet face confidence threshold.")
    parser.add_argument("--yunet-nms-threshold", type=float, default=0.30, help="YuNet overlap suppression threshold.")
    parser.add_argument("--yunet-input-size", type=int, default=640, help="Maximum YuNet input side; 0 keeps full resolution.")
    parser.add_argument("--cascade", type=Path, default=DEFAULT_CASCADE, help="Haar cascade XML path.")
    parser.add_argument("--lbf-model", type=Path, default=DEFAULT_LBF_MODEL, help="OpenCV LBF landmark model path.")
    parser.add_argument("--min-neighbors", type=int, default=5, help="Haar strictness.")
    parser.add_argument("--min-face-size", type=int, default=60, help="Minimum face size in pixels.")
    parser.add_argument("--max-faces", type=int, default=4, help="Classify up to N faces.")
    parser.add_argument("--min-detection-weight", type=float, default=1.0, help="Reject weak Haar candidates below this confidence-like weight.")
    parser.add_argument("--stable-frames", type=int, default=2, help="Require N consecutive plausible face frames before display.")
    parser.add_argument("--hold-frames", type=int, default=5, help="Keep last stable face for N missed frames.")
    parser.add_argument("--min-face-area-ratio", type=float, default=0.006, help="Reject faces smaller than this frame-area ratio.")
    parser.add_argument("--landmark-smoothing", type=float, default=0.85, help="EMA weight for per-track facial landmarks.")
    parser.add_argument("--min-confidence", type=float, default=0.48, help="Minimum smoothed expression confidence before switching.")
    parser.add_argument("--switch-margin", type=float, default=0.12, help="Required margin over second class before label switching can begin.")
    parser.add_argument("--prob-smoothing", type=float, default=0.85, help="EMA weight for expression probabilities.")
    parser.add_argument("--initial-label-frames", type=int, default=6, help="Consecutive frames required for the first expression label.")
    parser.add_argument("--switch-frames", type=int, default=8, help="Consecutive frames required to switch expression labels.")
    parser.add_argument("--min-label-hold-frames", type=int, default=30, help="Minimum frames to hold a confirmed expression label.")
    parser.add_argument("--mirror", action="store_true", help="Mirror webcam frames for a selfie-style display.")
    parser.add_argument("--no-effects", action="store_true", help="Disable expression-driven visual effects.")
    parser.add_argument("--benchmark-frames", type=int, default=0, help="Run N frames without opening a display window.")
    parser.add_argument("--warmup-frames", type=int, default=3, help="Warmup frames to ignore in benchmark metrics.")
    parser.add_argument("--window-name", default="Expert Expression Demo", help="OpenCV display window name.")
    return parser.parse_args()


def configure_camera(cap: cv2.VideoCapture, width: int, height: int) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def save_snapshot(frame, project_dir: Path) -> Path:
    snapshot_dir = project_dir / "snapshots"
    snapshot_dir.mkdir(exist_ok=True)
    output_path = snapshot_dir / f"expert_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(str(output_path), frame)
    return output_path


def load_classifier(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}. Run train_expert.py first."
        )
    payload = joblib.load(model_path)
    feature_version = payload.get("feature_version", LEGACY_FEATURE_VERSION)
    default_name = (
        "zhangyx_geometry_rbf_svm"
        if feature_version == ZHANGYX_FEATURE_VERSION
        else "legacy_rbf_svm"
    )
    return (
        payload["model"],
        payload["classes"],
        feature_version,
        payload.get("selected_model", payload.get("classifier", default_name)),
    )


def create_face_detector(args: argparse.Namespace) -> FaceDetector:
    if args.detector == "yunet":
        return YuNetFaceDetector(
            args.yunet_model,
            score_threshold=args.yunet_score_threshold,
            nms_threshold=args.yunet_nms_threshold,
            min_face_size=args.min_face_size,
            max_input_size=args.yunet_input_size,
        )
    return HaarFaceDetector(
        cascade_path=args.cascade,
        min_neighbors=args.min_neighbors,
        min_face_size=args.min_face_size,
        preprocess="clahe",
        min_detection_weight=args.min_detection_weight,
    )


def predict_expressions(
    model,
    classes: List[str],
    tracked_faces: Sequence[TrackedFace],
    smoothers: Dict[int, ProbabilitySmoother],
    *,
    alpha: float,
    min_confidence: float,
    switch_margin: float,
    initial_frames: int,
    switch_frames: int,
    min_hold_frames: int,
) -> List[ExpressionPrediction]:
    predictions: List[ExpressionPrediction] = []
    for tracked_face in tracked_faces:
        face = tracked_face.face
        feature = face.vector.reshape(1, -1)
        start = time.perf_counter()
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(feature).reshape(-1)
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(feature).reshape(-1)
            probabilities = softmax(scores)
        else:
            label_index = int(model.predict(feature)[0])
            probabilities = np.zeros(len(classes), dtype=np.float32)
            probabilities[label_index] = 1.0
        prediction_ms = (time.perf_counter() - start) * 1000.0
        smoother = smoothers.setdefault(
            tracked_face.track_id,
            ProbabilitySmoother(
                alpha=alpha,
                min_confidence=min_confidence,
                switch_margin=switch_margin,
                initial_frames=initial_frames,
                switch_frames=switch_frames,
                min_hold_frames=min_hold_frames,
            ),
        )
        label, confidence, _ = smoother.update(probabilities, classes)
        predictions.append(ExpressionPrediction(face=face, label=label, confidence=confidence, prediction_ms=prediction_ms))
    return predictions


def summarize_predictions(predictions: Sequence[ExpressionPrediction]) -> str:
    if not predictions:
        return "none"
    labels = [prediction.label for prediction in predictions]
    return ",".join(labels)


def main() -> int:
    args = parse_args()
    try:
        model, classes, feature_version, model_name = load_classifier(args.model)
        detector = create_face_detector(args)
        landmark_estimator = LbfLandmarkEstimator(args.lbf_model)
        feature_extractor = ExpertFeatureExtractor(
            detector,
            landmark_estimator,
            max_faces=args.max_faces,
            use_center_fallback=False,
            feature_version=feature_version,
        )
    except Exception as exc:
        print(f"[startup error] {exc}", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(args.camera_index)
    configure_camera(cap, args.width, args.height)
    if not cap.isOpened():
        print(f"[camera error] Cannot open webcam index {args.camera_index}.", file=sys.stderr)
        return 3

    expression_smoothers: Dict[int, ProbabilitySmoother] = {}
    face_stabilizer = MultiFaceStabilizer(
        max_faces=args.max_faces,
        stable_frames=args.stable_frames,
        hold_frames=args.hold_frames,
        landmark_smoothing=args.landmark_smoothing,
        min_area_ratio=args.min_face_area_ratio,
    )
    if args.benchmark_frames <= 0:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(args.window_name, args.width, args.height)
    fps_meter = FpsMeter()
    frame_index = 0
    benchmark_started = time.perf_counter()
    total_pipeline_ms = 0.0
    total_prediction_ms = 0.0
    frames_with_faces = 0
    frames_with_raw_faces = 0
    total_raw_faces = 0
    total_predicted_faces = 0
    benchmark_labels: Dict[int, str] = {}
    label_switches = 0
    measured_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[camera warning] Failed to read a frame; exiting.", file=sys.stderr)
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)

            start = time.perf_counter()
            raw_faces = feature_extractor.extract(frame)
            tracked_faces = face_stabilizer.update(raw_faces, frame.shape)
            active_track_ids = {tracked_face.track_id for tracked_face in tracked_faces}
            for track_id in list(expression_smoothers):
                if track_id not in active_track_ids:
                    expression_smoothers.pop(track_id, None)
            faces = [tracked_face.face for tracked_face in tracked_faces]
            predictions = predict_expressions(
                model,
                classes,
                tracked_faces,
                expression_smoothers,
                alpha=args.prob_smoothing,
                min_confidence=args.min_confidence,
                switch_margin=args.switch_margin,
                initial_frames=args.initial_label_frames,
                switch_frames=args.switch_frames,
                min_hold_frames=args.min_label_hold_frames,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if args.benchmark_frames > 0:
                frame_index += 1
                is_warmup = frame_index <= args.warmup_frames
                if not is_warmup:
                    total_pipeline_ms += elapsed_ms
                    total_prediction_ms += sum(prediction.prediction_ms for prediction in predictions)
                    measured_frames += 1
                    total_raw_faces += len(raw_faces)
                    total_predicted_faces += len(predictions)
                    for tracked_face, prediction in zip(tracked_faces, predictions):
                        if prediction.label == "uncertain":
                            continue
                        previous_label = benchmark_labels.get(tracked_face.track_id)
                        if previous_label is not None and previous_label != prediction.label:
                            label_switches += 1
                        benchmark_labels[tracked_face.track_id] = prediction.label
                    if raw_faces:
                        frames_with_raw_faces += 1
                    if faces:
                        frames_with_faces += 1
                print(
                    f"frame={frame_index}/{args.warmup_frames + args.benchmark_frames} "
                    f"warmup={str(is_warmup).lower()} "
                    f"raw_faces={len(raw_faces)} faces={len(faces)} labels={summarize_predictions(predictions)} "
                    f"pipeline_ms={elapsed_ms:.2f}",
                    flush=True,
                )
                if frame_index >= args.warmup_frames + args.benchmark_frames:
                    break
                continue

            primary_label = predictions[0].label if predictions else "none"
            if predictions and not args.no_effects:
                apply_expression_effect(frame, primary_label, frame_index)

            detections = [DetectionResult(box=face.box, landmarks=face.landmarks) for face in faces]
            draw_detections(frame, detections)
            draw_predictions(frame, predictions)
            draw_status(
                frame,
                fps=fps_meter.tick(),
                detector_name=f"{detector.name}+{model_name}:{primary_label}",
                preprocess="none" if detector.name == "yunet" else "clahe",
                face_count=len(faces),
                elapsed_ms=elapsed_ms,
            )
            cv2.imshow(args.window_name, frame)
            frame_index += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                output_path = save_snapshot(frame, EXPERT_DIR)
                print(f"[snapshot] saved to {output_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if args.benchmark_frames > 0:
        elapsed_seconds = max(time.perf_counter() - benchmark_started, 1e-9)
        processed = max(measured_frames, 1)
        avg_prediction_ms = total_prediction_ms / max(total_predicted_faces, 1)
        print(
            "benchmark_summary "
            f"measured_frames={measured_frames} "
            f"warmup_frames={min(frame_index, args.warmup_frames)} "
            f"overall_fps={frame_index / elapsed_seconds:.2f} "
            f"avg_pipeline_ms={total_pipeline_ms / processed:.2f} "
            f"avg_prediction_ms_per_face={avg_prediction_ms:.4f} "
            f"frames_with_raw_faces={frames_with_raw_faces} "
            f"frames_with_faces={frames_with_faces} "
            f"raw_faces={total_raw_faces} "
            f"predicted_faces={total_predicted_faces} "
            f"label_switches={label_switches}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
