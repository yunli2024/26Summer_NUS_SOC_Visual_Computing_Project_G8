from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Sequence

import cv2
import joblib
import numpy as np

from expression_effects import ExpressionPrediction, LabelSmoother, apply_expression_effect, draw_predictions, softmax
from expression_features import ExpertFeatureExtractor
from face_pipeline import DetectionResult, FpsMeter, HaarFaceDetector, LbfLandmarkEstimator, draw_detections, draw_status


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CASCADE = PROJECT_DIR / "haarcascade_frontalface_default.xml"
DEFAULT_LBF_MODEL = PROJECT_DIR / "lbfmodel.yaml"
DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "expression_classifier.joblib"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Expert demo: real-time landmark-based expression classification.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index used by cv2.VideoCapture.")
    parser.add_argument("--width", type=int, default=960, help="Requested capture width.")
    parser.add_argument("--height", type=int, default=540, help="Requested capture height.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Trained expression classifier path.")
    parser.add_argument("--cascade", type=Path, default=DEFAULT_CASCADE, help="Haar cascade XML path.")
    parser.add_argument("--lbf-model", type=Path, default=DEFAULT_LBF_MODEL, help="OpenCV LBF landmark model path.")
    parser.add_argument("--min-neighbors", type=int, default=4, help="Haar strictness.")
    parser.add_argument("--min-face-size", type=int, default=60, help="Minimum face size in pixels.")
    parser.add_argument("--max-faces", type=int, default=4, help="Classify up to N faces.")
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
    return payload["model"], payload["classes"]


def predict_expressions(model, classes: List[str], faces, smoother: LabelSmoother) -> List[ExpressionPrediction]:
    predictions: List[ExpressionPrediction] = []
    for face in faces:
        feature = face.vector.reshape(1, -1)
        start = time.perf_counter()
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(feature).reshape(-1)
            label_index = int(np.argmax(probabilities))
            confidence = float(probabilities[label_index])
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(feature).reshape(-1)
            label_index = int(np.argmax(scores))
            confidence = float(softmax(scores)[label_index])
        else:
            label_index = int(model.predict(feature)[0])
            confidence = 1.0
        prediction_ms = (time.perf_counter() - start) * 1000.0
        label = classes[label_index]
        if len(faces) == 1:
            label = smoother.update(label)
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
        model, classes = load_classifier(args.model)
        detector = HaarFaceDetector(
            cascade_path=args.cascade,
            min_neighbors=args.min_neighbors,
            min_face_size=args.min_face_size,
            preprocess="clahe",
        )
        landmark_estimator = LbfLandmarkEstimator(args.lbf_model)
        feature_extractor = ExpertFeatureExtractor(
            detector,
            landmark_estimator,
            max_faces=args.max_faces,
            use_center_fallback=False,
        )
    except Exception as exc:
        print(f"[startup error] {exc}", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(args.camera_index)
    configure_camera(cap, args.width, args.height)
    if not cap.isOpened():
        print(f"[camera error] Cannot open webcam index {args.camera_index}.", file=sys.stderr)
        return 3

    smoother = LabelSmoother(window=5)
    if args.benchmark_frames <= 0:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(args.window_name, args.width, args.height)
    fps_meter = FpsMeter()
    frame_index = 0
    benchmark_started = time.perf_counter()
    total_pipeline_ms = 0.0
    total_prediction_ms = 0.0
    frames_with_faces = 0
    total_predicted_faces = 0
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
            faces = feature_extractor.extract(frame)
            predictions = predict_expressions(model, classes, faces, smoother)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            if args.benchmark_frames > 0:
                frame_index += 1
                is_warmup = frame_index <= args.warmup_frames
                if not is_warmup:
                    total_pipeline_ms += elapsed_ms
                    total_prediction_ms += sum(prediction.prediction_ms for prediction in predictions)
                    measured_frames += 1
                    total_predicted_faces += len(predictions)
                    if faces:
                        frames_with_faces += 1
                print(
                    f"frame={frame_index}/{args.warmup_frames + args.benchmark_frames} "
                    f"warmup={str(is_warmup).lower()} "
                    f"faces={len(faces)} labels={summarize_predictions(predictions)} "
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
                detector_name=f"haar+rbf-svm:{primary_label}",
                preprocess="clahe",
                face_count=len(faces),
                elapsed_ms=elapsed_ms,
            )
            cv2.imshow(args.window_name, frame)
            frame_index += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                output_path = save_snapshot(frame, PROJECT_DIR)
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
            f"frames_with_faces={frames_with_faces} "
            f"predicted_faces={total_predicted_faces}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
