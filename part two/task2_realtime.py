"""Part Two Task 2: real-time expression recognition and video effects."""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import joblib
import numpy as np

from expression_features import FEATURE_VERSION, normalize_landmarks


BASE_DIR = Path(__file__).resolve().parent
PART_ONE_DIR = BASE_DIR.parent / "part one"
if str(PART_ONE_DIR) not in sys.path:
    sys.path.insert(0, str(PART_ONE_DIR))

from starter import FaceLandmarkDetector, LandmarkSmoother, open_camera  # noqa: E402


WINDOW_NAME = "Real-time Expression Effects"
DEFAULT_MODEL = BASE_DIR / "artifacts" / "expression_classifier.joblib"
EXPRESSION_COLORS = {
    "angry": (40, 60, 255),
    "disgust": (70, 200, 80),
    "fear": (210, 80, 210),
    "happy": (0, 220, 255),
    "neutral": (220, 210, 70),
    "sad": (255, 140, 50),
    "surprise": (0, 150, 255),
}


class FaceProbabilitySmoother:
    """EMA-smooth class probabilities after matching faces by center position."""

    def __init__(self, alpha: float = 0.35) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("Probability smoothing alpha must be in (0, 1].")
        self.alpha = alpha
        self._previous: list[tuple[np.ndarray, float, np.ndarray]] = []

    def reset(self) -> None:
        self._previous.clear()

    def update(
        self,
        faces: np.ndarray,
        probabilities: np.ndarray,
        enabled: bool,
    ) -> np.ndarray:
        if len(probabilities) == 0:
            self.reset()
            return probabilities
        if not enabled:
            self.reset()
            return probabilities.copy()

        outputs: list[np.ndarray] = []
        next_previous: list[tuple[np.ndarray, float, np.ndarray]] = []
        unused = set(range(len(self._previous)))

        for face, current in zip(faces, probabilities):
            x, y, width, height = face
            center = np.array([x + width / 2.0, y + height / 2.0], dtype=np.float32)
            scale = max(float(width), float(height), 1.0)
            best_index = None
            best_distance = float("inf")
            for previous_index in unused:
                previous_center, previous_scale, previous_probability = self._previous[previous_index]
                if previous_probability.shape != current.shape:
                    continue
                distance = float(
                    np.linalg.norm(center - previous_center)
                    / max(scale, previous_scale, 1.0)
                )
                if distance < best_distance:
                    best_index = previous_index
                    best_distance = distance

            if best_index is not None and best_distance < 1.0:
                previous_probability = self._previous[best_index][2]
                output = self.alpha * current + (1.0 - self.alpha) * previous_probability
                unused.remove(best_index)
            else:
                output = current

            output = output / max(float(output.sum()), 1e-8)
            outputs.append(output)
            next_previous.append((center, scale, output))

        self._previous = next_previous
        return np.stack(outputs).astype(np.float32)


def clipped_box(frame: np.ndarray, face: np.ndarray, padding: float = 0.08) -> tuple[int, int, int, int]:
    x, y, width, height = map(int, face)
    pad_x, pad_y = int(width * padding), int(height * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + width + pad_x)
    y2 = min(frame.shape[0], y + height + pad_y)
    return x1, y1, x2, y2


def tint_region(frame: np.ndarray, face: np.ndarray, color: tuple[int, int, int], alpha: float) -> None:
    x1, y1, x2, y2 = clipped_box(frame, face, 0.05)
    if x2 <= x1 or y2 <= y1:
        return
    region = frame[y1:y2, x1:x2]
    tint = np.full_like(region, color)
    cv2.addWeighted(tint, alpha, region, 1.0 - alpha, 0, region)


def draw_star(
    frame: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
) -> None:
    points = []
    for index in range(10):
        angle = -math.pi / 2 + index * math.pi / 5
        length = radius if index % 2 == 0 else radius * 0.42
        points.append(
            [int(center[0] + math.cos(angle) * length), int(center[1] + math.sin(angle) * length)]
        )
    cv2.fillPoly(frame, [np.asarray(points, dtype=np.int32)], color, cv2.LINE_AA)
    cv2.polylines(frame, [np.asarray(points, dtype=np.int32)], True, (255, 255, 255), 1, cv2.LINE_AA)


def apply_expression_effect(
    frame: np.ndarray,
    face: np.ndarray,
    expression: str,
    frame_index: int,
) -> None:
    """Draw a distinct animated effect for one predicted expression."""
    x, y, width, height = map(int, face)
    center_x, center_y = x + width // 2, y + height // 2
    unit = max(4, min(width, height) // 12)
    phase = frame_index * 0.12

    if expression == "happy":
        tint_region(frame, face, (20, 180, 255), 0.12)
        for index in range(8):
            angle = phase + index * math.pi / 4
            px = int(center_x + math.cos(angle) * width * 0.67)
            py = int(center_y + math.sin(angle) * height * 0.62)
            draw_star(frame, (px, py), unit, (0, 230, 255))

    elif expression == "surprise":
        tint_region(frame, face, (0, 140, 255), 0.10)
        star_x = min(frame.shape[1] - unit * 2 - 1, x + width + unit * 2)
        star_y = max(unit * 2 + 1, y + unit * 2 + int(math.sin(phase) * unit))
        draw_star(frame, (star_x, star_y), unit * 2, (0, 180, 255))
        cv2.putText(frame, "!", (star_x - unit // 2, star_y + unit // 2), cv2.FONT_HERSHEY_DUPLEX, 1.0, (40, 40, 200), 2, cv2.LINE_AA)

    elif expression == "angry":
        tint_region(frame, face, (30, 30, 255), 0.22)
        for direction in (-1, 1):
            start = (center_x + direction * width // 3, y + height // 4)
            end = (center_x + direction * width // 6, y + height // 3)
            cv2.line(frame, start, end, (20, 20, 255), max(2, unit // 2), cv2.LINE_AA)
        for index in range(5):
            offset = (index - 2) * unit * 2
            cv2.line(frame, (center_x + offset, max(0, y - unit)), (center_x + offset * 2, max(0, y - unit * 3)), (0, 80, 255), 2, cv2.LINE_AA)

    elif expression == "sad":
        tint_region(frame, face, (255, 120, 20), 0.18)
        for index in range(7):
            px = x + int((index + 0.5) * width / 7)
            py = y + int((frame_index * 7 + index * 31) % max(height, 1))
            cv2.line(frame, (px, py), (px - unit // 2, py + unit * 2), (255, 190, 80), 2, cv2.LINE_AA)

    elif expression == "fear":
        tint_region(frame, face, (180, 40, 170), 0.16)
        for offset in (unit, unit * 2):
            cv2.rectangle(
                frame,
                (max(0, x - offset), max(0, y - offset)),
                (min(frame.shape[1] - 1, x + width + offset), min(frame.shape[0] - 1, y + height + offset)),
                (210, 80, 210),
                1,
                cv2.LINE_AA,
            )

    elif expression == "disgust":
        tint_region(frame, face, (50, 190, 60), 0.24)
        for index in range(6):
            angle = phase * 0.4 + index * math.pi / 3
            px = int(center_x + math.cos(angle) * width * 0.58)
            py = int(center_y + math.sin(angle) * height * 0.52)
            radius = unit + (index % 3) * 2
            cv2.circle(frame, (px, py), radius, (80, 220, 90), 2, cv2.LINE_AA)

    else:  # neutral
        color = EXPRESSION_COLORS["neutral"]
        corner = unit * 2
        for px, py, dx, dy in (
            (x, y, 1, 1),
            (x + width, y, -1, 1),
            (x, y + height, 1, -1),
            (x + width, y + height, -1, -1),
        ):
            cv2.line(frame, (px, py), (px + dx * corner, py), color, 2, cv2.LINE_AA)
            cv2.line(frame, (px, py), (px, py + dy * corner), color, 2, cv2.LINE_AA)


def draw_landmarks(frame: np.ndarray, landmarks: np.ndarray) -> None:
    for x, y in np.rint(landmarks).astype(np.int32):
        cv2.circle(frame, (int(x), int(y)), 1, (80, 255, 100), -1, cv2.LINE_AA)


def draw_prediction(
    frame: np.ndarray,
    face: np.ndarray,
    expression: str,
    confidence: float,
) -> None:
    x, y, width, height = map(int, face)
    color = EXPRESSION_COLORS.get(expression, (255, 255, 255))
    cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2, cv2.LINE_AA)
    text = f"{expression.upper()}  {confidence * 100:.0f}%"
    (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.58, 1)
    label_y = max(text_height + baseline + 4, y)
    cv2.rectangle(
        frame,
        (x, label_y - text_height - baseline - 6),
        (min(frame.shape[1] - 1, x + text_width + 10), label_y + 2),
        color,
        -1,
    )
    cv2.putText(frame, text, (x + 5, label_y - baseline - 1), cv2.FONT_HERSHEY_DUPLEX, 0.58, (20, 20, 20), 1, cv2.LINE_AA)


def load_model(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Expression model not found: {model_path}")
    bundle = joblib.load(model_path)
    if bundle.get("feature_version") != FEATURE_VERSION:
        raise RuntimeError(
            f"Feature mismatch: model uses {bundle.get('feature_version')!r}, "
            f"application expects {FEATURE_VERSION!r}."
        )
    classifier = bundle.get("model")
    if classifier is None or not hasattr(classifier, "predict"):
        raise RuntimeError("The model bundle does not contain a valid classifier.")
    return classifier


def predict_probabilities(classifier, features: np.ndarray) -> np.ndarray:
    if hasattr(classifier, "predict_proba"):
        return np.asarray(classifier.predict_proba(features), dtype=np.float32)
    scores = np.asarray(classifier.decision_function(features), dtype=np.float32)
    scores -= scores.max(axis=1, keepdims=True)
    exponentials = np.exp(scores)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def run(args: argparse.Namespace) -> None:
    classifier = load_model(args.expression_model)
    classes = np.asarray(classifier.classes_)
    detector = FaceLandmarkDetector(
        PART_ONE_DIR / "haarcascade_frontalface_default.xml",
        PART_ONE_DIR / "lbfmodel.yaml",
    )
    landmark_smoother = LandmarkSmoother(args.landmark_alpha)
    probability_smoother = FaceProbabilitySmoother(args.probability_alpha)
    capture = open_camera(args.camera, args.width, args.height)

    effects_enabled = not args.effects_off
    landmarks_enabled = not args.landmarks_off
    smoothing_enabled = not args.smoothing_off
    clahe_enabled = not args.clahe_off
    frame_index = 0
    previous_time = time.perf_counter()
    smoothed_fps = 0.0
    inference_ms = 0.0

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("The camera stopped returning frames.")
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            faces, raw_landmarks = detector.detect(frame, use_clahe=clahe_enabled)
            smoothed_landmarks = landmark_smoother.update(
                faces, raw_landmarks, enabled=smoothing_enabled
            )

            valid_faces: list[np.ndarray] = []
            valid_landmarks: list[np.ndarray] = []
            feature_rows: list[np.ndarray] = []
            for face, points in zip(faces, smoothed_landmarks):
                try:
                    feature_rows.append(normalize_landmarks(points))
                    valid_faces.append(face)
                    valid_landmarks.append(points)
                except ValueError:
                    continue

            if feature_rows:
                feature_matrix = np.stack(feature_rows).astype(np.float32)
                inference_started = time.perf_counter()
                probabilities = predict_probabilities(classifier, feature_matrix)
                inference_ms = (time.perf_counter() - inference_started) * 1000.0 / len(feature_rows)
                face_array = np.stack(valid_faces).astype(np.int32)
                probabilities = probability_smoother.update(
                    face_array, probabilities, enabled=smoothing_enabled
                )
            else:
                face_array = np.empty((0, 4), dtype=np.int32)
                probabilities = np.empty((0, len(classes)), dtype=np.float32)
                probability_smoother.reset()

            for face, points, probability in zip(face_array, valid_landmarks, probabilities):
                best_index = int(np.argmax(probability))
                expression = str(classes[best_index])
                confidence = float(probability[best_index])
                if effects_enabled:
                    apply_expression_effect(frame, face, expression, frame_index)
                if landmarks_enabled:
                    draw_landmarks(frame, points)
                draw_prediction(frame, face, expression, confidence)

            current_time = time.perf_counter()
            instant_fps = 1.0 / max(current_time - previous_time, 1e-6)
            smoothed_fps = instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            previous_time = current_time

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 66), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
            cv2.putText(
                frame,
                f"FPS {smoothed_fps:.1f} | classifier {inference_ms:.1f} ms | faces {len(face_array)}",
                (12, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"[E] effects {'ON' if effects_enabled else 'OFF'}  "
                f"[L] points {'ON' if landmarks_enabled else 'OFF'}  "
                f"[S] smooth {'ON' if smoothing_enabled else 'OFF'}  "
                f"[C] CLAHE {'ON' if clahe_enabled else 'OFF'}  [Q] quit",
                (12, 52),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("e"), ord("E")):
                effects_enabled = not effects_enabled
            elif key in (ord("l"), ord("L")):
                landmarks_enabled = not landmarks_enabled
            elif key in (ord("s"), ord("S")):
                smoothing_enabled = not smoothing_enabled
                landmark_smoother.reset()
                probability_smoother.reset()
            elif key in (ord("c"), ord("C")):
                clahe_enabled = not clahe_enabled

            frame_index += 1
    finally:
        capture.release()
        cv2.destroyAllWindows()


def create_effect_preview(output_path: Path) -> None:
    """Create a deterministic visual QA sheet without opening a webcam."""
    tile_width, tile_height = 300, 260
    canvas = np.full((tile_height * 2, tile_width * 4, 3), 35, dtype=np.uint8)
    for index, expression in enumerate(EXPRESSION_COLORS):
        row, column = divmod(index, 4)
        x0, y0 = column * tile_width, row * tile_height
        tile = canvas[y0 : y0 + tile_height, x0 : x0 + tile_width]
        face = np.array([75, 55, 150, 150], dtype=np.int32)
        cv2.ellipse(tile, (150, 130), (65, 82), 0, 0, 360, (145, 165, 185), -1, cv2.LINE_AA)
        cv2.circle(tile, (125, 115), 6, (35, 35, 35), -1, cv2.LINE_AA)
        cv2.circle(tile, (175, 115), 6, (35, 35, 35), -1, cv2.LINE_AA)
        cv2.ellipse(tile, (150, 158), (25, 12), 0, 0, 180, (45, 45, 45), 3, cv2.LINE_AA)
        apply_expression_effect(tile, face, expression, 12)
        draw_prediction(tile, face, expression, 0.75)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"Could not save preview: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--expression-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--effects-off", action="store_true")
    parser.add_argument("--landmarks-off", action="store_true")
    parser.add_argument("--smoothing-off", action="store_true")
    parser.add_argument("--clahe-off", action="store_true")
    parser.add_argument("--landmark-alpha", type=float, default=0.6)
    parser.add_argument("--probability-alpha", type=float, default=0.35)
    parser.add_argument("--preview", type=Path, help="render an effect preview and exit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.preview:
            create_effect_preview(args.preview)
        else:
            run(args)
    except (FileNotFoundError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
