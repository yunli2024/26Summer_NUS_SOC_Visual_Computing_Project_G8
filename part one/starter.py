"""Real-time face detection and 68-point facial landmark visualization.

Run this file locally because it needs access to a webcam. Press Q or Esc to quit.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "Face Keypoints"
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CASCADE = BASE_DIR / "haarcascade_frontalface_default.xml"
DEFAULT_LBF_MODEL = BASE_DIR / "lbfmodel.yaml"

# The 68 LBF landmarks are grouped so that different facial parts are easy to see.
LANDMARK_GROUPS = (
    (range(0, 17), (0, 255, 255)),   # jaw: yellow
    (range(17, 27), (255, 200, 0)),  # eyebrows: cyan-blue
    (range(27, 36), (255, 0, 255)),  # nose: magenta
    (range(36, 48), (0, 255, 0)),    # eyes: green
    (range(48, 68), (0, 100, 255)),  # mouth: orange
)


class FaceLandmarkDetector:
    """Haar face detector followed by OpenCV's LBF landmark estimator."""

    def __init__(self, cascade_path: Path, lbf_model_path: Path) -> None:
        self._require_file(cascade_path, "Haar cascade")
        self._require_file(lbf_model_path, "LBF model")

        if not hasattr(cv2, "face") or not hasattr(cv2.face, "createFacemarkLBF"):
            raise RuntimeError(
                "This program needs the cv2.face module. Remove opencv-python and "
                "install opencv-contrib-python (see README.md)."
            )

        # Some Windows OpenCV builds cannot open model files through a Unicode
        # absolute path. Loading by the ASCII filename from its directory keeps
        # the project working when parent folders contain Chinese characters.
        with model_directory(cascade_path) as cascade_name:
            self.face_detector = cv2.CascadeClassifier(cascade_name)
        if self.face_detector.empty():
            raise RuntimeError(f"Could not load Haar cascade: {cascade_path}")

        self.facemark = cv2.face.createFacemarkLBF()
        with model_directory(lbf_model_path) as model_name:
            self.facemark.loadModel(model_name)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @staticmethod
    def _require_file(path: Path, description: str) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"{description} not found: {path}")

    def detect(
        self,
        frame: np.ndarray,
        use_clahe: bool = True,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Return face rectangles and one (68, 2) landmark array per fitted face."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # CLAHE enhances local contrast and is more useful than global histogram
        # equalization when only part of the face is dark or backlit.
        detection_image = self.clahe.apply(gray) if use_clahe else gray
        faces = self.face_detector.detectMultiScale(
            detection_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            return np.empty((0, 4), dtype=np.int32), []

        faces = np.asarray(faces, dtype=np.int32).reshape(-1, 4)
        success, raw_landmarks = self.facemark.fit(gray, faces)
        if not success:
            return faces, []

        landmarks = [np.asarray(points).reshape(-1, 2) for points in raw_landmarks]
        return faces, landmarks


class LandmarkSmoother:
    """Smooth landmarks over time while keeping different faces separated."""

    def __init__(self, alpha: float = 0.6) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("Smoothing alpha must be in the range (0, 1].")
        self.alpha = alpha
        self._previous: list[tuple[np.ndarray, float, np.ndarray]] = []

    def reset(self) -> None:
        self._previous.clear()

    def update(
        self,
        faces: np.ndarray,
        landmarks: list[np.ndarray],
        enabled: bool = True,
    ) -> list[np.ndarray]:
        """Apply EMA smoothing after matching faces by their center positions."""
        if not enabled or not landmarks:
            self.reset()
            return [points.copy() for points in landmarks]

        smoothed_landmarks: list[np.ndarray] = []
        next_previous: list[tuple[np.ndarray, float, np.ndarray]] = []
        unused_previous = set(range(len(self._previous)))

        for face_index, current_points in enumerate(landmarks):
            current_points = np.asarray(current_points, dtype=np.float32)
            if face_index >= len(faces):
                smoothed_landmarks.append(current_points)
                continue

            x, y, width, height = faces[face_index]
            center = np.array([x + width / 2.0, y + height / 2.0], dtype=np.float32)
            scale = max(float(width), float(height), 1.0)

            best_index = None
            best_distance = float("inf")
            for previous_index in unused_previous:
                previous_center, previous_scale, previous_points = self._previous[previous_index]
                if previous_points.shape != current_points.shape:
                    continue
                normalizer = max(scale, previous_scale, 1.0)
                distance = float(np.linalg.norm(center - previous_center) / normalizer)
                if distance < best_distance:
                    best_index = previous_index
                    best_distance = distance

            # A center movement smaller than one face width/height is considered
            # the same person. Larger jumps start a new smoothing track.
            if best_index is not None and best_distance < 1.0:
                previous_points = self._previous[best_index][2]
                output_points = (
                    self.alpha * current_points
                    + (1.0 - self.alpha) * previous_points
                )
                unused_previous.remove(best_index)
            else:
                output_points = current_points

            smoothed_landmarks.append(output_points)
            next_previous.append((center, scale, output_points))

        self._previous = next_previous
        return smoothed_landmarks


@contextmanager
def model_directory(model_path: Path):
    """Temporarily load an OpenCV model by filename instead of a Unicode path."""
    original_directory = Path.cwd()
    try:
        os.chdir(model_path.parent)
        yield model_path.name
    finally:
        os.chdir(original_directory)


def draw_results(
    frame: np.ndarray,
    faces: np.ndarray,
    landmarks: list[np.ndarray],
) -> None:
    """Draw face boxes, landmark indices/groups, and face count in-place."""
    for index, (x, y, width, height) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 80, 40), 2)
        cv2.putText(
            frame,
            f"Face {index + 1}",
            (x, max(22, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 80, 40),
            2,
            cv2.LINE_AA,
        )

        if index >= len(landmarks):
            continue

        points = np.rint(landmarks[index]).astype(np.int32)
        for indices, color in LANDMARK_GROUPS:
            for point_index in indices:
                if point_index < len(points):
                    cv2.circle(frame, tuple(points[point_index]), 2, color, -1, cv2.LINE_AA)

    cv2.putText(
        frame,
        f"Faces: {len(faces)}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    """Open a webcam and request the desired resolution."""
    backends = (
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "MSMF"),
        (cv2.CAP_ANY, "default"),
    ) if sys.platform == "win32" else ((cv2.CAP_ANY, "default"),)

    capture = None
    attempted_backends = []
    for backend, backend_name in backends:
        attempted_backends.append(backend_name)
        candidate = cv2.VideoCapture(camera_index, backend)
        if candidate.isOpened():
            capture = candidate
            break
        candidate.release()

    if capture is None:
        attempted = ", ".join(attempted_backends)
        raise RuntimeError(
            f"Could not open camera {camera_index} using {attempted}. First check that "
            "Windows Camera can see the device, then check camera permissions or try "
            "another index with --camera 1."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def run(args: argparse.Namespace) -> None:
    detector = FaceLandmarkDetector(args.cascade, args.model)
    smoother = LandmarkSmoother(args.smoothing_alpha)
    capture = open_camera(args.camera, args.width, args.height)

    use_clahe = not args.clahe_off
    use_smoothing = not args.smoothing_off

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)

    smoothed_fps = 0.0
    previous_time = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("The camera stopped returning frames.")

            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            faces, raw_landmarks = detector.detect(frame, use_clahe=use_clahe)
            landmarks = smoother.update(faces, raw_landmarks, enabled=use_smoothing)
            draw_results(frame, faces, landmarks)

            cv2.putText(
                frame,
                f"[I] CLAHE: {'ON' if use_clahe else 'OFF'}  |  "
                f"[S] Smoothing: {'ON' if use_smoothing else 'OFF'}",
                (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            current_time = time.perf_counter()
            instant_fps = 1.0 / max(current_time - previous_time, 1e-6)
            smoothed_fps = instant_fps if smoothed_fps == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            previous_time = current_time

            cv2.putText(
                frame,
                f"FPS: {smoothed_fps:.1f}  |  Q/Esc: quit",
                (12, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if key in (ord("i"), ord("I")):
                use_clahe = not use_clahe
            if key in (ord("s"), ord("S")):
                use_smoothing = not use_smoothing
                smoother.reset()
    finally:
        capture.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="webcam index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="requested frame width")
    parser.add_argument("--height", type=int, default=480, help="requested frame height")
    parser.add_argument("--no-mirror", action="store_true", help="do not mirror webcam frames")
    parser.add_argument("--clahe-off", action="store_true", help="start with CLAHE disabled")
    parser.add_argument("--smoothing-off", action="store_true", help="start with smoothing disabled")
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.6,
        help="EMA weight for current landmarks in (0, 1] (default: 0.6)",
    )
    parser.add_argument("--cascade", type=Path, default=DEFAULT_CASCADE, help="Haar cascade XML path")
    parser.add_argument("--model", type=Path, default=DEFAULT_LBF_MODEL, help="LBF YAML model path")
    return parser.parse_args()


def main() -> int:
    try:
        run(parse_args())
    except (FileNotFoundError, RuntimeError, ValueError, cv2.error) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
