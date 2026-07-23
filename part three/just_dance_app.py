"""Bonus Task 2: real-time Just Dance style pose-matching game.

The reference video is preprocessed by pose_analyzer.py.  At game time only
the webcam is sent through YOLO, which keeps the CPU version usable.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(tempfile.gettempdir()) / "visual-computing-yolo"))

from ultralytics import YOLO  # noqa: E402

from dance_scoring import MatchResult, best_reference_match, feedback_for_score  # noqa: E402
from pose_analyzer import PrimaryDancerTracker, draw_other_people, draw_pose, extract_detections  # noqa: E402


DEFAULT_REFERENCE = BASE_DIR / "task2_results" / "dance_example_1" / "annotated.mp4"
DEFAULT_CACHE = BASE_DIR / "task2_results" / "dance_example_1" / "pose_cache.npz"
DEFAULT_MODEL = BASE_DIR / "yolov8n-pose.pt"
DISPLAY_SIZE = (600, 338)


@dataclass(frozen=True)
class LiveResult:
    generation: int
    frame: np.ndarray
    points: Optional[np.ndarray]
    valid: Optional[np.ndarray]
    inference_ms: float


def open_camera(camera_index: int, width: int = 960, height: int = 540) -> cv2.VideoCapture:
    backends = (
        ((cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "default"))
        if sys.platform == "win32"
        else ((cv2.CAP_ANY, "default"),)
    )
    attempted: list[str] = []
    for backend, name in backends:
        attempted.append(name)
        capture = cv2.VideoCapture(camera_index, backend)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture
        capture.release()
    raise RuntimeError(f"Could not open camera {camera_index} using {', '.join(attempted)}.")


def load_pose_cache(path: Path) -> dict[str, np.ndarray | float]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Reference pose cache not found: {path}\n"
            "Run pose_analyzer.py first to generate pose_cache.npz."
        )
    with np.load(path, allow_pickle=False) as cache:
        required = {"points", "valid", "playback_fps"}
        missing = required.difference(cache.files)
        if missing:
            raise ValueError(f"Invalid pose cache; missing: {', '.join(sorted(missing))}")
        points = cache["points"].astype(np.float32)
        valid = cache["valid"].astype(bool)
        playback_fps = float(np.asarray(cache["playback_fps"]).reshape(()))
    if points.ndim != 3 or points.shape[1:] != (17, 2) or valid.shape != points.shape[:2]:
        raise ValueError("Invalid pose cache dimensions.")
    if playback_fps <= 0:
        raise ValueError("Invalid playback FPS in pose cache.")
    return {"points": points, "valid": valid, "playback_fps": playback_fps}


class DanceGameApp:
    def __init__(self, root, args: argparse.Namespace) -> None:
        import tkinter as tk
        from tkinter import messagebox

        self.tk = tk
        self.messagebox = messagebox
        self.root = root
        self.args = args
        self.root.title("Just Dance Pose Challenge — Bonus Task 2")
        self.root.geometry("1260x660")
        self.root.minsize(1050, 600)
        self.root.configure(bg="#171923")

        cache = load_pose_cache(args.cache)
        self.reference_points = cache["points"]
        self.reference_valid = cache["valid"]
        self.reference_fps = float(cache["playback_fps"])
        self.reference_cap = cv2.VideoCapture(str(args.reference))
        if not self.reference_cap.isOpened():
            raise RuntimeError(f"Could not open reference video: {args.reference}")
        video_frames = int(self.reference_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = min(video_frames, len(self.reference_points))
        if self.frame_count <= 0:
            raise RuntimeError("Reference video/cache contains no frames.")

        self.model = YOLO(str(args.model))
        self.stop_event = threading.Event()
        self.active_event = threading.Event()
        self.live_lock = threading.Lock()
        self.live_result: Optional[LiveResult] = None
        self.camera_thread: Optional[threading.Thread] = None
        self.camera_error: Optional[str] = None

        self.running = False
        self.paused = False
        self.finished = False
        self.start_time = 0.0
        self.pause_started = 0.0
        self.paused_duration = 0.0
        self.reference_index = -1
        self.reference_frame: Optional[np.ndarray] = None
        self.last_scored_generation = -1
        self.total_score = 0
        self.combo = 0
        self.best_combo = 0
        self.score_sum = 0.0
        self.score_count = 0
        self.latest_match: Optional[MatchResult] = None
        self.latest_feedback = "READY"
        self.latest_colour = (220, 220, 220)
        self.allow_mirror = tk.BooleanVar(value=not args.no_mirror)

        self.status_var = tk.StringVar(value="Ready — press Start")
        self.score_var = tk.StringVar(value="SCORE  0")
        self.combo_var = tk.StringVar(value="COMBO  0")
        self.average_var = tk.StringVar(value="AVERAGE  --")
        self._build_ui()
        self._show_initial_reference()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(20, self._tick)

    def _build_ui(self) -> None:
        tk = self.tk
        header = tk.Frame(self.root, bg="#171923")
        header.pack(fill=tk.X, padx=18, pady=(14, 8))
        tk.Label(
            header, text="JUST DANCE  ·  POSE CHALLENGE", font=("Segoe UI", 20, "bold"),
            fg="#65e6ff", bg="#171923",
        ).pack(side=tk.LEFT)
        tk.Label(
            header, textvariable=self.status_var, font=("Segoe UI", 11),
            fg="#f0f0f0", bg="#171923",
        ).pack(side=tk.RIGHT)

        panels = tk.Frame(self.root, bg="#171923")
        panels.pack(fill=tk.BOTH, expand=True, padx=18)
        left = tk.Frame(panels, bg="#232638", bd=1, relief=tk.FLAT)
        right = tk.Frame(panels, bg="#232638", bd=1, relief=tk.FLAT)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 7))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(7, 0))
        tk.Label(left, text="REFERENCE DANCER", font=("Segoe UI", 12, "bold"), fg="#ffd166", bg="#232638").pack(pady=8)
        tk.Label(right, text="YOU", font=("Segoe UI", 12, "bold"), fg="#80ed99", bg="#232638").pack(pady=8)
        self.reference_label = tk.Label(left, bg="#090a10")
        self.reference_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.live_label = tk.Label(right, bg="#090a10")
        self.live_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        scoreboard = tk.Frame(self.root, bg="#171923")
        scoreboard.pack(fill=tk.X, padx=18, pady=(8, 5))
        for variable, colour in (
            (self.score_var, "#65e6ff"),
            (self.combo_var, "#ff70a6"),
            (self.average_var, "#ffd166"),
        ):
            tk.Label(scoreboard, textvariable=variable, font=("Consolas", 15, "bold"), fg=colour, bg="#171923").pack(side=tk.LEFT, padx=(0, 32))

        controls = tk.Frame(self.root, bg="#171923")
        controls.pack(fill=tk.X, padx=18, pady=(3, 14))
        button_style = {"font": ("Segoe UI", 10, "bold"), "width": 13, "bd": 0, "padx": 5, "pady": 7}
        tk.Button(controls, text="Start / Restart", command=self.start_game, bg="#65e6ff", fg="#11131b", **button_style).pack(side=tk.LEFT, padx=(0, 8))
        self.pause_button = tk.Button(controls, text="Pause", command=self.toggle_pause, bg="#ffd166", fg="#11131b", **button_style)
        self.pause_button.pack(side=tk.LEFT, padx=8)
        tk.Button(controls, text="Stop", command=self.stop_game, bg="#ff70a6", fg="#11131b", **button_style).pack(side=tk.LEFT, padx=8)
        tk.Checkbutton(
            controls, text="Accept mirrored moves", variable=self.allow_mirror,
            font=("Segoe UI", 10), fg="#f0f0f0", bg="#171923", activebackground="#171923",
            activeforeground="#f0f0f0", selectcolor="#232638",
        ).pack(side=tk.LEFT, padx=20)
        tk.Button(controls, text="Quit", command=self.close, bg="#454a63", fg="white", **button_style).pack(side=tk.RIGHT)

    def _show_initial_reference(self) -> None:
        self.reference_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = self.reference_cap.read()
        if ok:
            self.reference_frame = frame
            self.reference_index = 0
            self._set_image(self.reference_label, frame)
        blank = np.full((DISPLAY_SIZE[1], DISPLAY_SIZE[0], 3), 18, dtype=np.uint8)
        cv2.putText(blank, "Camera starts with the game", (125, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (210, 210, 210), 2, cv2.LINE_AA)
        self._set_image(self.live_label, blank)

    def _set_image(self, label, frame: np.ndarray) -> None:
        from PIL import Image, ImageTk

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize(DISPLAY_SIZE, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=image)
        label.configure(image=photo)
        label.image = photo

    def _start_camera_worker(self) -> None:
        if self.camera_thread is not None and self.camera_thread.is_alive():
            return
        self.stop_event.clear()
        self.camera_error = None
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True, name="pose-camera")
        self.camera_thread.start()

    def _camera_loop(self) -> None:
        try:
            capture = open_camera(self.args.camera, self.args.width, self.args.height)
        except Exception as exc:
            self.camera_error = str(exc)
            return
        tracker = PrimaryDancerTracker(self.args.keypoint_confidence, self.args.smoothing)
        generation = 0
        try:
            while not self.stop_event.is_set():
                if not self.active_event.is_set():
                    time.sleep(0.03)
                    continue
                ok, frame = capture.read()
                if not ok:
                    self.camera_error = "The camera stopped returning frames."
                    break
                frame = cv2.flip(frame, 1)
                started = time.perf_counter()
                predictions = self.model.predict(
                    frame, conf=self.args.confidence, imgsz=self.args.image_size,
                    device="cpu", verbose=False,
                )
                inference_ms = (time.perf_counter() - started) * 1000.0
                detections = extract_detections(predictions[0]) if predictions else []
                chosen, points, valid, _ = tracker.select(detections, frame.shape)
                annotated = frame.copy()
                draw_other_people(annotated, detections, chosen)
                if chosen is not None and points is not None and valid is not None:
                    pose_confidence = float(np.mean(chosen.keypoint_confidence[valid])) if np.any(valid) else 0.0
                    draw_pose(annotated, points, valid, chosen.box, pose_confidence)
                else:
                    cv2.putText(annotated, "NO PERSON DETECTED", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (60, 60, 255), 2, cv2.LINE_AA)
                generation += 1
                result = LiveResult(
                    generation, annotated,
                    None if points is None else points.copy(),
                    None if valid is None else valid.copy(),
                    inference_ms,
                )
                with self.live_lock:
                    self.live_result = result
        except Exception as exc:
            self.camera_error = f"Camera inference failed: {exc}"
        finally:
            capture.release()

    def start_game(self) -> None:
        self._start_camera_worker()
        self.active_event.set()
        self.reference_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.running = True
        self.paused = False
        self.finished = False
        self.start_time = time.perf_counter()
        self.paused_duration = 0.0
        self.reference_index = -1
        self.last_scored_generation = self.live_result.generation if self.live_result else -1
        self.total_score = self.combo = self.best_combo = 0
        self.score_sum = 0.0
        self.score_count = 0
        self.latest_match = None
        self.latest_feedback = "GO!"
        self.pause_button.configure(text="Pause")
        self.status_var.set("Playing")
        self._update_scoreboard()

    def toggle_pause(self) -> None:
        if not self.running or self.finished:
            return
        if not self.paused:
            self.paused = True
            self.pause_started = time.perf_counter()
            self.active_event.clear()
            self.pause_button.configure(text="Resume")
            self.status_var.set("Paused")
        else:
            self.paused_duration += time.perf_counter() - self.pause_started
            self.paused = False
            self.active_event.set()
            self.pause_button.configure(text="Pause")
            self.status_var.set("Playing")

    def stop_game(self) -> None:
        self.running = False
        self.paused = False
        self.active_event.clear()
        self.status_var.set("Stopped — press Start to restart")
        self.pause_button.configure(text="Pause")

    def _advance_reference(self) -> None:
        elapsed = time.perf_counter() - self.start_time - self.paused_duration
        wanted = min(int(elapsed * self.reference_fps), self.frame_count - 1)
        if wanted != self.reference_index:
            self.reference_cap.set(cv2.CAP_PROP_POS_FRAMES, wanted)
            ok, frame = self.reference_cap.read()
            if ok:
                self.reference_index = wanted
                self.reference_frame = frame
                self._set_image(self.reference_label, frame)
        if elapsed >= self.frame_count / self.reference_fps:
            self.running = False
            self.finished = True
            self.active_event.clear()
            self.status_var.set(f"Finished · best combo {self.best_combo}")

    def _score_live_result(self, live: LiveResult) -> None:
        if live.generation == self.last_scored_generation or live.points is None or live.valid is None:
            return
        self.last_scored_generation = live.generation
        max_lag = int(round(self.args.max_lag * self.reference_fps))
        match = best_reference_match(
            live.points, live.valid,
            self.reference_points, self.reference_valid,
            self.reference_index, max_lag,
            allow_mirror=self.allow_mirror.get(),
        )
        if match is None:
            self.latest_feedback = "NO POSE"
            return
        self.latest_match = match
        label, game_points, colour = feedback_for_score(match.score)
        self.latest_feedback = label
        self.latest_colour = colour
        self.total_score += game_points
        self.score_sum += match.score
        self.score_count += 1
        if game_points > 0:
            self.combo += 1
            self.best_combo = max(self.best_combo, self.combo)
        else:
            self.combo = 0
        self._update_scoreboard()

    def _update_scoreboard(self) -> None:
        self.score_var.set(f"SCORE  {self.total_score:,}")
        self.combo_var.set(f"COMBO  {self.combo}")
        average = self.score_sum / self.score_count if self.score_count else None
        self.average_var.set("AVERAGE  --" if average is None else f"AVERAGE  {average:5.1f}")

    def _decorate_live(self, live: LiveResult) -> np.ndarray:
        frame = live.frame.copy()
        score = self.latest_match.score if self.latest_match else 0.0
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 86), (15, 17, 28), -1)
        cv2.putText(frame, self.latest_feedback, (20, 42), cv2.FONT_HERSHEY_DUPLEX, 1.15, self.latest_colour, 2, cv2.LINE_AA)
        detail = f"similarity {score:5.1f}   inference {live.inference_ms:.0f} ms"
        if self.latest_match:
            lag_seconds = self.latest_match.lag_frames / self.reference_fps
            mirror_text = "  mirror" if self.latest_match.mirrored else ""
            detail += f"   lag {lag_seconds:.2f}s{mirror_text}"
        cv2.putText(frame, detail, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        return frame

    def _tick(self) -> None:
        if self.camera_error:
            error = self.camera_error
            self.camera_error = None
            self.stop_game()
            self.messagebox.showerror("Camera error", error)

        if self.running and not self.paused:
            self._advance_reference()

        with self.live_lock:
            live = self.live_result
        if live is not None:
            if self.running and not self.paused:
                self._score_live_result(live)
            self._set_image(self.live_label, self._decorate_live(live))

        if not self.stop_event.is_set():
            self.root.after(25, self._tick)

    def close(self) -> None:
        self.running = False
        self.active_event.set()
        self.stop_event.set()
        self.reference_cap.release()
        self.root.destroy()


def validate_inputs(args: argparse.Namespace) -> int:
    cache = load_pose_cache(args.cache)
    points = cache["points"]
    valid = cache["valid"]
    video = cv2.VideoCapture(str(args.reference))
    if not video.isOpened():
        raise RuntimeError(f"Could not open reference video: {args.reference}")
    video_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_ok, sample_frame = video.read()
    video.release()
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    usable = np.flatnonzero(np.count_nonzero(valid[:, 5:17], axis=1) >= 4)
    if usable.size == 0:
        raise RuntimeError("No scoreable pose exists in the reference cache.")
    index = int(usable[0])
    match = best_reference_match(points[index], valid[index], points, valid, index, 0)
    if not frame_ok:
        raise RuntimeError("Could not decode a frame from the reference video.")
    model = YOLO(str(args.model))
    prediction = model.predict(
        sample_frame, conf=args.confidence, imgsz=args.image_size, device="cpu", verbose=False,
    )[0]
    detected_people = len(extract_detections(prediction))
    print("Bonus Task 2 validation passed")
    print(f"  cached poses: {len(points)}")
    print(f"  video frames: {video_frames}")
    print(f"  playback FPS: {cache['playback_fps']:.3f}")
    print(f"  self-match score: {match.score:.2f}" if match else "  self-match score: unavailable")
    print(f"  YOLO people in sample frame: {detected_people}")
    print(f"  model: {args.model.name}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bonus Task 2: Just Dance pose matching game")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE, help="Annotated reference MP4")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="pose_cache.npz from Task 1")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLO pose model")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--image-size", type=int, default=416, help="YOLO inference size; lower is faster")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument("--keypoint-confidence", type=float, default=0.35)
    parser.add_argument("--smoothing", type=float, default=0.60)
    parser.add_argument("--max-lag", type=float, default=0.80, help="Allowed reaction delay in seconds")
    parser.add_argument("--no-mirror", action="store_true", help="Require anatomical left/right to match exactly")
    parser.add_argument("--check", action="store_true", help="Validate files and scoring without opening the GUI")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.check:
            return validate_inputs(args)
        import tkinter as tk

        root = tk.Tk()
        DanceGameApp(root, args)
        root.mainloop()
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
