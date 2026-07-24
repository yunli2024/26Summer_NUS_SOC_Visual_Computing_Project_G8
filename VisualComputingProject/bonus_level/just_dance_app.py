"""Bonus Task 2: real-time Just Dance style pose-matching game.

The reference video is preprocessed by pose_analyzer.py.  At game time only
the webcam is sent through YOLO, which keeps the CPU version usable.
"""

from __future__ import annotations

import argparse
from collections import deque
import os
import subprocess
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
PROJECT_DIR = BASE_DIR.parent
YOLO_CONFIG_PATH = Path(tempfile.gettempdir()) / "visual-computing-yolo"
YOLO_CONFIG_PATH.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_PATH))

from ultralytics import YOLO  # noqa: E402

from dance_scoring import (  # noqa: E402
    HoldStateFilter,
    MatchResult,
    best_reference_match,
    feedback_for_score,
)
from pose_analyzer import PrimaryDancerTracker, draw_other_people, draw_pose, extract_detections  # noqa: E402


DEFAULT_REFERENCE = BASE_DIR / "task2_results" / "dance_example_1" / "annotated.mp4"
DEFAULT_CACHE = BASE_DIR / "task2_results" / "dance_example_1" / "pose_cache.npz"
DEFAULT_MODEL = PROJECT_DIR / "resources" / "pose_models" / "yolov8n-pose.pt"
DEFAULT_SOURCE_VIDEO = PROJECT_DIR / "resources" / "videos" / "dance_example_1.mp4"
DEFAULT_PREPARE_OUTPUT = BASE_DIR / "task2_results"
DISPLAY_SIZE = (600, 338)


@dataclass(frozen=True)
class LiveResult:
    generation: int
    frame: np.ndarray
    points: Optional[np.ndarray]
    valid: Optional[np.ndarray]
    inference_ms: float
    timestamp: float


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


def prepare_reference_assets(args: argparse.Namespace) -> bool:
    """Create the default annotated video and pose cache when they are absent."""
    reference_ready = args.reference.is_file()
    cache_ready = args.cache.is_file()
    force_rebuild = bool(args.rebuild_reference)
    if reference_ready and cache_ready and not force_rebuild:
        return False

    using_default_outputs = (
        args.reference.resolve() == DEFAULT_REFERENCE.resolve()
        and args.cache.resolve() == DEFAULT_CACHE.resolve()
    )
    if not using_default_outputs:
        missing = [
            str(path)
            for path in (args.reference, args.cache)
            if not path.is_file()
        ]
        raise FileNotFoundError(
            "Custom reference inputs are incomplete:\n"
            + "\n".join(f"  - {path}" for path in missing)
            + "\nGenerate them with pose_analyzer.py or use the default paths."
        )
    if args.no_auto_prepare and not force_rebuild:
        missing = [
            str(path)
            for path in (args.reference, args.cache)
            if not path.is_file()
        ]
        raise FileNotFoundError(
            "Dance reference assets are missing and automatic preparation is disabled:\n"
            + "\n".join(f"  - {path}" for path in missing)
        )
    if not DEFAULT_SOURCE_VIDEO.is_file():
        raise FileNotFoundError(f"Source dance video not found: {DEFAULT_SOURCE_VIDEO}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Pose model not found: {args.model}")

    print("Dance reference assets are missing; preparing them now.", flush=True)
    print("This one-time CPU preprocessing can take several minutes.", flush=True)
    command = [
        sys.executable,
        str(BASE_DIR / "pose_analyzer.py"),
        str(DEFAULT_SOURCE_VIDEO),
        "--model",
        str(args.model),
        "--output",
        str(DEFAULT_PREPARE_OUTPUT),
        "--image-size",
        str(args.prepare_image_size),
        "--stride",
        str(args.prepare_stride),
        "--contact-every",
        "120",
    ]
    completed = subprocess.run(command, cwd=str(BASE_DIR), check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "Reference preprocessing failed. Review the pose_analyzer.py output above."
        )
    if not args.reference.is_file() or not args.cache.is_file():
        raise RuntimeError(
            "Reference preprocessing finished but the expected runtime files were not created:\n"
            f"  - {args.reference}\n"
            f"  - {args.cache}"
        )
    print(f"Reference video ready: {args.reference}", flush=True)
    print(f"Pose cache ready: {args.cache}", flush=True)
    return True


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
        self.pose_history: deque[
            tuple[float, np.ndarray, np.ndarray]
        ] = deque(maxlen=32)
        self.hold_filter = HoldStateFilter()
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
                    time.perf_counter(),
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
        self.pose_history.clear()
        self.hold_filter.reset()
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
            self.pose_history.clear()
            self.hold_filter.reset()
            self.pause_button.configure(text="Resume")
            self.status_var.set("Paused")
        else:
            self.paused_duration += time.perf_counter() - self.pause_started
            self.paused = False
            self.active_event.set()
            self.pose_history.clear()
            self.hold_filter.reset()
            self.pause_button.configure(text="Pause")
            self.status_var.set("Playing")

    def stop_game(self) -> None:
        self.running = False
        self.paused = False
        self.active_event.clear()
        self.pose_history.clear()
        self.hold_filter.reset()
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
        if live.generation == self.last_scored_generation:
            return
        self.last_scored_generation = live.generation
        if live.points is None or live.valid is None:
            self.pose_history.clear()
            self.hold_filter.reset()
            self.latest_feedback = "NO POSE"
            return

        target_age = self.args.motion_window
        minimum_age = max(0.10, target_age * 0.50)
        maximum_age = max(0.80, target_age * 2.50)
        while (
            self.pose_history
            and live.timestamp - self.pose_history[0][0] > maximum_age
        ):
            self.pose_history.popleft()
        eligible_history = [
            item
            for item in self.pose_history
            if minimum_age <= live.timestamp - item[0] <= maximum_age
        ]
        previous = (
            min(
                eligible_history,
                key=lambda item: abs(
                    (live.timestamp - item[0]) - target_age
                ),
            )
            if eligible_history
            else None
        )
        previous_points = previous[1] if previous is not None else None
        previous_valid = previous[2] if previous is not None else None
        motion_delta_frames = (
            max(
                1,
                int(
                    round(
                        (live.timestamp - previous[0])
                        * self.reference_fps
                    )
                ),
            )
            if previous is not None
            else 1
        )

        max_lag = int(round(self.args.max_lag * self.reference_fps))
        match = best_reference_match(
            live.points, live.valid,
            self.reference_points, self.reference_valid,
            self.reference_index, max_lag,
            allow_mirror=self.allow_mirror.get(),
            player_previous_points=previous_points,
            player_previous_valid=previous_valid,
            motion_delta_frames=motion_delta_frames,
        )
        self.pose_history.append(
            (live.timestamp, live.points.copy(), live.valid.copy())
        )
        if match is None:
            self.hold_filter.reset()
            self.latest_feedback = "NO POSE"
            return
        self.latest_match = match
        reference_is_holding = self.hold_filter.update(
            match.reference_motion, match.motion_used
        )
        label, game_points, colour = feedback_for_score(match.score)
        score_event = True
        if not match.motion_used:
            label, game_points, colour = "SYNC", 0, (220, 220, 220)
            score_event = False
        elif reference_is_holding:
            label, game_points, colour = "HOLD", 0, (220, 210, 70)
            score_event = False
        elif (
            match.motion_used
            and match.player_motion < max(0.06, 0.45 * match.reference_motion)
        ):
            label, game_points, colour = "MOVE!", 0, (80, 80, 255)
        self.latest_feedback = label
        self.latest_colour = colour
        if score_event:
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
            if self.latest_match.motion_used:
                detail += (
                    f"   motion {self.latest_match.motion_score:.0f}"
                    f"  activity {self.latest_match.player_motion:.2f}"
                    f"/{self.latest_match.reference_motion:.2f}"
                )
                if self.hold_filter.smoothed_motion is not None:
                    detail += f"  ref~{self.hold_filter.smoothed_motion:.2f}"
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
    if not args.model.is_file():
        video.release()
        raise FileNotFoundError(f"Model not found: {args.model}")
    usable = np.flatnonzero(np.count_nonzero(valid[:, 5:17], axis=1) >= 4)
    if usable.size == 0:
        video.release()
        raise RuntimeError("No scoreable pose exists in the reference cache.")
    index = int(usable[0])
    video.set(cv2.CAP_PROP_POS_FRAMES, index)
    frame_ok, sample_frame = video.read()
    video.release()
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
    parser.add_argument(
        "--motion-window",
        type=float,
        default=0.40,
        help="Seconds of player/reference history used for motion scoring",
    )
    parser.add_argument("--no-mirror", action="store_true", help="Require anatomical left/right to match exactly")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Prepare default reference files when needed, then exit without opening the GUI",
    )
    parser.add_argument(
        "--rebuild-reference",
        action="store_true",
        help="Regenerate the default annotated video and pose cache",
    )
    parser.add_argument(
        "--no-auto-prepare",
        action="store_true",
        help="Fail instead of generating missing default reference files",
    )
    parser.add_argument(
        "--prepare-stride",
        type=int,
        default=2,
        help="Process every Nth reference frame during one-time preparation (default: 2)",
    )
    parser.add_argument(
        "--prepare-image-size",
        type=int,
        default=416,
        help="YOLO image size used during one-time reference preparation",
    )
    parser.add_argument("--check", action="store_true", help="Validate files and scoring without opening the GUI")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.max_lag < 0.0 or args.motion_window <= 0.0:
            raise ValueError("--max-lag must be non-negative and --motion-window must be positive.")
        if args.prepare_stride < 1 or args.prepare_image_size < 160:
            raise ValueError("--prepare-stride must be at least 1 and --prepare-image-size at least 160.")
        prepare_reference_assets(args)
        if args.prepare_only:
            print("Dance reference preparation completed.")
            return 0
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
