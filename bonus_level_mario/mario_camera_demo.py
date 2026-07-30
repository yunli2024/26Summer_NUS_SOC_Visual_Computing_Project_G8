"""Camera-controlled, Mario-style platform game demo.

This is deliberately isolated from Bonus Task 1 and Task 2. It reuses their
read-only YOLO pose utilities but does not modify their code or outputs.
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


DEMO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DEMO_DIR.parent
ASSET_DIR = DEMO_DIR / "assets"
MODEL_PATH = PROJECT_DIR / "resources" / "pose_models" / "yolov8n-pose.pt"
SAMPLE_VIDEO = PROJECT_DIR / "resources" / "videos" / "dance_example_1.mp4"
YOLO_CONFIG_PATH = Path(tempfile.gettempdir()) / "visual-computing-yolo"
YOLO_CONFIG_PATH.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_PATH))
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from gesture_controller import GestureController, GestureState  # noqa: E402
from pose_analyzer import PrimaryDancerTracker, draw_other_people, draw_pose, extract_detections  # noqa: E402


GAME_WIDTH = 880
GAME_HEIGHT = 560
CAMERA_WIDTH = 390
CAMERA_HEIGHT = 230
JUMP_BUFFER_SECONDS = 0.28
LEVEL_WIDTH = 3900.0
GROUND_Y = 480.0
MAX_LIVES = 10


@dataclass
class Player:
    x: float = 110.0
    y: float = 426.0
    vx: float = 0.0
    vy: float = 0.0
    width: float = 38.0
    height: float = 54.0
    grounded: bool = False
    crouching: bool = False
    invincible_until: float = 0.0
    facing: int = 1
    powered: bool = False


@dataclass
class Coin:
    x: float
    y: float
    collected: bool = False


@dataclass
class Enemy:
    x: float
    y: float
    left_bound: float
    right_bound: float
    direction: float = 1.0
    alive: bool = True


@dataclass
class Block:
    x: float
    y: float
    kind: str = "brick"
    contents: str = ""
    used: bool = False
    active: bool = True


@dataclass(frozen=True)
class Pipe:
    x: float
    y: float
    width: float
    height: float


@dataclass
class MovingPlatform:
    x: float
    y: float
    width: float
    height: float
    left_bound: float
    right_bound: float
    speed: float = 55.0
    direction: float = 1.0
    delta_x: float = 0.0


@dataclass
class PowerUp:
    x: float
    y: float
    collected: bool = False


@dataclass(frozen=True)
class CameraPacket:
    generation: int
    frame: np.ndarray
    gesture: GestureState
    inference_ms: float
    timestamp: float


class SoundPlayer:
    """Queue WAV effects so Windows never truncates one with the next."""

    def __init__(self, asset_dir: Path) -> None:
        self.asset_dir = asset_dir
        self.available = sys.platform == "win32"
        self.last_played = ""
        self.last_error = ""
        self._queue: queue.Queue[Optional[Path]] = queue.Queue(maxsize=12)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        if self.available:
            self._worker = threading.Thread(
                target=self._playback_loop,
                name="mario-sound-player",
                daemon=True,
            )
            self._worker.start()

    def _playback_loop(self) -> None:
        try:
            import winsound
        except ImportError:
            self.available = False
            return
        while not self._stop_event.is_set():
            try:
                path = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if path is None:
                break
            try:
                # Deliberately synchronous inside this dedicated thread. The
                # GUI stays responsive and each short effect finishes before
                # the next queued effect begins.
                winsound.PlaySound(
                    str(path),
                    winsound.SND_FILENAME | winsound.SND_NODEFAULT,
                )
                self.last_played = path.stem
            except RuntimeError as exc:
                self.last_error = str(exc)
                self.available = False
                break
            finally:
                self._queue.task_done()

    def play(self, name: str) -> None:
        if not self.available:
            return
        path = self.asset_dir / f"{name}.wav"
        if not path.is_file():
            return
        try:
            self._queue.put_nowait(path)
        except queue.Full:
            # Prefer a current action sound over an old queued one.
            try:
                self._queue.get_nowait()
                self._queue.task_done()
                self._queue.put_nowait(path)
            except (queue.Empty, queue.Full):
                pass

    def stop(self) -> None:
        if self.available:
            self._stop_event.set()
            try:
                import winsound

                self._queue.put_nowait(None)
                winsound.PlaySound(None, 0)
                if self._worker is not None:
                    self._worker.join(timeout=0.5)
            except (ImportError, RuntimeError, queue.Full):
                pass


def empty_gesture(label: str = "CAMERA OFF") -> GestureState:
    return GestureController.empty_state(label)


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    backends = (
        ((cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "MSMF"), (cv2.CAP_ANY, "default"))
        if sys.platform == "win32"
        else ((cv2.CAP_ANY, "default"),)
    )
    attempted: list[str] = []
    for backend, name in backends:
        attempted.append(name)
        capture = cv2.VideoCapture(index, backend)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return capture
        capture.release()
    raise RuntimeError(f"Could not open camera {index} using {', '.join(attempted)}.")


class CameraPlatformDemo:
    def __init__(self, root, args: argparse.Namespace) -> None:
        import tkinter as tk
        from tkinter import messagebox

        self.root = root
        self.tk = tk
        self.messagebox = messagebox
        self.args = args
        self.root.title("Mario Camera Jump Demo")
        self.root.geometry("1345x650")
        self.root.resizable(False, False)
        self.root.configure(bg="#151827")

        self.packet_lock = threading.Lock()
        self.latest_packet: Optional[CameraPacket] = None
        self.camera_thread: Optional[threading.Thread] = None
        self.camera_stop = threading.Event()
        self.reset_gesture_event = threading.Event()
        self.camera_error: Optional[str] = None
        self.camera_status = "Camera off"
        self.last_packet_generation = -1
        self.active_gesture = empty_gesture()
        self.sounds = SoundPlayer(ASSET_DIR)
        self.images: dict[str, object] = {}

        self.keys: set[str] = set()
        self.player = Player()
        self.platforms: list[tuple[float, float, float, float]] = []
        self.blocks: list[Block] = []
        self.pipes: list[Pipe] = []
        self.moving_platforms: list[MovingPlatform] = []
        self.powerups: list[PowerUp] = []
        self.coins: list[Coin] = []
        self.enemies: list[Enemy] = []
        self.score = 0
        self.coin_count = 0
        self.lives = MAX_LIVES
        self.finished = False
        self.started_at = time.perf_counter()
        self.last_tick = time.perf_counter()
        self.camera_x = 0.0
        self.goal_x = 3740.0
        self.checkpoint_marker_x = 1980.0
        self.checkpoint_activated = False
        self.checkpoint_x = self.player.x
        self.checkpoint_y = self.player.y
        self.input_frozen_until = 0.0
        self.jump_buffer_until = 0.0
        self.buffered_jump_height = 0.22
        self.message = "Start the camera, then control the game with your hands"
        self.message_until = time.perf_counter() + 5.0
        self._build_level()
        self._build_ui()
        self._load_assets()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(16, self._tick)

    def _build_level(self) -> None:
        """Build one forgiving level with several classic platform-game beats."""
        self.platforms = [
            # Four broad ground zones, separated by gesture-friendly gaps.
            (0, GROUND_Y, 720, 80),
            (810, GROUND_Y, 620, 80),
            (1540, GROUND_Y, 760, 80),
            (2420, GROUND_Y, 1480, 80),
            # Optional upper coin routes; the main ground route remains valid.
            (270, 390, 170, 22),
            (1320, 330, 120, 22),
            (1800, 390, 80, 22),
            (1860, 320, 170, 22),
            (2020, 360, 100, 22),
            (2460, 390, 70, 22),
            (2690, 390, 70, 22),
            (2790, 320, 180, 22),
            (3050, 390, 150, 22),
            # A wide staircase leads to the flag and teaches controlled jumps.
            (3270, 440, 70, 40),
            (3340, 400, 70, 80),
            (3410, 360, 70, 120),
            (3480, 320, 90, 160),
        ]
        self.pipes = [
            Pipe(1200, 380, 86, 100),
            Pipe(2150, 395, 80, 85),
            Pipe(2960, 415, 76, 65),
        ]
        self.blocks = [
            Block(910, 330, "question", "coin"),
            Block(952, 330, "question", "mushroom"),
            Block(994, 330, "brick"),
            Block(1036, 330, "brick"),
            Block(1660, 345, "question", "coin"),
            Block(1702, 345, "brick"),
            Block(1744, 345, "question", "coin"),
            Block(2550, 335, "brick"),
            Block(2592, 335, "question", "coin"),
            Block(2634, 335, "brick"),
        ]
        self.moving_platforms = [
            MovingPlatform(1375, 395, 125, 20, 1335, 1465, 48.0),
            MovingPlatform(2285, 380, 145, 20, 2260, 2400, 42.0),
        ]
        self.powerups = []
        self.coins = [
            Coin(315, 345), Coin(380, 345), Coin(680, 425),
            Coin(760, 405), Coin(860, 435), Coin(1110, 420),
            Coin(1360, 285), Coin(1415, 285), Coin(1480, 405),
            Coin(1640, 345), Coin(1720, 345), Coin(1905, 275),
            Coin(1980, 275), Coin(2260, 420), Coin(2355, 340),
            Coin(2525, 290), Coin(2610, 290), Coin(2830, 275),
            Coin(2905, 275), Coin(3120, 345), Coin(3335, 345),
            Coin(3420, 305), Coin(3510, 260), Coin(3670, 420),
        ]
        # Only two enemies: gesture control should emphasize traversal, not combat.
        self.enemies = [
            Enemy(1090, 446, 830, 1160, -1.0),
            Enemy(2740, 446, 2460, 2890, 1.0),
        ]

    def _build_ui(self) -> None:
        tk = self.tk
        header = tk.Frame(self.root, bg="#151827")
        header.pack(fill=tk.X, padx=15, pady=(10, 6))
        tk.Label(
            header, text="CAMERA PLATFORM ADVENTURE", font=("Segoe UI", 19, "bold"),
            fg="#ffd166", bg="#151827",
        ).pack(side=tk.LEFT)
        self.hud_var = tk.StringVar(value="")
        tk.Label(header, textvariable=self.hud_var, font=("Consolas", 13, "bold"), fg="white", bg="#151827").pack(side=tk.RIGHT)

        content = tk.Frame(self.root, bg="#151827")
        content.pack(padx=15, pady=(0, 10))
        self.canvas = tk.Canvas(content, width=GAME_WIDTH, height=GAME_HEIGHT, bg="#76c8ff", highlightthickness=0)
        self.canvas.configure(scrollregion=(0, 0, GAME_WIDTH, GAME_HEIGHT))
        self.canvas.pack(side=tk.LEFT)

        side = tk.Frame(content, width=410, height=GAME_HEIGHT, bg="#23273a")
        side.pack(side=tk.LEFT, padx=(12, 0))
        side.pack_propagate(False)
        tk.Label(side, text="LIVE POSE CONTROL", font=("Segoe UI", 12, "bold"), fg="#80ed99", bg="#23273a").pack(pady=(12, 7))
        self.camera_label = tk.Label(side, bg="#090a10", width=CAMERA_WIDTH, height=CAMERA_HEIGHT)
        self.camera_label.pack(padx=10)
        self.gesture_var = tk.StringVar(value="Gesture: CAMERA OFF")
        self.metrics_var = tk.StringVar(value="Keep your shoulders and hands visible")
        self.sound_enabled = tk.BooleanVar(value=True)
        tk.Label(side, textvariable=self.gesture_var, font=("Consolas", 15, "bold"), fg="#65e6ff", bg="#23273a").pack(pady=(7, 2))
        tk.Label(side, textvariable=self.metrics_var, font=("Segoe UI", 10), fg="#e8e8e8", bg="#23273a", wraplength=370, justify=tk.LEFT).pack(padx=14)

        instructions = (
            "Camera controls\n"
            "• Extend/raise left hand → run left\n"
            "• Extend/raise right hand → run right\n"
            "• Raise both hands → jump\n"
            "• Put both hands together at chest → crouch\n\n"
            "Keyboard fallback: A/D or ←/→, Space, S/↓"
        )
        tk.Label(side, text=instructions, font=("Segoe UI", 9), fg="#d5d8e8", bg="#23273a", justify=tk.LEFT).pack(anchor="w", padx=18, pady=(8, 5))

        controls = tk.Frame(side, bg="#23273a")
        controls.pack(fill=tk.X, padx=12, pady=3)
        for column in (0, 1):
            controls.columnconfigure(column, weight=1, uniform="controls")
        style = {"font": ("Segoe UI", 8, "bold"), "bd": 0, "pady": 5}
        tk.Button(controls, text="Start Camera", command=self.start_camera, bg="#65e6ff", fg="#10131d", **style).grid(row=0, column=0, sticky="ew", padx=2, pady=2)
        tk.Button(controls, text="Stop Camera", command=self.stop_camera, bg="#ff7b9c", fg="#10131d", **style).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        tk.Button(controls, text="Reset Gestures", command=self.reset_gestures, bg="#ffd166", fg="#10131d", **style).grid(row=1, column=0, sticky="ew", padx=2, pady=2)
        tk.Button(controls, text="Test All Sounds", command=self.test_sounds, bg="#bca7ff", fg="#10131d", **style).grid(row=1, column=1, sticky="ew", padx=2, pady=2)
        tk.Button(controls, text="Restart Level", command=self.restart_level, bg="#80ed99", fg="#10131d", **style).grid(row=2, column=0, sticky="ew", padx=2, pady=2)
        tk.Button(controls, text="Quit", command=self.close, bg="#50566f", fg="white", **style).grid(row=2, column=1, sticky="ew", padx=2, pady=2)
        tk.Checkbutton(
            controls, text="Sound effects", variable=self.sound_enabled,
            font=("Segoe UI", 9), fg="white", bg="#23273a", activebackground="#23273a",
            activeforeground="white", selectcolor="#30364f",
        ).grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=2)

        blank = np.full((CAMERA_HEIGHT, CAMERA_WIDTH, 3), 18, dtype=np.uint8)
        cv2.putText(blank, "CAMERA OFF", (112, 120), cv2.FONT_HERSHEY_DUPLEX, 0.8, (210, 210, 210), 2, cv2.LINE_AA)
        self._set_camera_image(blank)

    def _load_assets(self) -> None:
        from PIL import Image, ImageTk

        sizes = {
            **{
                f"{pose}_{direction}": (40, 60)
                for pose in ("hero_idle", "hero_run1", "hero_run2", "hero_jump", "hero_crouch")
                for direction in ("left", "right")
            },
            "enemy": (40, 40),
            "coin": (25, 35),
            "ground_tile": (32, 32),
            "platform_tile": (32, 22),
            "goal_flag": (64, 128),
        }
        for name, size in sizes.items():
            path = ASSET_DIR / f"{name}.png"
            if path.is_file():
                image = Image.open(path).convert("RGBA").resize(size, Image.Resampling.NEAREST)
                self.images[name] = ImageTk.PhotoImage(image)
        background_path = ASSET_DIR / "pixel_landscape.png"
        if background_path.is_file():
            background = Image.open(background_path).convert("RGB").resize(
                (GAME_WIDTH, GAME_HEIGHT), Image.Resampling.NEAREST,
            )
            self.images["background"] = ImageTk.PhotoImage(background)

    def _play_sound(self, name: str) -> None:
        enabled = getattr(self, "sound_enabled", None)
        if enabled is not None and enabled.get():
            self.sounds.play(name)

    def test_sounds(self) -> None:
        self.sound_enabled.set(True)
        for name in ("coin", "bump", "stomp", "hurt", "checkpoint", "win"):
            self.sounds.play(name)
        self.message = "Audio test: coin, bump, stomp, hurt, checkpoint, win"
        self.message_until = time.perf_counter() + 4.0

    def _bind_keys(self) -> None:
        self.root.bind("<KeyPress>", self._key_down)
        self.root.bind("<KeyRelease>", self._key_up)
        self.root.focus_set()

    def _key_down(self, event) -> None:
        key = event.keysym.lower()
        first_press = key not in self.keys
        self.keys.add(key)
        if first_press and key in ("space", "w", "up"):
            self._request_jump()
        if key == "r":
            self.restart_level()

    def _key_up(self, event) -> None:
        self.keys.discard(event.keysym.lower())

    def _set_camera_image(self, frame: np.ndarray) -> None:
        from PIL import Image, ImageTk

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize((CAMERA_WIDTH, CAMERA_HEIGHT), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=image)
        self.camera_label.configure(image=photo)
        self.camera_label.image = photo

    def start_camera(self) -> None:
        if self.camera_thread is not None and self.camera_thread.is_alive():
            self.message = "Camera is already running"
            self.message_until = time.perf_counter() + 2.0
            return
        self.camera_stop.clear()
        self.reset_gesture_event.clear()
        with self.packet_lock:
            self.latest_packet = None
        self.active_gesture = empty_gesture("LOADING")
        self.camera_error = None
        self.camera_status = "Loading YOLO model..."
        self.camera_thread = threading.Thread(target=self._camera_loop, name="mario-pose-camera", daemon=True)
        self.camera_thread.start()

    def stop_camera(self) -> None:
        self.camera_stop.set()
        self.camera_status = "Camera stopping..."
        self.active_gesture = empty_gesture()
        with self.packet_lock:
            self.latest_packet = None

    def reset_gestures(self) -> None:
        if self.camera_thread is None or not self.camera_thread.is_alive():
            self.start_camera()
        self.reset_gesture_event.set()
        self.message = "Gesture state reset — return hands to neutral"
        self.message_until = time.perf_counter() + 2.5

    def _camera_loop(self) -> None:
        from ultralytics import YOLO

        controller = GestureController(confirmation_frames=self.args.gesture_confirmation_frames)
        tracker = PrimaryDancerTracker(
            keypoint_threshold=self.args.keypoint_confidence,
            smoothing_alpha=0.65,
        )
        try:
            model = YOLO(str(self.args.model))
            capture = open_camera(self.args.camera, self.args.camera_width, self.args.camera_height)
            self.camera_status = "Show shoulders and hands"
        except Exception as exc:
            self.camera_error = str(exc)
            return

        generation = 0
        last_state = empty_gesture("SHOW HANDS")
        try:
            while not self.camera_stop.is_set():
                if self.reset_gesture_event.is_set():
                    controller.reset()
                    tracker.reset()
                    self.reset_gesture_event.clear()
                    last_state = empty_gesture("GESTURES RESET")
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Camera stopped returning frames.")
                frame = cv2.flip(frame, 1)
                started = time.perf_counter()
                result = model.predict(
                    frame, conf=self.args.confidence, imgsz=self.args.image_size,
                    device=self.args.device, verbose=False,
                )[0]
                inference_ms = (time.perf_counter() - started) * 1000.0
                detections = extract_detections(result)
                chosen, points, valid, _ = tracker.select(detections, frame.shape)
                annotated = frame.copy()
                draw_other_people(annotated, detections, chosen)
                if chosen is not None and points is not None and valid is not None:
                    pose_confidence = float(np.mean(chosen.keypoint_confidence[valid])) if np.any(valid) else 0.0
                    draw_pose(annotated, points, valid, chosen.box, pose_confidence)
                    last_state = controller.update(points, valid, time.perf_counter())
                else:
                    last_state = empty_gesture("HANDS NOT FOUND")
                    cv2.putText(annotated, "SHOW SHOULDERS AND HANDS", (18, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (60, 60, 255), 2, cv2.LINE_AA)

                self._decorate_camera(annotated, last_state, inference_ms)
                generation += 1
                packet = CameraPacket(generation, annotated, last_state, inference_ms, time.perf_counter())
                with self.packet_lock:
                    self.latest_packet = packet
                self.camera_status = "Camera active"
        except Exception as exc:
            self.camera_error = f"Camera pose control failed: {exc}"
        finally:
            capture.release()
            self.camera_status = "Camera off"

    @staticmethod
    def _decorate_camera(frame: np.ndarray, state: GestureState, inference_ms: float) -> None:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 78), (17, 20, 33), -1)
        colour = (70, 235, 120) if state.ready else (0, 210, 255)
        cv2.putText(frame, state.label, (16, 36), cv2.FONT_HERSHEY_DUPLEX, 1.0, colour, 2, cv2.LINE_AA)
        gap = "--" if not np.isfinite(state.wrist_gap) else f"{state.wrist_gap:.2f}"
        cv2.putText(
            frame,
            f"L {state.left_score:.2f}  R {state.right_score:.2f}  gap {gap}  {inference_ms:.0f} ms",
            (16, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (235, 235, 235), 1, cv2.LINE_AA,
        )

    def restart_level(self) -> None:
        # Reassert the fixed viewport in case Windows DPI/layout negotiation
        # changed a requested widget size before the restart click.
        self.canvas.configure(
            width=GAME_WIDTH,
            height=GAME_HEIGHT,
            scrollregion=(0, 0, GAME_WIDTH, GAME_HEIGHT),
        )
        self.player = Player()
        self.score = 0
        self.coin_count = 0
        self.lives = MAX_LIVES
        self.finished = False
        self.started_at = time.perf_counter()
        self.camera_x = 0.0
        self.checkpoint_x = self.player.x
        self.checkpoint_y = self.player.y
        self.checkpoint_activated = False
        self.input_frozen_until = 0.0
        self.jump_buffer_until = 0.0
        self.buffered_jump_height = 0.22
        self._build_level()
        self.message = "Level restarted"
        self.message_until = time.perf_counter() + 2.0

    def _request_jump(self, physical_height: float = 0.22) -> None:
        now = time.perf_counter()
        if self.finished or now < self.input_frozen_until:
            return
        if self.player.grounded:
            self.jump_buffer_until = 0.0
            self._launch_jump(physical_height)
        else:
            # Keep a jump requested just before landing instead of discarding
            # it. This removes the short dead period between consecutive jumps.
            self.jump_buffer_until = now + JUMP_BUFFER_SECONDS
            self.buffered_jump_height = physical_height

    def _launch_jump(self, physical_height: float) -> None:
        # The base jump reaches about 150 px. A stronger real jump can add up
        # to another 100 px/s of launch speed (about 50 px more height).
        strength = float(np.clip((physical_height - 0.22) / 0.35, 0.0, 1.0))
        self.player.vy = -(620.0 + 100.0 * strength)
        self.player.grounded = False
        self.score += 10

    def _consume_jump_buffer(self, now: float) -> bool:
        if self.jump_buffer_until <= 0.0:
            return False
        if now > self.jump_buffer_until:
            self.jump_buffer_until = 0.0
            return False
        if not self.player.grounded:
            return False
        height = self.buffered_jump_height
        self.jump_buffer_until = 0.0
        self._launch_jump(height)
        return True

    @staticmethod
    def _overlap(
        first_x: float, first_y: float, first_w: float, first_h: float,
        second_x: float, second_y: float, second_w: float, second_h: float,
    ) -> bool:
        return (
            first_x < second_x + second_w and first_x + first_w > second_x
            and first_y < second_y + second_h and first_y + first_h > second_y
        )

    def _update_facing(self, horizontal: float) -> None:
        if horizontal < -0.05:
            self.player.facing = -1
        elif horizontal > 0.05:
            self.player.facing = 1

    def _solid_surfaces(
        self, include_one_way: bool = True,
    ) -> list[tuple[float, float, float, float, Optional[Block]]]:
        # Platforms one tile thick are one-way: they catch a falling player but
        # never block upward or sideways movement. Ground, stairs, pipes and
        # full blocks remain solid in every direction.
        surfaces = [
            (*platform, None) for platform in self.platforms
            if include_one_way or platform[3] > 22.0
        ]
        surfaces.extend((pipe.x, pipe.y, pipe.width, pipe.height, None) for pipe in self.pipes)
        surfaces.extend(
            (block.x, block.y, 40.0, 40.0, block)
            for block in self.blocks if block.active
        )
        if include_one_way:
            surfaces.extend(
                (platform.x, platform.y, platform.width, platform.height, None)
                for platform in self.moving_platforms
            )
        return surfaces

    def _update_moving_platforms(self, dt: float) -> None:
        player_bottom = self.player.y + self.player.height
        for platform in self.moving_platforms:
            previous_x = platform.x
            platform.x += platform.direction * platform.speed * dt
            if platform.x <= platform.left_bound or platform.x >= platform.right_bound:
                platform.x = float(np.clip(platform.x, platform.left_bound, platform.right_bound))
                platform.direction *= -1.0
            platform.delta_x = platform.x - previous_x
            standing_on_platform = (
                self.player.grounded
                and abs(player_bottom - platform.y) <= 5.0
                and self.player.x + self.player.width > previous_x
                and self.player.x < previous_x + platform.width
            )
            if standing_on_platform:
                self.player.x += platform.delta_x

    def _move_player_horizontal(self, distance: float) -> None:
        if abs(distance) < 1e-6:
            return
        previous_x = self.player.x
        target_x = float(np.clip(previous_x + distance, 0.0, LEVEL_WIDTH - self.player.width))
        for solid_x, solid_y, solid_w, solid_h, _ in self._solid_surfaces(include_one_way=False):
            vertical_overlap = (
                self.player.y < solid_y + solid_h - 1.0
                and self.player.y + self.player.height > solid_y + 1.0
            )
            if not vertical_overlap:
                continue
            if distance > 0.0:
                previous_right = previous_x + self.player.width
                target_right = target_x + self.player.width
                if previous_right <= solid_x + 3.0 and target_right > solid_x:
                    target_x = min(target_x, solid_x - self.player.width)
            else:
                previous_left = previous_x
                if previous_left >= solid_x + solid_w - 3.0 and target_x < solid_x + solid_w:
                    target_x = max(target_x, solid_x + solid_w)
        self.player.x = target_x

    def _activate_block(self, block: Block) -> None:
        if not block.active:
            return
        self._play_sound("bump")
        if block.used:
            return
        now = time.perf_counter()
        if block.kind == "brick":
            if self.player.powered:
                block.active = False
                self.score += 50
                self.message = "BRICK BREAK!"
                self._play_sound("stomp")
            else:
                self.message = "Find a power mushroom to break bricks"
            self.message_until = now + 1.4
            return
        block.used = True
        if block.contents == "mushroom":
            self.powerups.append(PowerUp(block.x + 7.0, block.y - 29.0))
            self.message = "POWER MUSHROOM APPEARED!"
        else:
            self.coin_count += 1
            self.score += 200
            self.message = "QUESTION BLOCK +200"
            self._play_sound("coin")
        self.message_until = now + 1.5

    def _update_player_vertical(self, dt: float) -> float:
        previous_top = self.player.y
        previous_bottom = self.player.y + self.player.height
        self.player.vy = min(self.player.vy + 1280.0 * dt, 750.0)
        self.player.y += self.player.vy * dt
        landing_surfaces = self._solid_surfaces(include_one_way=True)
        self.player.grounded = False

        if self.player.vy < 0.0:
            collisions = []
            for solid_x, solid_y, solid_w, solid_h, block in self._solid_surfaces(include_one_way=False):
                horizontal_overlap = (
                    self.player.x + self.player.width > solid_x
                    and self.player.x < solid_x + solid_w
                )
                underside = solid_y + solid_h
                if horizontal_overlap and previous_top >= underside - 3.0 and self.player.y <= underside:
                    collisions.append((underside, block))
            if collisions:
                underside, block = max(collisions, key=lambda collision: collision[0])
                if block is not None:
                    self._activate_block(block)
                if block is None or block.active:
                    self.player.y = underside
                    self.player.vy = 35.0
            return previous_bottom

        landing_candidates = []
        new_bottom = self.player.y + self.player.height
        for solid_x, solid_y, solid_w, solid_h, _ in landing_surfaces:
            horizontal_overlap = (
                self.player.x + self.player.width > solid_x
                and self.player.x < solid_x + solid_w
            )
            if horizontal_overlap and previous_bottom <= solid_y + 5.0 and new_bottom >= solid_y:
                landing_candidates.append(solid_y)
        if landing_candidates:
            landing_y = min(landing_candidates)
            self.player.y = landing_y - self.player.height
            self.player.vy = 0.0
            self.player.grounded = True
        return previous_bottom

    def _collect_powerups(self, now: float) -> None:
        for powerup in self.powerups:
            if powerup.collected:
                continue
            if self._overlap(
                self.player.x, self.player.y, self.player.width, self.player.height,
                powerup.x, powerup.y, 26.0, 26.0,
            ):
                powerup.collected = True
                self.player.powered = True
                self.player.invincible_until = now + 0.45
                self.score += 1000
                self.message = "POWER UP! One hit is now protected"
                self.message_until = now + 2.0
                self._play_sound("coin")

    def _hurt_player(self, reason: str, now: float) -> None:
        if self.player.powered:
            self.player.powered = False
            self.player.invincible_until = now + 1.4
            self.player.vx = -self.player.facing * 120.0
            self.message = "POWER LOST - you are still alive"
            self.message_until = now + 1.8
            self._play_sound("hurt")
        else:
            self._lose_life(reason)

    def _activate_checkpoint(self, now: float) -> bool:
        if self.checkpoint_activated or self.player.x < self.checkpoint_marker_x:
            return False
        self.checkpoint_activated = True
        self.checkpoint_x = 2020.0
        self.checkpoint_y = GROUND_Y - self.player.height
        self.score += 500
        self.message = "CHECKPOINT SAVED!"
        self.message_until = now + 2.0
        self._play_sound("checkpoint")
        return True

    def _update_game(self, dt: float) -> None:
        if self.finished:
            return
        now = time.perf_counter()
        self._consume_jump_buffer(now)
        self._update_moving_platforms(dt)
        controls_frozen = now < self.input_frozen_until
        gesture_horizontal = self.active_gesture.horizontal if self.active_gesture.ready else 0.0
        keyboard_horizontal = 0.0
        if self.keys.intersection({"a", "left"}):
            keyboard_horizontal -= 1.0
        if self.keys.intersection({"d", "right"}):
            keyboard_horizontal += 1.0
        horizontal = keyboard_horizontal if keyboard_horizontal else gesture_horizontal
        if controls_frozen:
            horizontal = 0.0
        self._update_facing(horizontal)
        self.player.crouching = (
            not controls_frozen
            and (self.active_gesture.crouching or bool(self.keys.intersection({"s", "down"})))
        )
        speed = 150.0 if self.player.crouching else 245.0
        target_vx = horizontal * speed
        if self.player.grounded:
            self.player.vx = target_vx
        elif abs(horizontal) > 0.05:
            # Air control is gradual and preserves momentum from the run-up.
            self.player.vx += (target_vx - self.player.vx) * min(1.0, dt * 6.0)
        else:
            self.player.vx *= max(0.0, 1.0 - dt * 0.35)
        self._move_player_horizontal(self.player.vx * dt)
        previous_bottom = self._update_player_vertical(dt)

        # A request received during the last fraction of the fall is executed
        # immediately on the landing frame.
        self._consume_jump_buffer(now)

        self._activate_checkpoint(now)

        self._collect_powerups(now)

        for coin in self.coins:
            if not coin.collected and self._overlap(self.player.x, self.player.y, self.player.width, self.player.height, coin.x - 11, coin.y - 11, 22, 22):
                coin.collected = True
                self.coin_count += 1
                self.score += 100
                self._play_sound("coin")

        for enemy in self.enemies:
            if not enemy.alive:
                continue
            enemy.x += enemy.direction * 65.0 * dt
            if enemy.x <= enemy.left_bound or enemy.x >= enemy.right_bound:
                enemy.direction *= -1.0
                enemy.x = float(np.clip(enemy.x, enemy.left_bound, enemy.right_bound))
            if self._overlap(self.player.x, self.player.y, self.player.width, self.player.height, enemy.x, enemy.y, 34, 34):
                if self.player.vy > 80.0 and previous_bottom <= enemy.y + 13:
                    enemy.alive = False
                    self.player.vy = -300.0
                    self.score += 250
                    self._play_sound("stomp")
                elif now >= self.player.invincible_until:
                    self._hurt_player("Hit an enemy", now)
                    break

        if self.player.y > GAME_HEIGHT + 120:
            self._lose_life("Missed the platform")
        if self.player.x >= self.goal_x:
            self.finished = True
            elapsed = time.perf_counter() - self.started_at
            time_bonus = max(0, int(8000 - elapsed * 80))
            self.score += time_bonus
            self.message = f"LEVEL COMPLETE! Time bonus +{time_bonus}"
            self.message_until = math.inf
            self._play_sound("win")

        target_camera = float(np.clip(
            self.player.x - GAME_WIDTH * 0.38,
            0.0,
            LEVEL_WIDTH - GAME_WIDTH,
        ))
        self.camera_x += (target_camera - self.camera_x) * min(1.0, dt * 5.5)

    def _lose_life(self, reason: str) -> None:
        self.lives -= 1
        self.jump_buffer_until = 0.0
        self.player.powered = False
        self._play_sound("hurt")
        self.message = reason
        self.message_until = time.perf_counter() + 2.0
        if self.lives <= 0:
            self.finished = True
            self.message = "GAME OVER — click Restart Level"
            self.message_until = math.inf
            return
        self.player.x = self.checkpoint_x
        self.player.y = self.checkpoint_y
        self.player.vx = self.player.vy = 0.0
        self.player.grounded = True
        now = time.perf_counter()
        self.player.invincible_until = now + 1.5
        self.input_frozen_until = now + 0.6
        self.camera_x = float(np.clip(self.checkpoint_x - GAME_WIDTH * 0.38, 0.0, 2020.0))
        self.message = f"{reason} — respawned at checkpoint"
        self.message_until = now + 2.0

    def _render_game(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        width = GAME_WIDTH
        height = GAME_HEIGHT
        background = self.images.get("background")
        if background is not None:
            canvas.create_image(GAME_WIDTH / 2, GAME_HEIGHT / 2, image=background)
        else:
            canvas.create_rectangle(0, 0, width, height, fill="#75c9ff", outline="")

        # Parallax sky decorations.
        if background is None:
            for cloud_x, cloud_y in ((120, 95), (560, 145), (1050, 85), (1690, 130), (2350, 90)):
                screen_x = cloud_x - self.camera_x * 0.25
                canvas.create_oval(screen_x, cloud_y, screen_x + 70, cloud_y + 35, fill="white", outline="")
                canvas.create_oval(screen_x + 35, cloud_y - 17, screen_x + 105, cloud_y + 36, fill="white", outline="")
                canvas.create_oval(screen_x + 75, cloud_y + 2, screen_x + 135, cloud_y + 36, fill="white", outline="")
            for hill_x in (50, 730, 1420, 2160):
                screen_x = hill_x - self.camera_x * 0.55
                canvas.create_polygon(screen_x, 480, screen_x + 170, 260, screen_x + 340, 480, fill="#42b96c", outline="#269555", width=3)

        for x, y, w, h in self.platforms:
            sx = x - self.camera_x
            if sx + w < -20 or sx > width + 20:
                continue
            tile_name = "ground_tile" if h >= 60 else "platform_tile"
            tile = self.images.get(tile_name)
            if tile is not None:
                tile_height = 32 if h >= 60 else 22
                for tile_y in np.arange(y, y + h, tile_height):
                    for tile_x in np.arange(sx, sx + w, 32):
                        canvas.create_image(tile_x, tile_y, image=tile, anchor="nw")
            else:
                canvas.create_rectangle(sx, y, sx + w, y + h, fill="#a96232", outline="#70401e", width=2)
                canvas.create_rectangle(sx, y, sx + w, y + min(h, 11), fill="#47b04b", outline="#22772d", width=2)

        for platform in self.moving_platforms:
            sx = platform.x - self.camera_x
            canvas.create_rectangle(
                sx, platform.y, sx + platform.width, platform.y + platform.height,
                fill="#f4a340", outline="#8f4d1d", width=3,
            )
            for bolt_x in (sx + 12, sx + platform.width - 12):
                canvas.create_oval(
                    bolt_x - 3, platform.y + 7, bolt_x + 3, platform.y + 13,
                    fill="#ffe6a7", outline="#805327",
                )

        for pipe in self.pipes:
            sx = pipe.x - self.camera_x
            if sx + pipe.width < -20 or sx > width + 20:
                continue
            canvas.create_rectangle(
                sx + 8, pipe.y + 13, sx + pipe.width - 8, pipe.y + pipe.height,
                fill="#2fbf5b", outline="#126c36", width=3,
            )
            canvas.create_rectangle(
                sx, pipe.y, sx + pipe.width, pipe.y + 20,
                fill="#45d66f", outline="#126c36", width=3,
            )
            canvas.create_line(sx + 18, pipe.y + 5, sx + 18, pipe.y + pipe.height, fill="#8bf09f", width=4)

        for block in self.blocks:
            if not block.active:
                continue
            sx = block.x - self.camera_x
            if block.used:
                fill, outline = "#8c806f", "#514a42"
            elif block.kind == "question":
                fill, outline = "#f5b83b", "#9b5b17"
            else:
                fill, outline = "#c76532", "#743316"
            canvas.create_rectangle(sx, block.y, sx + 40, block.y + 40, fill=fill, outline=outline, width=3)
            if block.kind == "question" and not block.used:
                canvas.create_text(sx + 20, block.y + 20, text="?", fill="#fff3bd", font=("Consolas", 23, "bold"))
            elif block.kind == "brick":
                canvas.create_line(sx, block.y + 20, sx + 40, block.y + 20, fill=outline, width=2)
                canvas.create_line(sx + 20, block.y, sx + 20, block.y + 20, fill=outline, width=2)
                canvas.create_line(sx + 10, block.y + 20, sx + 10, block.y + 40, fill=outline, width=2)

        checkpoint_sx = self.checkpoint_marker_x - self.camera_x
        checkpoint_colour = "#60e68a" if self.checkpoint_activated else "#d8dee9"
        canvas.create_rectangle(checkpoint_sx, 372, checkpoint_sx + 5, GROUND_Y, fill="#eceff4", outline="#687386")
        canvas.create_polygon(
            checkpoint_sx + 5, 378, checkpoint_sx + 58, 392, checkpoint_sx + 5, 410,
            fill=checkpoint_colour, outline="#245a46", width=2,
        )
        canvas.create_text(checkpoint_sx + 29, 430, text="CHECK", fill="#263044", font=("Consolas", 8, "bold"))

        for coin in self.coins:
            if coin.collected:
                continue
            sx = coin.x - self.camera_x
            coin_image = self.images.get("coin")
            if coin_image is not None:
                canvas.create_image(sx, coin.y, image=coin_image)
            else:
                canvas.create_oval(sx - 9, coin.y - 13, sx + 9, coin.y + 13, fill="#ffd43b", outline="#e38b00", width=3)

        for powerup in self.powerups:
            if powerup.collected:
                continue
            sx = powerup.x - self.camera_x
            canvas.create_rectangle(sx + 8, powerup.y + 14, sx + 19, powerup.y + 27, fill="#f8dcc0", outline="#7c4a2c", width=2)
            canvas.create_oval(sx, powerup.y, sx + 27, powerup.y + 20, fill="#ef4b45", outline="#84251f", width=2)
            canvas.create_oval(sx + 5, powerup.y + 4, sx + 11, powerup.y + 10, fill="#fff2d5", outline="")
            canvas.create_oval(sx + 17, powerup.y + 3, sx + 23, powerup.y + 9, fill="#fff2d5", outline="")

        for enemy in self.enemies:
            if not enemy.alive:
                continue
            sx = enemy.x - self.camera_x
            enemy_image = self.images.get("enemy")
            if enemy_image is not None:
                canvas.create_image(sx + 17, enemy.y + 17, image=enemy_image)
            else:
                canvas.create_oval(sx, enemy.y, sx + 34, enemy.y + 34, fill="#7a4fc6", outline="#30205e", width=2)

        goal_sx = self.goal_x - self.camera_x
        castle_x = self.goal_x + 75.0 - self.camera_x
        canvas.create_rectangle(castle_x, 365, castle_x + 105, GROUND_Y, fill="#d68a4c", outline="#713c28", width=3)
        canvas.create_rectangle(castle_x + 12, 335, castle_x + 38, 385, fill="#d68a4c", outline="#713c28", width=3)
        canvas.create_rectangle(castle_x + 67, 335, castle_x + 93, 385, fill="#d68a4c", outline="#713c28", width=3)
        canvas.create_arc(castle_x + 38, 420, castle_x + 68, 485, start=0, extent=180, fill="#34283a", outline="#713c28", width=3)
        for window_x in (castle_x + 22, castle_x + 78):
            canvas.create_rectangle(window_x, 392, window_x + 10, 410, fill="#65d6e8", outline="#713c28")
        flag_image = self.images.get("goal_flag")
        if flag_image is not None:
            canvas.create_image(goal_sx + 32, 416, image=flag_image)
        else:
            canvas.create_rectangle(goal_sx, 245, goal_sx + 7, 480, fill="#eeeeee", outline="#888888")
            canvas.create_polygon(goal_sx + 7, 255, goal_sx + 85, 277, goal_sx + 7, 303, fill="#20bfa9", outline="#075f62", width=2)

        self._draw_player()
        elapsed = time.perf_counter() - self.started_at
        power_label = "BIG" if self.player.powered else "SMALL"
        canvas.create_rectangle(12, 12, 535, 58, fill="#192033", outline="#65708f", width=2)
        canvas.create_text(28, 35, anchor="w", text=f"LEVEL 1/1   SCORE {self.score:05d}   COINS {self.coin_count:02d}   LIFE {self.lives}   POWER {power_label}   TIME {elapsed:04.1f}", fill="white", font=("Consolas", 11, "bold"))
        if time.perf_counter() < self.message_until:
            canvas.create_rectangle(width / 2 - 265, 78, width / 2 + 265, 128, fill="#181b29", outline="#ffd166", width=2)
            canvas.create_text(width / 2, 103, text=self.message, fill="#ffd166", font=("Segoe UI", 13, "bold"))
        if self.finished:
            canvas.create_rectangle(width / 2 - 260, 165, width / 2 + 260, 355, fill="#151827", outline="#65e6ff", width=4)
            title = "LEVEL COMPLETE" if self.lives > 0 else "GAME OVER"
            canvas.create_text(width / 2, 215, text=title, fill="#ffd166", font=("Segoe UI", 26, "bold"))
            canvas.create_text(width / 2, 270, text=f"FINAL SCORE  {self.score:06d}", fill="white", font=("Consolas", 18, "bold"))
            canvas.create_text(width / 2, 320, text="Press R or click Restart Level", fill="#b8c5dc", font=("Segoe UI", 12))

    def _draw_player(self) -> None:
        canvas = self.canvas
        player = self.player
        sx = player.x - self.camera_x
        crouch_offset = 17 if player.crouching and player.grounded else 0
        sy = player.y + crouch_offset
        height = player.height - crouch_offset
        if time.perf_counter() < player.invincible_until and int(time.perf_counter() * 10) % 2 == 0:
            return
        if player.powered:
            canvas.create_oval(
                sx - 5, sy - 5, sx + player.width + 5, sy + height + 5,
                fill="", outline="#ffd84d", width=3,
            )
        if player.crouching and player.grounded:
            pose_name = "hero_crouch"
        elif not player.grounded:
            pose_name = "hero_jump"
        elif abs(player.vx) > 30:
            pose_name = "hero_run1" if int(time.perf_counter() * 8) % 2 == 0 else "hero_run2"
        else:
            pose_name = "hero_idle"
        direction_name = "left" if player.facing < 0 else "right"
        image_name = f"{pose_name}_{direction_name}"
        hero_image = self.images.get(image_name)
        if hero_image is not None:
            canvas.create_image(sx + player.width / 2, sy + height / 2, image=hero_image)
            return
        # Fallback hero if the generated PNG assets are unavailable.
        canvas.create_rectangle(sx + 8, sy + 20, sx + 31, sy + height - 9, fill="#2878d0", outline="#173f82", width=2)
        canvas.create_rectangle(sx + 3, sy + 21, sx + 12, sy + height - 18, fill="#e9463f", outline="#8d211e")
        canvas.create_rectangle(sx + 28, sy + 21, sx + 37, sy + height - 18, fill="#e9463f", outline="#8d211e")
        canvas.create_oval(sx + 9, sy + 6, sx + 32, sy + 30, fill="#f1b486", outline="#8c5437", width=2)
        canvas.create_rectangle(sx + 6, sy + 2, sx + 33, sy + 10, fill="#e53935", outline="#8d211e", width=2)
        canvas.create_rectangle(sx + 15, sy, sx + 31, sy + 6, fill="#e53935", outline="#8d211e")
        canvas.create_oval(sx + 25, sy + 14, sx + 29, sy + 18, fill="#212121", outline="")
        canvas.create_rectangle(sx + 8, sy + height - 10, sx + 18, sy + height, fill="#5d321f", outline="#2f170d")
        canvas.create_rectangle(sx + 24, sy + height - 10, sx + 36, sy + height, fill="#5d321f", outline="#2f170d")

    def _consume_camera_packet(self) -> None:
        if self.camera_stop.is_set():
            self.active_gesture = empty_gesture()
            self.gesture_var.set("Gesture: CAMERA OFF")
            return
        with self.packet_lock:
            packet = self.latest_packet
        if packet is None:
            self.gesture_var.set(f"Gesture: {self.camera_status.upper()}")
            return
        self.active_gesture = packet.gesture
        if packet.generation != self.last_packet_generation:
            self.last_packet_generation = packet.generation
            if packet.gesture.jump:
                self._request_jump(0.50)
        self._set_camera_image(packet.frame)
        self.gesture_var.set(f"Gesture: {packet.gesture.label}")
        gap = "--" if not np.isfinite(packet.gesture.wrist_gap) else f"{packet.gesture.wrist_gap:.2f}"
        if packet.gesture.ready:
            self.metrics_var.set(
                f"left score {packet.gesture.left_score:.2f}   right score {packet.gesture.right_score:.2f}   both up {int(packet.gesture.both_up)}\n"
                f"wrist gap {gap}   visible upper body {packet.gesture.coverage * 100:.0f}%   inference {packet.inference_ms:.0f} ms"
            )
        else:
            self.metrics_var.set(
                "Keep both shoulders and at least one wrist visible.\n"
                "Lower-body keypoints are not required."
            )

    def _tick(self) -> None:
        if self.camera_error:
            error = self.camera_error
            self.camera_error = None
            self.stop_camera()
            self.messagebox.showerror("Camera error", error)
        self._consume_camera_packet()
        now = time.perf_counter()
        dt = min(now - self.last_tick, 0.04)
        self.last_tick = now
        self._update_game(dt)
        self._render_game()
        self.hud_var.set(f"{self.active_gesture.label:>11}  |  CAMERA + KEYBOARD")
        if self.root.winfo_exists():
            self.root.after(16, self._tick)

    def close(self) -> None:
        self.camera_stop.set()
        self.sounds.stop()
        self.root.destroy()


def validate_demo(args: argparse.Namespace) -> int:
    from ultralytics import YOLO

    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    required_assets = (
        "pixel_landscape.png",
        "hero_idle_left.png", "hero_idle_right.png",
        "hero_run1_left.png", "hero_run1_right.png",
        "hero_run2_left.png", "hero_run2_right.png",
        "hero_jump_left.png", "hero_jump_right.png",
        "hero_crouch_left.png", "hero_crouch_right.png",
        "enemy.png", "coin.png",
        "ground_tile.png", "platform_tile.png", "goal_flag.png",
        "coin.wav", "bump.wav", "stomp.wav", "hurt.wav", "checkpoint.wav", "win.wav",
    )
    missing_assets = [name for name in required_assets if not (ASSET_DIR / name).is_file()]
    if missing_assets:
        raise FileNotFoundError(f"Missing demo assets: {', '.join(missing_assets)}")
    video = cv2.VideoCapture(str(SAMPLE_VIDEO))
    if not video.isOpened():
        raise RuntimeError(f"Could not open validation video: {SAMPLE_VIDEO}")
    video.set(cv2.CAP_PROP_POS_FRAMES, 300)
    ok, frame = video.read()
    video.release()
    if not ok:
        raise RuntimeError("Could not decode validation frame.")
    model = YOLO(str(args.model))
    result = model.predict(
        frame, conf=args.confidence, imgsz=args.image_size,
        device=args.device, verbose=False,
    )[0]
    detections = extract_detections(result)
    tracker = PrimaryDancerTracker(keypoint_threshold=args.keypoint_confidence)
    _, points, valid, _ = tracker.select(detections, frame.shape)
    if points is None or valid is None:
        raise RuntimeError("YOLO did not find a scoreable person in the validation frame.")
    controller = GestureController(confirmation_frames=args.gesture_confirmation_frames)
    state = controller.update(points, valid, 0.0)
    print("Mario camera demo validation passed")
    print(f"  detected people: {len(detections)}")
    print(f"  visible keypoints: {int(np.count_nonzero(valid))}/17")
    print(f"  upper-body hand control: {'ready' if state.ready else 'incomplete'}")
    print(f"  original visual/audio assets: {len(required_assets)} files")
    print(f"  model: {args.model.name}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Camera-controlled Mario-style platform game demo")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--device", default="cpu", help="Ultralytics device: cpu, 0, 1, ...")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=960)
    parser.add_argument("--camera-height", type=int, default=540)
    parser.add_argument("--image-size", type=int, default=320, help="YOLO input size; 320 is recommended on CPU")
    parser.add_argument("--confidence", type=float, default=0.30)
    parser.add_argument(
        "--keypoint-confidence",
        type=float,
        default=0.35,
        help="Minimum pose-keypoint confidence; 0.35 keeps noisy wrists usable",
    )
    parser.add_argument(
        "--gesture-confirmation-frames",
        type=int,
        default=2,
        help="Consecutive gesture frames required; 2 favors responsive play",
    )
    parser.add_argument("--check", action="store_true", help="Validate model and gesture pipeline without opening the GUI")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.check:
            return validate_demo(args)
        import tkinter as tk

        root = tk.Tk()
        CameraPlatformDemo(root, args)
        root.mainloop()
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
