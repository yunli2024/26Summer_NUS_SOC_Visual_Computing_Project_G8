"""Side-by-side video tester for the Bonus Task 2 pose scorer.

Both panels play the same cached reference clip.  The simulated player panel
can be delayed and/or mirrored, while the scorer's lag and mirror tolerance
can be changed independently.  This makes the effects of those two scoring
features directly observable without a camera or additional YOLO inference.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from dance_scoring import (
    MatchResult,
    best_reference_match,
    feedback_for_score,
    mirror_pose,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_REFERENCE = BASE_DIR / "task2_results" / "dance_example_1" / "annotated.mp4"
DEFAULT_CACHE = BASE_DIR / "task2_results" / "dance_example_1" / "pose_cache.npz"
DISPLAY_SIZE = (600, 338)


def load_pose_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Pose cache not found: {path}")
    with np.load(path, allow_pickle=False) as cache:
        required = {"points", "valid", "playback_fps"}
        missing = required.difference(cache.files)
        if missing:
            raise ValueError(f"Pose cache is missing: {', '.join(sorted(missing))}")
        points = cache["points"].astype(np.float32)
        valid = cache["valid"].astype(bool)
        fps = float(np.asarray(cache["playback_fps"]).reshape(()))
    if points.ndim != 3 or points.shape[1:] != (17, 2):
        raise ValueError("Expected cached points with shape (frames, 17, 2).")
    if valid.shape != points.shape[:2] or fps <= 0:
        raise ValueError("Invalid pose-cache validity mask or playback FPS.")
    return {"points": points, "valid": valid, "fps": fps}


def read_video_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read video frame {index}.")
    return frame


class ScoringVideoTester:
    def __init__(self, root, args: argparse.Namespace) -> None:
        import tkinter as tk
        from tkinter import messagebox

        self.tk = tk
        self.messagebox = messagebox
        self.root = root
        self.args = args
        self.root.title("Pose Scoring A/B Tester")
        self.root.geometry("1260x760")
        self.root.minsize(1050, 680)
        self.root.configure(bg="#171923")

        cache = load_pose_cache(args.cache)
        self.points = cache["points"]
        self.valid = cache["valid"]
        self.fps = float(cache["fps"])
        self.reference_cap = cv2.VideoCapture(str(args.reference))
        self.player_cap = cv2.VideoCapture(str(args.reference))
        if not self.reference_cap.isOpened() or not self.player_cap.isOpened():
            raise RuntimeError(f"Could not open reference video: {args.reference}")
        video_frames = int(self.reference_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = min(video_frames, len(self.points))
        if self.frame_count <= 0:
            raise RuntimeError("Reference video and cache contain no matching frames.")

        self.running = False
        self.paused = False
        self.start_time = 0.0
        self.pause_started = 0.0
        self.paused_duration = 0.0
        self.last_reference_index = -1
        self.latest_match: MatchResult | None = None

        self.player_delay = tk.DoubleVar(value=args.player_delay)
        self.allowed_lag = tk.DoubleVar(value=args.max_lag)
        self.mirror_player = tk.BooleanVar(value=args.mirror_player)
        self.allow_mirror = tk.BooleanVar(value=not args.no_mirror)
        self.status_var = tk.StringVar(value="Ready - press Start")
        self.result_var = tk.StringVar(value="SIMILARITY  --")
        self.detail_var = tk.StringVar(
            value=f"Reference cache: {self.fps:.1f} FPS, {self.frame_count} frames"
        )

        self._build_ui()
        self._show_frame(0)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.after(25, self._tick)

    def _build_ui(self) -> None:
        tk = self.tk
        header = tk.Frame(self.root, bg="#171923")
        header.pack(fill=tk.X, padx=18, pady=(14, 8))
        tk.Label(
            header,
            text="POSE SCORING  A/B TESTER",
            font=("Segoe UI", 20, "bold"),
            fg="#65e6ff",
            bg="#171923",
        ).pack(side=tk.LEFT)
        tk.Label(
            header,
            textvariable=self.status_var,
            font=("Segoe UI", 11),
            fg="#f0f0f0",
            bg="#171923",
        ).pack(side=tk.RIGHT)

        panels = tk.Frame(self.root, bg="#171923")
        panels.pack(fill=tk.BOTH, expand=True, padx=18)
        left = tk.Frame(panels, bg="#232638")
        right = tk.Frame(panels, bg="#232638")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 7))
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(7, 0))
        tk.Label(
            left,
            text="REFERENCE (CURRENT TIME)",
            font=("Segoe UI", 12, "bold"),
            fg="#ffd166",
            bg="#232638",
        ).pack(pady=8)
        tk.Label(
            right,
            text="SIMULATED PLAYER (SAME VIDEO)",
            font=("Segoe UI", 12, "bold"),
            fg="#80ed99",
            bg="#232638",
        ).pack(pady=8)
        self.reference_label = tk.Label(left, bg="#090a10")
        self.player_label = tk.Label(right, bg="#090a10")
        self.reference_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.player_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        score_row = tk.Frame(self.root, bg="#171923")
        score_row.pack(fill=tk.X, padx=18, pady=(8, 3))
        tk.Label(
            score_row,
            textvariable=self.result_var,
            font=("Consolas", 17, "bold"),
            fg="#65e6ff",
            bg="#171923",
        ).pack(side=tk.LEFT)
        tk.Label(
            score_row,
            textvariable=self.detail_var,
            font=("Consolas", 11),
            fg="#f0f0f0",
            bg="#171923",
        ).pack(side=tk.RIGHT)

        controls = tk.Frame(self.root, bg="#171923")
        controls.pack(fill=tk.X, padx=18, pady=(5, 5))
        button_style = {
            "font": ("Segoe UI", 10, "bold"),
            "width": 12,
            "bd": 0,
            "padx": 5,
            "pady": 7,
        }
        tk.Button(
            controls,
            text="Start / Restart",
            command=self.start,
            bg="#65e6ff",
            fg="#11131b",
            **button_style,
        ).pack(side=tk.LEFT, padx=(0, 7))
        self.pause_button = tk.Button(
            controls,
            text="Pause",
            command=self.toggle_pause,
            bg="#ffd166",
            fg="#11131b",
            **button_style,
        )
        self.pause_button.pack(side=tk.LEFT, padx=7)
        tk.Checkbutton(
            controls,
            text="Mirror player video",
            variable=self.mirror_player,
            font=("Segoe UI", 10),
            fg="#f0f0f0",
            bg="#171923",
            activebackground="#171923",
            activeforeground="#f0f0f0",
            selectcolor="#232638",
            command=self._refresh_paused,
        ).pack(side=tk.LEFT, padx=(18, 8))
        tk.Checkbutton(
            controls,
            text="Accept mirrored moves",
            variable=self.allow_mirror,
            font=("Segoe UI", 10),
            fg="#f0f0f0",
            bg="#171923",
            activebackground="#171923",
            activeforeground="#f0f0f0",
            selectcolor="#232638",
            command=self._refresh_paused,
        ).pack(side=tk.LEFT, padx=8)
        tk.Button(
            controls,
            text="Quit",
            command=self.close,
            bg="#454a63",
            fg="white",
            **button_style,
        ).pack(side=tk.RIGHT)

        sliders = tk.Frame(self.root, bg="#171923")
        sliders.pack(fill=tk.X, padx=18, pady=(0, 14))
        tk.Label(
            sliders,
            text="Player delay (seconds)",
            font=("Segoe UI", 10),
            fg="#f0f0f0",
            bg="#171923",
        ).grid(row=0, column=0, sticky="w")
        tk.Scale(
            sliders,
            variable=self.player_delay,
            from_=0.0,
            to=1.5,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            length=390,
            bg="#171923",
            fg="#f0f0f0",
            highlightthickness=0,
            troughcolor="#34384e",
            command=lambda _value: self._refresh_paused(),
        ).grid(row=0, column=1, sticky="ew", padx=(8, 30))
        tk.Label(
            sliders,
            text="Allowed reaction lag (seconds)",
            font=("Segoe UI", 10),
            fg="#f0f0f0",
            bg="#171923",
        ).grid(row=0, column=2, sticky="w")
        tk.Scale(
            sliders,
            variable=self.allowed_lag,
            from_=0.0,
            to=1.5,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            length=390,
            bg="#171923",
            fg="#f0f0f0",
            highlightthickness=0,
            troughcolor="#34384e",
            command=lambda _value: self._refresh_paused(),
        ).grid(row=0, column=3, sticky="ew", padx=(8, 0))
        sliders.columnconfigure(1, weight=1)
        sliders.columnconfigure(3, weight=1)

    def _set_image(self, label, frame: np.ndarray) -> None:
        from PIL import Image, ImageTk

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize(DISPLAY_SIZE, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image=image)
        label.configure(image=photo)
        label.image = photo

    def _score(self, reference_index: int, player_index: int) -> MatchResult | None:
        player_points = self.points[player_index]
        player_valid = self.valid[player_index]
        if self.mirror_player.get():
            player_points, player_valid = mirror_pose(player_points, player_valid)
        max_lag_frames = int(round(max(0.0, self.allowed_lag.get()) * self.fps))
        return best_reference_match(
            player_points,
            player_valid,
            self.points,
            self.valid,
            reference_index,
            max_lag_frames,
            allow_mirror=self.allow_mirror.get(),
        )

    def _show_frame(self, reference_index: int) -> None:
        delay_frames = int(round(max(0.0, self.player_delay.get()) * self.fps))
        player_index = max(0, reference_index - delay_frames)
        reference_frame = read_video_frame(self.reference_cap, reference_index)
        player_frame = read_video_frame(self.player_cap, player_index)
        if self.mirror_player.get():
            player_frame = cv2.flip(player_frame, 1)

        cv2.putText(
            reference_frame,
            f"frame {reference_index}",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            player_frame,
            f"frame {player_index}  delay {(reference_index - player_index) / self.fps:.1f}s",
            (18, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        self.latest_match = self._score(reference_index, player_index)
        if self.latest_match is None:
            self.result_var.set("SIMILARITY   0.0  NO POSE")
            self.detail_var.set("No scoreable common joints")
        else:
            label, _, _ = feedback_for_score(self.latest_match.score)
            matched_lag = self.latest_match.lag_frames / self.fps
            expected_lag = (reference_index - player_index) / self.fps
            mirror_text = "yes" if self.latest_match.mirrored else "no"
            self.result_var.set(
                f"SIMILARITY {self.latest_match.score:5.1f}  {label}"
            )
            self.detail_var.set(
                f"expected lag {expected_lag:.1f}s | matched lag {matched_lag:.1f}s | "
                f"mirror selected {mirror_text}"
            )

        self._set_image(self.reference_label, reference_frame)
        self._set_image(self.player_label, player_frame)
        self.last_reference_index = reference_index

    def start(self) -> None:
        self.running = True
        self.paused = False
        self.start_time = time.perf_counter()
        self.paused_duration = 0.0
        self.last_reference_index = -1
        self.pause_button.configure(text="Pause")
        self.status_var.set("Playing")

    def toggle_pause(self) -> None:
        if not self.running:
            return
        if self.paused:
            self.paused_duration += time.perf_counter() - self.pause_started
            self.paused = False
            self.pause_button.configure(text="Pause")
            self.status_var.set("Playing")
        else:
            self.paused = True
            self.pause_started = time.perf_counter()
            self.pause_button.configure(text="Resume")
            self.status_var.set("Paused - controls update the current frame")

    def _refresh_paused(self) -> None:
        if not self.running or self.paused:
            self._show_frame(max(0, self.last_reference_index))

    def _tick(self) -> None:
        try:
            if self.running and not self.paused:
                elapsed = time.perf_counter() - self.start_time - self.paused_duration
                reference_index = int(elapsed * self.fps)
                if reference_index >= self.frame_count:
                    self.running = False
                    reference_index = self.frame_count - 1
                    self.status_var.set("Finished - press Start to replay")
                if reference_index != self.last_reference_index:
                    self._show_frame(reference_index)
        except Exception as error:
            self.running = False
            self.messagebox.showerror("Scoring tester error", str(error))
        finally:
            self.root.after(25, self._tick)

    def close(self) -> None:
        self.running = False
        self.reference_cap.release()
        self.player_cap.release()
        self.root.destroy()


def run_check(args: argparse.Namespace) -> int:
    cache = load_pose_cache(args.cache)
    points = cache["points"]
    valid = cache["valid"]
    fps = float(cache["fps"])
    capture = cv2.VideoCapture(str(args.reference))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open reference video: {args.reference}")
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    count = min(video_frames, len(points))
    if count < 20:
        raise RuntimeError("Need at least 20 aligned video/cache frames for the check.")

    current = min(count - 1, max(20, count // 2))
    lag_frames = int(round(args.max_lag * fps))
    player_index = max(0, current - lag_frames)
    direct = best_reference_match(
        points[player_index],
        valid[player_index],
        points,
        valid,
        current,
        lag_frames,
        allow_mirror=False,
    )
    mirrored_points, mirrored_valid = mirror_pose(points[player_index], valid[player_index])
    mirrored = best_reference_match(
        mirrored_points,
        mirrored_valid,
        points,
        valid,
        current,
        lag_frames,
        allow_mirror=True,
    )
    if direct is None or mirrored is None:
        raise RuntimeError("The cache did not contain a scoreable check pose.")
    expected_lag = current - player_index
    if direct.lag_frames != expected_lag or direct.score < 99.9:
        raise RuntimeError("Delay check failed.")
    if mirrored.lag_frames != expected_lag or mirrored.score < 99.9 or not mirrored.mirrored:
        raise RuntimeError("Mirror check failed.")

    print(f"Video/cache frames: {count}")
    print(f"Playback FPS: {fps:.1f}")
    print(
        f"Delay check: requested {expected_lag / fps:.1f}s, "
        f"matched {direct.lag_frames / fps:.1f}s, score {direct.score:.2f}"
    )
    print(
        f"Mirror check: matched={mirrored.mirrored}, "
        f"lag {mirrored.lag_frames / fps:.1f}s, score {mirrored.score:.2f}"
    )
    print("Scoring video tester check passed.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--player-delay", type=float, default=0.5)
    parser.add_argument("--max-lag", type=float, default=0.8)
    parser.add_argument("--mirror-player", action="store_true")
    parser.add_argument("--no-mirror", action="store_true")
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.player_delay < 0.0 or args.max_lag < 0.0:
        raise ValueError("Delay values must not be negative.")
    if args.check:
        return run_check(args)

    import tkinter as tk

    root = tk.Tk()
    try:
        ScoringVideoTester(root, args)
        root.mainloop()
    except Exception:
        root.destroy()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
