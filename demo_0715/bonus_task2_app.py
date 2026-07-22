from __future__ import annotations

import argparse
import time
import tkinter as tk
from collections import deque
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Deque, Union

import cv2
import numpy as np
from PIL import Image, ImageTk

from pose_pipeline import PoseEstimator, PoseStreamTracker, draw_pose_overlay
from pose_scoring import PoseMatch, TemporalPoseMatcher, feedback_label


APP_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO = APP_DIR / "dance_example_1.mp4"
DEFAULT_MODEL = APP_DIR / "yolov8n-pose.pt"
VideoSource = Union[int, Path]


class BonusTask2App:
    def __init__(
        self,
        root: tk.Tk,
        reference_video: Path,
        user_source: VideoSource,
        model_path: Path,
        *,
        reference_start_frame: int = 0,
        user_start_frame: int = 0,
    ) -> None:
        self.root = root
        self.root.title("Bonus Task 2 - Dance Pose Scoring")
        self.root.geometry("1280x820")

        self.reference_video = reference_video
        self.user_source = user_source
        self.reference_start_frame = max(0, reference_start_frame)
        self.user_start_frame = max(0, user_start_frame)
        self.estimator = PoseEstimator(model_path)
        self.matcher = TemporalPoseMatcher(max_lag_frames=12, keypoint_conf=0.25)
        self.ref_tracker = PoseStreamTracker()
        self.user_tracker = PoseStreamTracker()

        self.ref_cap: cv2.VideoCapture | None = None
        self.user_cap: cv2.VideoCapture | None = None
        self.running = False
        self.ref_frame_index = self.reference_start_frame
        self.user_frame_index = self.user_start_frame
        self.combo = 0
        self.recent_scores: Deque[float] = deque(maxlen=5)
        self.display_width = 600
        self.display_height = 338
        self.last_combined_bgr = None
        self.last_update_time = time.perf_counter()

        self.show_video = tk.BooleanVar(value=True)
        self.main_only = tk.BooleanVar(value=True)
        self.loop_reference = tk.BooleanVar(value=True)
        self.mirror_user = tk.BooleanVar(value=isinstance(user_source, int))
        self.confidence = tk.DoubleVar(value=0.25)
        self.keypoint_confidence = tk.DoubleVar(value=0.20)
        self.lag_frames = tk.IntVar(value=12)

        self._build_ui()
        self._show_placeholders()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self) -> None:
        outer = tk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        title = tk.Label(outer, text="Dance Pose Matching (Bonus Task 2)", font=("Segoe UI", 15, "bold"))
        title.pack(anchor="w")

        panels = tk.Frame(outer)
        panels.pack(fill=tk.BOTH, expand=True, pady=(10, 8))

        ref_panel = tk.Frame(panels)
        ref_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        tk.Label(ref_panel, text="Reference", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.ref_label = tk.Label(ref_panel, bg="#111111", width=self.display_width, height=self.display_height)
        self.ref_label.pack(fill=tk.BOTH, expand=True)

        user_panel = tk.Frame(panels)
        user_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        tk.Label(user_panel, text="User / Webcam", font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self.user_label = tk.Label(user_panel, bg="#111111", width=self.display_width, height=self.display_height)
        self.user_label.pack(fill=tk.BOTH, expand=True)

        controls = tk.Frame(outer)
        controls.pack(fill=tk.X)
        tk.Button(controls, text="Open Reference", command=self.open_reference).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="Open User Video", command=self.open_user_video).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="Use Webcam", command=self.use_webcam).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="Start", command=self.start).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="Restart", command=self.restart).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="Save Snapshot", command=self.save_snapshot).pack(side=tk.LEFT, padx=4)

        toggles = tk.Frame(outer)
        toggles.pack(fill=tk.X, pady=(8, 4))
        tk.Checkbutton(toggles, text="Show video", variable=self.show_video).pack(side=tk.LEFT, padx=4)
        tk.Checkbutton(toggles, text="Main dancer only", variable=self.main_only).pack(side=tk.LEFT, padx=4)
        tk.Checkbutton(toggles, text="Loop reference", variable=self.loop_reference).pack(side=tk.LEFT, padx=4)
        tk.Checkbutton(toggles, text="Mirror user", variable=self.mirror_user).pack(side=tk.LEFT, padx=4)

        sliders = tk.Frame(outer)
        sliders.pack(fill=tk.X, pady=(4, 4))
        tk.Label(sliders, text="Detector conf").pack(side=tk.LEFT)
        tk.Scale(sliders, variable=self.confidence, from_=0.1, to=0.7, resolution=0.05, orient=tk.HORIZONTAL, length=170).pack(side=tk.LEFT)
        tk.Label(sliders, text="Keypoint conf").pack(side=tk.LEFT, padx=(14, 0))
        tk.Scale(sliders, variable=self.keypoint_confidence, from_=0.05, to=0.7, resolution=0.05, orient=tk.HORIZONTAL, length=170).pack(side=tk.LEFT)
        tk.Label(sliders, text="Lag frames").pack(side=tk.LEFT, padx=(14, 0))
        tk.Scale(sliders, variable=self.lag_frames, from_=0, to=45, resolution=1, orient=tk.HORIZONTAL, length=170, command=self._update_lag_window).pack(side=tk.LEFT)

        self.score_text = tk.StringVar(value="score=0.0 | feedback=Waiting")
        tk.Label(outer, textvariable=self.score_text, anchor="w", justify=tk.LEFT, font=("Consolas", 13, "bold")).pack(fill=tk.X, pady=(8, 0))

        self.status_text = tk.StringVar(value=self._source_status())
        tk.Label(outer, textvariable=self.status_text, anchor="w", justify=tk.LEFT, font=("Consolas", 10)).pack(fill=tk.X, pady=(4, 0))

    def _show_placeholders(self) -> None:
        for label in (self.ref_label, self.user_label):
            placeholder = Image.new("RGB", (self.display_width, self.display_height), (28, 28, 28))
            imgtk = ImageTk.PhotoImage(placeholder)
            label.imgtk = imgtk
            label.configure(image=imgtk)

    def open_reference(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(APP_DIR),
            filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")],
        )
        if path:
            self.reference_video = Path(path)
            self.restart_if_running()

    def open_user_video(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(APP_DIR),
            filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")],
        )
        if path:
            self.user_source = Path(path)
            self.mirror_user.set(False)
            self.restart_if_running()

    def use_webcam(self) -> None:
        self.user_source = 0
        self.mirror_user.set(True)
        self.restart_if_running()

    def restart_if_running(self) -> None:
        was_running = self.running
        self.stop()
        self.status_text.set(self._source_status())
        if was_running:
            self.start()

    def start(self) -> None:
        if not self.reference_video.exists():
            messagebox.showerror("Missing reference", f"Reference video not found:\n{self.reference_video}")
            return
        if self.running:
            return

        self.ref_cap = cv2.VideoCapture(str(self.reference_video))
        self.user_cap = cv2.VideoCapture(self._opencv_source(self.user_source))
        if not self.ref_cap.isOpened():
            messagebox.showerror("Open failed", f"Cannot open reference video:\n{self.reference_video}")
            self.stop()
            return
        if not self.user_cap.isOpened():
            messagebox.showerror("Open failed", f"Cannot open user source:\n{self._source_name(self.user_source)}")
            self.stop()
            return
        if self.reference_start_frame > 0:
            self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, self.reference_start_frame)
        if isinstance(self.user_source, Path) and self.user_start_frame > 0:
            self.user_cap.set(cv2.CAP_PROP_POS_FRAMES, self.user_start_frame)
        if isinstance(self.user_source, int):
            self.user_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.user_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        self.running = True
        self.ref_frame_index = self.reference_start_frame
        self.user_frame_index = self.user_start_frame
        self.matcher.reset()
        self.matcher.set_max_lag_frames(int(self.lag_frames.get()))
        self.ref_tracker.reset()
        self.user_tracker.reset()
        self.combo = 0
        self.recent_scores.clear()
        self.last_update_time = time.perf_counter()
        self._process_next_pair()

    def stop(self) -> None:
        self.running = False
        for cap_name in ("ref_cap", "user_cap"):
            cap = getattr(self, cap_name)
            if cap is not None:
                cap.release()
                setattr(self, cap_name, None)

    def restart(self) -> None:
        self.stop()
        self.start()

    def close(self) -> None:
        self.stop()
        self.root.destroy()

    def save_snapshot(self) -> None:
        if self.last_combined_bgr is None:
            messagebox.showwarning("No frame", "Run the app first.")
            return
        output_dir = APP_DIR / "outputs"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"bonus_task2_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(output_path), self.last_combined_bgr)
        messagebox.showinfo("Snapshot saved", str(output_path))

    def _process_next_pair(self) -> None:
        if not self.running or self.ref_cap is None or self.user_cap is None:
            return

        ref_ok, ref_frame = self.ref_cap.read()
        if not ref_ok:
            if self.loop_reference.get():
                self.ref_cap.set(cv2.CAP_PROP_POS_FRAMES, self.reference_start_frame)
                self.ref_frame_index = self.reference_start_frame
                self.matcher.reset()
                self.root.after(1, self._process_next_pair)
                return
            self.stop()
            return

        user_ok, user_frame = self.user_cap.read()
        if not user_ok:
            if isinstance(self.user_source, Path):
                self.user_cap.set(cv2.CAP_PROP_POS_FRAMES, self.user_start_frame)
                self.user_frame_index = self.user_start_frame
                self.root.after(1, self._process_next_pair)
                return
            self.stop()
            return

        if self.mirror_user.get():
            user_frame = cv2.flip(user_frame, 1)

        keypoint_conf = float(self.keypoint_confidence.get())
        ref_result, user_result = self.estimator.infer_batch(
            [ref_frame, user_frame],
            frame_indices=[self.ref_frame_index, self.user_frame_index],
            conf=float(self.confidence.get()),
            keypoint_conf=keypoint_conf,
        )
        ref_result = self.ref_tracker.update(ref_result)
        user_result = self.user_tracker.update(user_result)

        self.matcher.keypoint_conf = keypoint_conf
        self.matcher.push_reference(ref_result)
        match = self.matcher.match_user(user_result)
        display_match = self._smooth_match(match)
        self.combo = self.combo + 1 if display_match.score >= 85.0 and display_match.common_keypoints >= 7 else 0

        ref_view, drawn_ref = draw_pose_overlay(
            ref_frame,
            ref_result,
            keypoint_conf=keypoint_conf,
            show_video=bool(self.show_video.get()),
            main_only=bool(self.main_only.get()),
        )
        user_view, drawn_user = draw_pose_overlay(
            user_frame,
            user_result,
            keypoint_conf=keypoint_conf,
            show_video=bool(self.show_video.get()),
            main_only=bool(self.main_only.get()),
        )
        draw_match_overlay(user_view, display_match, combo=self.combo)
        self._update_images(ref_view, user_view)

        now = time.perf_counter()
        fps = 1.0 / max(now - self.last_update_time, 1e-6)
        self.last_update_time = now
        self.score_text.set(
            f"score={display_match.score:5.1f} | feedback={display_match.feedback:<10} | "
            f"common={display_match.common_keypoints:02d}/12 | lag={format_lag(display_match.lag_frames)} | "
            f"mode={'mirror' if display_match.mirror_used else 'direct'} | combo={self.combo}"
        )
        self.status_text.set(
            f"ref_frame={self.ref_frame_index} persons={drawn_ref.person_count} infer={drawn_ref.inference_ms:.1f}ms | "
            f"user_frame={self.user_frame_index} persons={drawn_user.person_count} infer={drawn_user.inference_ms:.1f}ms | gui_fps={fps:.1f} | "
            f"{self._source_status()}"
        )

        self.ref_frame_index += 1
        self.user_frame_index += 1
        self.root.after(1, self._process_next_pair)

    def _update_images(self, ref_bgr: np.ndarray, user_bgr: np.ndarray) -> None:
        self.last_combined_bgr = make_combined_snapshot(ref_bgr, user_bgr)
        self._update_label(self.ref_label, ref_bgr)
        self._update_label(self.user_label, user_bgr)

    def _update_label(self, label: tk.Label, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((self.display_width, self.display_height), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=image)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    def _update_lag_window(self, _value: str) -> None:
        new_lag = int(self.lag_frames.get())
        if new_lag == self.matcher.max_lag_frames:
            return
        existing = list(self.matcher.reference_buffer)[-(new_lag + 1) :]
        self.matcher.set_max_lag_frames(new_lag, existing)

    def _smooth_match(self, match: PoseMatch) -> PoseMatch:
        if match.common_keypoints < 7:
            self.recent_scores.clear()
            return match
        self.recent_scores.append(match.score)
        smoothed_score = sum(self.recent_scores) / len(self.recent_scores)
        return replace(match, score=smoothed_score, feedback=feedback_label(smoothed_score, match.common_keypoints))

    def _source_status(self) -> str:
        user_suffix = f"@{self.user_start_frame}" if isinstance(self.user_source, Path) else ""
        return f"reference={self.reference_video.name}@{self.reference_start_frame} | user_source={self._source_name(self.user_source)}{user_suffix}"

    @staticmethod
    def _source_name(source: VideoSource) -> str:
        return f"camera:{source}" if isinstance(source, int) else source.name

    @staticmethod
    def _opencv_source(source: VideoSource) -> str | int:
        return source if isinstance(source, int) else str(source)


def draw_match_overlay(frame_bgr: np.ndarray, match: PoseMatch, *, combo: int = 0) -> None:
    feedback_colors = {
        "Perfect": (0, 220, 0),
        "Super": (40, 210, 255),
        "Good": (0, 180, 255),
        "Keep Going": (0, 130, 255),
        "Miss": (40, 40, 255),
        "Partial": (170, 170, 170),
    }
    color = feedback_colors.get(match.feedback, (180, 180, 180))
    text = f"{match.feedback}  {match.score:.0f}"
    if combo > 1:
        text = f"{text}  x{combo}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = 16, frame_bgr.shape[0] - 24
    cv2.rectangle(frame_bgr, (8, y - text_height - 14), (x + text_width + 12, y + baseline + 8), (0, 0, 0), -1)
    cv2.putText(frame_bgr, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def make_combined_snapshot(ref_bgr: np.ndarray, user_bgr: np.ndarray) -> np.ndarray:
    height = min(ref_bgr.shape[0], user_bgr.shape[0])
    ref_scaled = resize_to_height(ref_bgr, height)
    user_scaled = resize_to_height(user_bgr, height)
    return np.hstack([ref_scaled, user_scaled])


def resize_to_height(frame_bgr: np.ndarray, height: int) -> np.ndarray:
    scale = height / max(frame_bgr.shape[0], 1)
    width = max(1, int(frame_bgr.shape[1] * scale))
    return cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)


def format_lag(lag: int | None) -> str:
    return "n/a" if lag is None else f"{lag:+d}f"


def parse_source(text: str) -> VideoSource:
    stripped = text.strip()
    if stripped.isdigit():
        return int(stripped)
    return resolve_app_path(Path(stripped))


def resolve_app_path(path: Path) -> Path:
    return path if path.is_absolute() else APP_DIR / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bonus Task 2 reference-vs-webcam dance pose scoring GUI.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Reference video path.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLOv8 pose model path.")
    parser.add_argument("--user-source", default="0", help="Webcam index, or a video file for offline testing.")
    parser.add_argument("--ref-start-frame", type=int, default=0, help="Reference frame to start from.")
    parser.add_argument("--user-start-frame", type=int, default=0, help="User video frame to start from when user-source is a file.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = tk.Tk()
    BonusTask2App(
        root,
        resolve_app_path(args.video),
        parse_source(args.user_source),
        resolve_app_path(args.model),
        reference_start_frame=args.ref_start_frame,
        user_start_frame=args.user_start_frame,
    )
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
