from __future__ import annotations

import argparse
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import cv2
from PIL import Image, ImageTk

from pose_pipeline import PoseEstimator, PoseStreamTracker, draw_pose_overlay


APP_DIR = Path(__file__).resolve().parent
DEFAULT_VIDEO = APP_DIR / "dance_example_1.mp4"
DEFAULT_MODEL = APP_DIR / "yolov8n-pose.pt"


class BonusTask1App:
    def __init__(self, root: tk.Tk, video_path: Path, model_path: Path) -> None:
        self.root = root
        self.root.title("Bonus Task 1 - Reference Pose Detection")
        self.root.geometry("1120x760")

        self.video_path = video_path
        self.model_path = model_path
        self.estimator = PoseEstimator(model_path)
        self.tracker = PoseStreamTracker()

        self.cap: cv2.VideoCapture | None = None
        self.running = False
        self.frame_index = 0
        self.last_frame_bgr = None
        self.last_update_time = time.perf_counter()
        self.display_width = 960
        self.display_height = 540

        self.show_video = tk.BooleanVar(value=True)
        self.main_only = tk.BooleanVar(value=False)
        self.loop_video = tk.BooleanVar(value=True)
        self.confidence = tk.DoubleVar(value=0.30)
        self.keypoint_confidence = tk.DoubleVar(value=0.25)

        self._build_ui()
        self._show_placeholder()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self) -> None:
        outer = tk.Frame(self.root)
        outer.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        title = tk.Label(outer, text="Reference Video Pose Detection (Bonus Task 1)", font=("Segoe UI", 15, "bold"))
        title.pack(anchor="w")

        self.video_label = tk.Label(outer, bg="#111111", width=self.display_width, height=self.display_height)
        self.video_label.pack(fill=tk.BOTH, expand=True, pady=(10, 8))

        controls = tk.Frame(outer)
        controls.pack(fill=tk.X)
        tk.Button(controls, text="Open Video", command=self.open_video).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="Start", command=self.start).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="Restart", command=self.restart).pack(side=tk.LEFT, padx=4)
        tk.Button(controls, text="Save Snapshot", command=self.save_snapshot).pack(side=tk.LEFT, padx=4)

        toggles = tk.Frame(outer)
        toggles.pack(fill=tk.X, pady=(8, 4))
        tk.Checkbutton(toggles, text="Show original video", variable=self.show_video).pack(side=tk.LEFT, padx=4)
        tk.Checkbutton(toggles, text="Main dancer only", variable=self.main_only).pack(side=tk.LEFT, padx=4)
        tk.Checkbutton(toggles, text="Loop video", variable=self.loop_video).pack(side=tk.LEFT, padx=4)

        sliders = tk.Frame(outer)
        sliders.pack(fill=tk.X, pady=(4, 4))
        tk.Label(sliders, text="Detector conf").pack(side=tk.LEFT)
        tk.Scale(sliders, variable=self.confidence, from_=0.1, to=0.7, resolution=0.05, orient=tk.HORIZONTAL, length=190).pack(side=tk.LEFT)
        tk.Label(sliders, text="Keypoint conf").pack(side=tk.LEFT, padx=(16, 0))
        tk.Scale(sliders, variable=self.keypoint_confidence, from_=0.05, to=0.7, resolution=0.05, orient=tk.HORIZONTAL, length=190).pack(side=tk.LEFT)

        self.status_text = tk.StringVar(value=f"Video: {self.video_path.name}")
        tk.Label(outer, textvariable=self.status_text, anchor="w", justify=tk.LEFT, font=("Consolas", 10)).pack(fill=tk.X, pady=(6, 0))

    def _show_placeholder(self) -> None:
        placeholder = Image.new("RGB", (self.display_width, self.display_height), (28, 28, 28))
        imgtk = ImageTk.PhotoImage(placeholder)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def open_video(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(APP_DIR),
            filetypes=[("Video files", "*.mp4;*.avi;*.mov;*.mkv"), ("All files", "*.*")],
        )
        if path:
            self.video_path = Path(path)
            self.stop()
            self.status_text.set(f"Video selected: {self.video_path}")

    def start(self) -> None:
        if not self.video_path.exists():
            messagebox.showerror("Missing video", f"Video file not found:\n{self.video_path}")
            return
        if self.running:
            return
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            messagebox.showerror("Open failed", f"Cannot open video:\n{self.video_path}")
            return
        self.running = True
        self.frame_index = 0
        self.tracker.reset()
        self.last_update_time = time.perf_counter()
        self._process_next_frame()

    def stop(self) -> None:
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def restart(self) -> None:
        self.stop()
        self.start()

    def close(self) -> None:
        self.stop()
        self.root.destroy()

    def save_snapshot(self) -> None:
        if self.last_frame_bgr is None:
            messagebox.showwarning("No frame", "Run the video first.")
            return
        output_dir = APP_DIR / "outputs"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"bonus_task1_snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(output_path), self.last_frame_bgr)
        messagebox.showinfo("Snapshot saved", str(output_path))

    def _process_next_frame(self) -> None:
        if not self.running or self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            if self.loop_video.get():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_index = 0
                self.root.after(1, self._process_next_frame)
                return
            self.stop()
            return

        result = self.estimator.infer(
            frame,
            frame_index=self.frame_index,
            conf=float(self.confidence.get()),
            keypoint_conf=float(self.keypoint_confidence.get()),
            main_only=False,
        )
        result = self.tracker.update(result)
        annotated, drawn_result = draw_pose_overlay(
            frame,
            result,
            keypoint_conf=float(self.keypoint_confidence.get()),
            show_video=bool(self.show_video.get()),
            main_only=bool(self.main_only.get()),
        )
        self.last_frame_bgr = annotated
        self._update_image(annotated)

        now = time.perf_counter()
        fps = 1.0 / max(now - self.last_update_time, 1e-6)
        self.last_update_time = now
        main = drawn_result.main_detection
        visible = main.visible_count if main is not None else 0
        self.status_text.set(
            f"video={self.video_path.name} | frame={self.frame_index} | persons={drawn_result.person_count} | "
            f"main_visible={visible}/17 | infer={drawn_result.inference_ms:.1f}ms | draw={drawn_result.draw_ms:.1f}ms | gui_fps={fps:.1f}"
        )
        self.frame_index += 1

        video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        delay_ms = max(1, int(1000.0 / min(video_fps, 30.0)))
        self.root.after(delay_ms, self._process_next_frame)

    def _update_image(self, frame_bgr) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((self.display_width, self.display_height), Image.Resampling.LANCZOS)
        imgtk = ImageTk.PhotoImage(image=image)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bonus Task 1 reference video pose GUI.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO, help="Reference video path.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="YOLOv8 pose model path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = tk.Tk()
    BonusTask1App(root, args.video, args.model)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
