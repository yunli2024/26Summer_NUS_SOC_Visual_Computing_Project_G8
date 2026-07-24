"""Tkinter Bonus Level app: reference video + webcam pose scoring."""

from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from . import config
from .dancer_tracker import DancerTracker
from .pose_detector import PoseDetector, draw_pose
from .pose_features import add_joint_angles
from .pose_filter import PoseFilter
from .pose_normalizer import normalize_pose, swap_left_right_keypoints
from .pose_types import PoseFrame
from .score_manager import ScoreManager
from .sound_feedback import SoundFeedback
from .temporal_alignment import TemporalAligner


class BonusPoseApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Bonus Level - Pose Dance Scoring")
        self.root.geometry("1380x740")

        self.detector: PoseDetector | None = None
        self.ui_queue: queue.Queue = queue.Queue(maxsize=20)

        self.video_path = str(config.DEFAULT_VIDEO_PATH) if config.DEFAULT_VIDEO_PATH.exists() else ""
        self.reference_running = False
        self.reference_paused = False
        self.webcam_running = False
        self.cap_reference = None
        self.cap_webcam = None
        self.reference_thread = None
        self.webcam_thread = None

        self.ref_tracker = DancerTracker()
        self.user_tracker = DancerTracker()
        self.ref_filter = PoseFilter()
        self.user_filter = PoseFilter()
        self.aligner = TemporalAligner()
        self.score_manager = ScoreManager()
        self.sound_feedback = SoundFeedback()

        self.reference_start_time = 0.0
        self.webcam_start_time = 0.0
        self.reference_frame_index = 0
        self.webcam_frame_index = 0
        self.reference_total_frames = 0
        self.reference_fps = 0.0
        self.reference_duration = 0.0
        self.latest_score_text = "Score 0 | Ready"
        self.placeholder_images = {}
        self.default_score_bg = None
        self.feedback_flash_job = None
        self.last_visual_feedback = ""
        self.settlement_window = None
        self.game_over = False

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(20, self.process_ui_queue)

    def _build_ui(self):
        title = tk.Label(self.root, text="Bonus Level: Pose Detection and Just Dance Scoring", font=("Arial", 18, "bold"))
        title.pack(pady=8)

        top = tk.Frame(self.root)
        top.pack(fill=tk.X, expand=False)

        self.left = tk.Frame(top)
        self.left.pack(side=tk.LEFT, padx=10, fill=tk.Y)
        self.right = tk.Frame(top)
        self.right.pack(side=tk.RIGHT, padx=10, fill=tk.Y)

        tk.Label(self.left, text="Reference Video", font=("Arial", 13, "bold")).pack()
        self.label_reference = tk.Label(self.left, bg="black")
        self.label_reference.pack()
        self.set_placeholder(self.label_reference, "Click Start to play reference video")
        self.status_reference = tk.Label(self.left, text=self._video_status_text(), anchor="w", justify=tk.LEFT)
        self.status_reference.pack(fill=tk.X)

        controls_ref = tk.Frame(self.left)
        controls_ref.pack(pady=6)
        tk.Button(controls_ref, text="Open", command=self.open_video).pack(side=tk.LEFT, padx=3)
        tk.Button(controls_ref, text="Start", command=self.start_reference).pack(side=tk.LEFT, padx=3)
        tk.Button(controls_ref, text="Pause", command=self.pause_reference).pack(side=tk.LEFT, padx=3)
        tk.Button(controls_ref, text="Resume", command=self.resume_reference).pack(side=tk.LEFT, padx=3)
        tk.Button(controls_ref, text="Stop", command=self.stop_reference).pack(side=tk.LEFT, padx=3)
        tk.Button(controls_ref, text="Restart", command=self.restart_reference).pack(side=tk.LEFT, padx=3)

        tk.Label(self.right, text="Webcam Player", font=("Arial", 13, "bold")).pack()
        self.label_webcam = tk.Label(self.right, bg="black")
        self.label_webcam.pack()
        self.set_placeholder(self.label_webcam, "Click Start Webcam to begin")
        self.status_webcam = tk.Label(self.right, text="Webcam: idle", anchor="w", justify=tk.LEFT)
        self.status_webcam.pack(fill=tk.X)

        controls_cam = tk.Frame(self.right)
        controls_cam.pack(pady=6)
        tk.Button(controls_cam, text="Start Webcam", command=self.start_webcam).pack(side=tk.LEFT, padx=3)
        tk.Button(controls_cam, text="Stop Webcam", command=self.stop_webcam).pack(side=tk.LEFT, padx=3)
        tk.Button(controls_cam, text="Reset Score", command=self.reset_score).pack(side=tk.LEFT, padx=3)
        tk.Button(controls_cam, text="Show Summary", command=self.show_summary).pack(side=tk.LEFT, padx=3)

        score_panel = tk.Frame(self.root)
        score_panel.pack(fill=tk.X, padx=12, pady=6)
        self.score_label = tk.Label(score_panel, text="Score: 0.0 | Feedback: Ready | Combo: 0 | Total: 0", font=("Arial", 16, "bold"))
        self.score_label.pack(side=tk.LEFT)
        self.default_score_bg = self.score_label.cget("bg")
        self.config_label = tk.Label(
            score_panel,
            text=f"Model: {config.MODEL_PATH.name} | GPU auto | keypoint conf {config.KEYPOINT_CONF_THRESHOLD}",
            anchor="e",
        )
        self.config_label.pack(side=tk.RIGHT)

        stats_panel = tk.Frame(self.root)
        stats_panel.pack(fill=tk.X, padx=12, pady=4)
        self.stats_label = tk.Label(
            stats_panel,
            text="Average: 0.0 | Best: 0.0 | Samples: 0 | P/S/G/M: 0/0/0/0",
            font=("Arial", 11),
            anchor="w",
        )
        self.stats_label.pack(side=tk.LEFT)
        self.breakdown_label = tk.Label(
            stats_panel,
            text="Pose: 0.0 | Position: 0.0 | Angle: 0.0 | Motion: 0.0",
            font=("Arial", 11),
            anchor="e",
        )
        self.breakdown_label.pack(side=tk.RIGHT)

        progress_panel = tk.Frame(self.root)
        progress_panel.pack(fill=tk.X, padx=12, pady=4)
        self.progress_label = tk.Label(progress_panel, text="Progress: 0.0s / 0.0s", width=24, anchor="w")
        self.progress_label.pack(side=tk.LEFT)
        self.progress_canvas = tk.Canvas(progress_panel, height=14, bg="#dddddd", highlightthickness=0)
        self.progress_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.summary_label = tk.Label(
            self.root,
            text="Summary will appear here after the dance or when you click Show Summary.",
            font=("Arial", 11),
            anchor="w",
            justify=tk.LEFT,
        )
        self.summary_label.pack(fill=tk.X, padx=12, pady=4)

    def set_placeholder(self, label, text):
        width, height = config.DISPLAY_SIZE
        image = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(image, text, (32, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)
        self.update_label(label, image)

    def _video_status_text(self):
        if self.video_path:
            return f"Reference video: {os.path.basename(self.video_path)}"
        return "Reference video: not selected"

    def ensure_detector(self):
        if self.detector is None:
            try:
                self.detector = PoseDetector()
                self.config_label.configure(text=f"Model: {config.MODEL_PATH} | device {self.detector.device}")
            except Exception as exc:
                messagebox.showerror("YOLO Model Error", str(exc))
                raise
        return self.detector

    def open_video(self):
        if self.reference_running:
            messagebox.showwarning("Reference Running", "Please stop the reference video before choosing another file.")
            return
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")])
        if path:
            self.video_path = path
            self.status_reference.configure(text=self._video_status_text())

    def start_reference(self):
        self.close_settlement_window()
        if not self.video_path:
            messagebox.showwarning("No Video", "Please select a reference video first.")
            return
        if self.reference_running:
            return
        self.ensure_detector()
        self.game_over = False
        self.reference_running = True
        self.reference_paused = False
        self.ref_tracker.reset()
        self.ref_filter.reset()
        self.aligner.reset()
        self.reference_frame_index = 0
        self.reference_start_time = time.perf_counter()
        self.reference_thread = threading.Thread(target=self.reference_loop, daemon=True)
        self.reference_thread.start()

    def pause_reference(self):
        self.reference_paused = True

    def resume_reference(self):
        self.reference_paused = False

    def stop_reference(self):
        self.reference_running = False
        self.reference_paused = False
        if self.cap_reference is not None:
            self.cap_reference.release()
            self.cap_reference = None

    def restart_reference(self):
        self.close_settlement_window()
        self.stop_reference()
        self.root.after(150, self.start_reference)

    def start_webcam(self):
        self.close_settlement_window()
        if self.webcam_running:
            return
        self.ensure_detector()
        self.game_over = False
        self.webcam_running = True
        self.user_tracker.reset()
        self.user_filter.reset()
        self.webcam_frame_index = 0
        self.webcam_start_time = time.perf_counter()
        self.webcam_thread = threading.Thread(target=self.webcam_loop, daemon=True)
        self.webcam_thread.start()

    def stop_webcam(self):
        self.webcam_running = False
        if self.cap_webcam is not None:
            self.cap_webcam.release()
            self.cap_webcam = None

    def reset_score(self):
        self.close_settlement_window()
        self.game_over = False
        self.score_manager.reset()
        self.sound_feedback.reset()
        self.last_visual_feedback = ""
        self.latest_score_text = "Score 0 | Ready"
        self.score_label.configure(text="Score: 0.0 | Feedback: Ready | Combo: 0 | Total: 0")
        self.stats_label.configure(text="Average: 0.0 | Best: 0.0 | Samples: 0 | P/S/G/M: 0/0/0/0")
        self.breakdown_label.configure(text="Pose: 0.0 | Position: 0.0 | Angle: 0.0 | Motion: 0.0")
        self.summary_label.configure(text="Summary will appear here after the dance or when you click Show Summary.")

    def show_summary(self):
        self.summary_label.configure(text=self.score_manager.summary_text().replace("\n", " | "))
        messagebox.showinfo("Dance Summary", self.score_manager.summary_text())

    def finish_game(self):
        if self.game_over:
            return
        self.game_over = True
        self.reference_running = False
        self.reference_paused = False
        self.webcam_running = False
        if self.cap_reference is not None:
            self.cap_reference.release()
            self.cap_reference = None
        if self.cap_webcam is not None:
            self.cap_webcam.release()
            self.cap_webcam = None
        self.post_status(self.status_reference, "Reference: ended")
        self.post_status(self.status_webcam, "Webcam: stopped")
        self.post_frame(self.label_reference, self.placeholder_frame("Dance complete"))
        self.post_frame(self.label_webcam, self.placeholder_frame("Camera stopped"))
        self._post(("game_over", None, self.score_manager.state()))

    def reference_loop(self):
        self.cap_reference = cv2.VideoCapture(self.video_path)
        if not self.cap_reference.isOpened():
            self.reference_running = False
            self.post_status(self.status_reference, "Reference video: cannot open")
            return
        fps = self.cap_reference.get(cv2.CAP_PROP_FPS)
        total_frames = int(self.cap_reference.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.reference_fps = fps if fps and fps > 1 else 30.0
        self.reference_total_frames = total_frames
        self.reference_duration = total_frames / self.reference_fps if total_frames else 0.0
        frame_delay = 1.0 / fps if fps and fps > 1 else 1.0 / 30.0

        while self.reference_running and self.cap_reference.isOpened():
            if self.reference_paused:
                time.sleep(0.05)
                continue
            frame_start = time.perf_counter()
            ok, frame = self.cap_reference.read()
            if not ok:
                break
            self.reference_frame_index += 1
            timestamp = time.perf_counter() - self.reference_start_time
            pose_frame, selected, people_count, infer_ms = self.process_frame(
                frame, "reference", timestamp, self.reference_frame_index, self.ref_tracker, self.ref_filter
            )
            if pose_frame.valid_count >= config.MIN_VALID_KEYPOINTS:
                self.aligner.add_reference(pose_frame)
            rendered = draw_pose(frame, selected, people_count, infer_ms, "Reference")
            self.post_frame(self.label_reference, rendered)
            self.post_status(
                self.status_reference,
                f"Reference: playing | frame {self.reference_frame_index} | people {people_count} | "
                f"selected {pose_frame.selected_index} | valid {pose_frame.valid_count} | infer {infer_ms:.1f} ms",
            )
            self.post_progress(timestamp)
            if self.reference_duration > 0 and timestamp >= self.reference_duration:
                break
            if self.reference_total_frames > 0 and self.reference_frame_index >= self.reference_total_frames:
                break
            elapsed = time.perf_counter() - frame_start
            time.sleep(max(0.0, frame_delay - elapsed))

        completed = self.reference_running
        if self.cap_reference is not None:
            self.cap_reference.release()
            self.cap_reference = None
        self.reference_running = False
        self.reference_paused = False
        if completed:
            self.finish_game()
        else:
            self.post_status(self.status_reference, "Reference: stopped")

    def webcam_loop(self):
        self.cap_webcam = cv2.VideoCapture(0)
        if not self.cap_webcam.isOpened():
            self.webcam_running = False
            self.post_status(self.status_webcam, "Webcam: cannot open")
            return

        while self.webcam_running and self.cap_webcam.isOpened():
            ok, frame = self.cap_webcam.read()
            if not ok:
                break
            if config.MIRROR_WEBCAM:
                frame = cv2.flip(frame, 1)
            self.webcam_frame_index += 1
            timestamp = time.perf_counter() - self.reference_start_time if self.reference_start_time else time.perf_counter() - self.webcam_start_time
            pose_frame, selected, people_count, infer_ms = self.process_frame(
                frame,
                "webcam",
                timestamp,
                self.webcam_frame_index,
                self.user_tracker,
                self.user_filter,
                swap_left_right=config.MIRROR_WEBCAM and config.SWAP_LEFT_RIGHT,
            )
            if not self.webcam_running or self.game_over:
                break
            result = {"score": 0.0, "feedback": "Start reference"}
            offset = 0.0
            if pose_frame.valid_count >= config.MIN_VALID_KEYPOINTS:
                _, result, offset = self.aligner.match(pose_frame)
            score_state = self.score_manager.update(float(result["score"]), str(result["feedback"]), timestamp, result)
            self.latest_score_text = f"{score_state['feedback']} {score_state['smooth']:.0f} | Combo {score_state['combo']}"
            rendered = draw_pose(
                frame,
                selected,
                people_count,
                infer_ms,
                self.latest_score_text,
                result.get("error_keypoints", []),
                result.get("error_joints", []),
                str(result.get("error_summary", "")),
            )
            self.post_frame(self.label_webcam, rendered)
            self.post_status(
                self.status_webcam,
                f"Webcam: running | people {people_count} | valid {pose_frame.valid_count} | "
                f"infer {infer_ms:.1f} ms | offset {offset:+.2f}s | user buffer {len(self.aligner.user_frames)}",
            )
            self.post_score(score_state)

        if self.cap_webcam is not None:
            self.cap_webcam.release()
            self.cap_webcam = None
        self.webcam_running = False
        self.post_status(self.status_webcam, "Webcam: stopped")

    def process_frame(self, frame, source, timestamp, frame_index, tracker, pose_filter, swap_left_right=False):
        people, infer_ms = self.detector.detect(frame)
        selected_index, selected = tracker.select(people, frame.shape)
        selected = pose_filter.apply(selected)

        if selected is None:
            pose_frame = PoseFrame.empty(timestamp, frame_index, source, len(people))
            pose_frame.infer_ms = infer_ms
            return pose_frame, None, len(people), infer_ms

        if swap_left_right:
            selected.keypoints, selected.valid_mask, selected.confidences = swap_left_right_keypoints(
                selected.keypoints, selected.valid_mask, selected.confidences
            )

        pose_frame = PoseFrame(
            timestamp=timestamp,
            frame_index=frame_index,
            source=source,
            bbox=selected.bbox,
            keypoints=selected.keypoints,
            confidences=selected.confidences,
            valid_mask=selected.valid_mask,
            pose_confidence=selected.pose_confidence,
            people_count=len(people),
            selected_index=-1 if selected_index is None else selected_index,
            infer_ms=infer_ms,
        )
        normalize_pose(pose_frame)
        add_joint_angles(pose_frame)
        return pose_frame, selected, len(people), infer_ms

    def post_frame(self, label, frame):
        self._post(("frame", label, frame))

    def post_status(self, label, text):
        self._post(("status", label, text))

    def post_score(self, state):
        self._post(("score", None, state))

    def post_progress(self, timestamp):
        self._post(("progress", None, timestamp))

    def post_summary(self):
        self._post(("summary", None, self.score_manager.summary_text()))

    def _post(self, item):
        try:
            self.ui_queue.put_nowait(item)
        except queue.Full:
            pass

    def process_ui_queue(self):
        while True:
            try:
                kind, target, payload = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "frame":
                self.update_label(target, payload)
            elif kind == "status":
                target.configure(text=payload)
            elif kind == "score":
                self.update_score_labels(payload)
            elif kind == "progress":
                self.update_progress(float(payload))
            elif kind == "summary":
                self.summary_label.configure(text=str(payload).replace("\n", " | "))
            elif kind == "game_over":
                self.show_settlement(payload)
        self.root.after(20, self.process_ui_queue)

    def update_score_labels(self, state):
        self.score_label.configure(
            text=(
                f"Score: {state['smooth']:.1f} | Feedback: {state['feedback']} | "
                f"Combo: {state['combo']} | Total: {state['total']} | Best Combo: {state['best_combo']}"
            )
        )
        self.stats_label.configure(
            text=(
                f"Average: {state['average']:.1f} | Best: {state['best_score']:.1f} | "
                f"Samples: {state['samples']} | P/S/G/M: "
                f"{state['perfect']}/{state['super']}/{state['good']}/{state['miss']} | "
                f"Delay median/p90: {state['median_lag']:.2f}/{state['p90_lag']:.2f}s"
            )
        )
        self.breakdown_label.configure(
            text=(
                f"Pose: {state['pose']:.1f} | Position: {state['position']:.1f} | "
                f"Angle: {state['angle']:.1f} | Motion: {state['motion']:.1f} | "
                f"Coverage: {state['coverage'] * 100:.0f}% | "
                f"Fix: {state['error_summary'] or 'None'}"
            )
        )
        self.trigger_feedback_effect(str(state["feedback"]))

    def trigger_feedback_effect(self, feedback: str):
        if feedback == self.last_visual_feedback:
            return
        self.last_visual_feedback = feedback
        self.sound_feedback.play(feedback)
        color = config.FEEDBACK_FLASH_COLORS.get(feedback)
        if not color:
            return
        if self.feedback_flash_job is not None:
            try:
                self.root.after_cancel(self.feedback_flash_job)
            except Exception:
                pass
        self.score_label.configure(bg=color)
        self.feedback_flash_job = self.root.after(config.FEEDBACK_FLASH_MS, self.clear_feedback_flash)

    def clear_feedback_flash(self):
        self.feedback_flash_job = None
        self.score_label.configure(bg=self.default_score_bg)

    def update_progress(self, timestamp):
        duration = self.reference_duration
        progress = min(1.0, timestamp / duration) if duration > 0 else 0.0
        self.progress_label.configure(text=f"Progress: {timestamp:.1f}s / {duration:.1f}s")
        self.progress_canvas.delete("all")
        width = max(1, self.progress_canvas.winfo_width())
        height = max(1, self.progress_canvas.winfo_height())
        self.progress_canvas.create_rectangle(0, 0, width, height, fill="#dddddd", outline="")
        self.progress_canvas.create_rectangle(0, 0, int(width * progress), height, fill="#2e86de", outline="")

    def update_label(self, label, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb).resize(config.DISPLAY_SIZE)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    def placeholder_frame(self, text):
        width, height = config.DISPLAY_SIZE
        image = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(image, text, (48, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2, cv2.LINE_AA)
        return image

    def show_settlement(self, state):
        self.summary_label.configure(text=self.score_manager.summary_text().replace("\n", " | "))
        self.update_score_labels(state)
        if self.settlement_window is not None and self.settlement_window.winfo_exists():
            self.settlement_window.lift()
            return

        window = tk.Toplevel(self.root)
        self.settlement_window = window
        window.title("Dance Complete")
        window.geometry("460x360")
        window.resizable(False, False)
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", self.close_settlement_window)

        tk.Label(window, text="Dance Complete", font=("Arial", 22, "bold")).pack(pady=(22, 8))
        tk.Label(window, text=f"Final Score: {state['total']}", font=("Arial", 28, "bold"), fg="#1f6feb").pack(pady=6)
        tk.Label(window, text=f"Average: {state['average']:.1f}   Best: {state['best_score']:.1f}", font=("Arial", 13)).pack(pady=4)
        tk.Label(window, text=f"Best Combo: {state['best_combo']}", font=("Arial", 13)).pack(pady=4)
        tk.Label(
            window,
            text=f"Perfect / Super / Good / Miss: {state['perfect']} / {state['super']} / {state['good']} / {state['miss']}",
            font=("Arial", 12),
        ).pack(pady=4)
        tk.Label(window, text=f"Scored Samples: {state['samples']}", font=("Arial", 12)).pack(pady=4)

        buttons = tk.Frame(window)
        buttons.pack(pady=20)
        tk.Button(buttons, text="New Round", width=14, command=self.prepare_new_round).pack(side=tk.LEFT, padx=8)
        tk.Button(buttons, text="Close", width=14, command=self.close_settlement_window).pack(side=tk.LEFT, padx=8)

    def prepare_new_round(self):
        self.close_settlement_window()
        self.reset_score()
        self.game_over = False
        self.ref_tracker.reset()
        self.user_tracker.reset()
        self.ref_filter.reset()
        self.user_filter.reset()
        self.aligner.reset()
        self.reference_frame_index = 0
        self.webcam_frame_index = 0
        self.set_placeholder(self.label_reference, "Click Start to play reference video")
        self.set_placeholder(self.label_webcam, "Click Start Webcam to begin")
        self.status_reference.configure(text=self._video_status_text())
        self.status_webcam.configure(text="Webcam: idle")
        self.update_progress(0.0)

    def close_settlement_window(self):
        if self.settlement_window is not None:
            try:
                if self.settlement_window.winfo_exists():
                    self.settlement_window.destroy()
            except Exception:
                pass
            self.settlement_window = None

    def on_close(self):
        self.close_settlement_window()
        self.reference_running = False
        self.webcam_running = False
        self.reference_paused = False
        if self.cap_reference is not None:
            self.cap_reference.release()
        if self.cap_webcam is not None:
            self.cap_webcam.release()
        self.root.after(100, self.root.destroy)


def run_app() -> int:
    root = tk.Tk()
    BonusPoseApp(root)
    root.mainloop()
    return 0
