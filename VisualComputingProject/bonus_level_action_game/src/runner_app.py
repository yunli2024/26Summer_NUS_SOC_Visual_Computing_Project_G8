"""Tkinter/OpenCV prototype for an upper-body controlled three-lane runner."""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageTk

from . import config
from .dancer_tracker import DancerTracker
from .pose_detector import PoseDetector, draw_pose
from .pose_filter import PoseFilter
from .pose_normalizer import swap_left_right_keypoints
from .pose_types import PoseFrame
from .runner_actions import Action, GestureRecognizer
from .runner_game import RunnerGame


class RunnerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Upper Body Three-Lane Runner")
        self.root.geometry(f"{config.RUNNER_WINDOW_SIZE[0]}x{config.RUNNER_WINDOW_SIZE[1]}")
        self.root.resizable(False, False)

        self.game = RunnerGame()
        self.recognizer = GestureRecognizer()
        self.detector: PoseDetector | None = None
        self.tracker = DancerTracker()
        self.pose_filter = PoseFilter(alpha=config.SMOOTHING_ALPHA)
        self.ui_queue: queue.Queue = queue.Queue(maxsize=8)
        self.capture = None
        self.capture_thread = None
        self.capture_running = False
        self.frame_index = 0
        self.start_time = time.perf_counter()
        self.last_tick = time.perf_counter()
        self.latest_action = Action.NONE
        self.latest_preview = None
        self.warning = ""

        self.label = tk.Label(root, bg="black")
        self.label.pack()
        self._bind_keys()
        self.root.protocol("WM_DELETE_WINDOW", self.shutdown)
        self.start_pose_input()
        self.root.after(0, self.game_loop)
        self.root.after(15, self.process_ui_queue)

    def _bind_keys(self):
        bindings = {
            "<a>": Action.LEFT,
            "<A>": Action.LEFT,
            "<Left>": Action.LEFT,
            "<d>": Action.RIGHT,
            "<D>": Action.RIGHT,
            "<Right>": Action.RIGHT,
            "<w>": Action.JUMP,
            "<W>": Action.JUMP,
            "<Up>": Action.JUMP,
            "<s>": Action.SLIDE,
            "<S>": Action.SLIDE,
            "<Down>": Action.SLIDE,
        }
        for key, action in bindings.items():
            self.root.bind(key, lambda _event, act=action: self.handle_action(act))
        self.root.bind("<c>", lambda _event: self.recalibrate())
        self.root.bind("<C>", lambda _event: self.recalibrate())
        self.root.bind("<r>", lambda _event: self.restart())
        self.root.bind("<R>", lambda _event: self.restart())
        self.root.bind("<Escape>", lambda _event: self.shutdown())

    def start_pose_input(self):
        try:
            self.detector = PoseDetector()
        except Exception as exc:
            self.warning = f"Pose model unavailable: {exc}"
            self.detector = None
            return
        self.capture = self.open_capture()
        if self.capture is None or not self.capture.isOpened():
            self.warning = "Camera/video unavailable; keyboard control active"
            return
        self.capture_running = True
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()

    def open_capture(self):
        video_path = str(config.RUNNER_VIDEO_PATH).strip()
        if video_path:
            path = Path(video_path)
            if not path.exists():
                self.warning = f"Video not found: {path}"
                return None
            return cv2.VideoCapture(str(path))
        cap = cv2.VideoCapture(config.RUNNER_CAMERA_INDEX, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(config.RUNNER_CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.RUNNER_CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.RUNNER_CAMERA_HEIGHT)
        return cap

    def capture_loop(self):
        while self.capture_running and self.capture is not None and self.capture.isOpened():
            ok, frame = self.capture.read()
            if not ok:
                if str(config.RUNNER_VIDEO_PATH).strip():
                    self.warning = "Video ended; keyboard control active"
                    self.capture_running = False
                    break
                time.sleep(0.02)
                continue
            if config.MIRROR_WEBCAM and not str(config.RUNNER_VIDEO_PATH).strip():
                frame = cv2.flip(frame, 1)
            self.frame_index += 1
            timestamp = time.perf_counter() - self.start_time
            pose_frame, selected, people_count, infer_ms = self.process_pose_frame(frame, timestamp)
            action = self.recognizer.update(pose_frame)
            preview = draw_pose(frame, selected, people_count, infer_ms, self.recognizer.status)
            self._post(("pose", action, preview))

    def process_pose_frame(self, frame, timestamp):
        if self.detector is None:
            return PoseFrame.empty(timestamp, self.frame_index, "none", 0), None, 0, 0.0
        infer_frame = self.resize_for_inference(frame)
        scale_x = frame.shape[1] / infer_frame.shape[1]
        scale_y = frame.shape[0] / infer_frame.shape[0]
        people, infer_ms = self.detector.detect(infer_frame)
        for person in people:
            person.keypoints[:, 0] *= scale_x
            person.keypoints[:, 1] *= scale_y
            x1, y1, x2, y2 = person.bbox
            person.bbox = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
        selected_index, selected = self.tracker.select(people, frame.shape)
        selected = self.pose_filter.apply(selected)
        if selected is None:
            pose_frame = PoseFrame.empty(timestamp, self.frame_index, "runner", len(people))
            pose_frame.infer_ms = infer_ms
            return pose_frame, None, len(people), infer_ms
        if config.MIRROR_WEBCAM and not str(config.RUNNER_VIDEO_PATH).strip() and config.SWAP_LEFT_RIGHT:
            selected.keypoints, selected.valid_mask, selected.confidences = swap_left_right_keypoints(
                selected.keypoints, selected.valid_mask, selected.confidences
            )
        pose_frame = PoseFrame(
            timestamp=timestamp,
            frame_index=self.frame_index,
            source="runner",
            bbox=selected.bbox,
            keypoints=selected.keypoints,
            confidences=selected.confidences,
            valid_mask=selected.valid_mask,
            pose_confidence=selected.pose_confidence,
            people_count=len(people),
            selected_index=-1 if selected_index is None else selected_index,
            infer_ms=infer_ms,
        )
        return pose_frame, selected, len(people), infer_ms

    def resize_for_inference(self, frame):
        width = frame.shape[1]
        if width <= config.RUNNER_INFER_WIDTH:
            return frame
        scale = config.RUNNER_INFER_WIDTH / width
        height = max(1, int(frame.shape[0] * scale))
        return cv2.resize(frame, (config.RUNNER_INFER_WIDTH, height))

    def handle_action(self, action: Action):
        self.latest_action = action
        self.game.handle_action(action)

    def recalibrate(self):
        self.recognizer.reset_calibration()
        self.warning = "Recalibrating"

    def restart(self):
        self.game.reset()
        self.latest_action = Action.NONE

    def game_loop(self):
        now = time.perf_counter()
        dt = min(0.05, now - self.last_tick)
        self.last_tick = now
        if self.recognizer.calibrated or self._keyboard_fallback_active():
            self.game.update(dt)
        frame = self.game.render(
            camera_preview=self.latest_preview,
            action=self.latest_action,
            recognizer_status=self.recognizer.status,
            calibrated=self.recognizer.calibrated,
            warning=self.warning,
        )
        self.update_label(frame)
        delay_ms = int(1000 / config.RUNNER_GAME_FPS)
        self.root.after(delay_ms, self.game_loop)

    def _post(self, item):
        try:
            self.ui_queue.put_nowait(item)
        except queue.Full:
            pass

    def process_ui_queue(self):
        while True:
            try:
                kind, action, preview = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "pose":
                if action is not Action.NONE:
                    self.handle_action(action)
                self.latest_preview = preview
        self.root.after(15, self.process_ui_queue)

    def _keyboard_fallback_active(self) -> bool:
        text = self.warning.lower()
        return "unavailable" in text or "ended" in text or "not found" in text

    def update_label(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=image)
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

    def shutdown(self):
        self.capture_running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        cv2.destroyAllWindows()
        self.root.after(60, self._finish_shutdown)

    def _finish_shutdown(self):
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def run_app() -> int:
    root = tk.Tk()
    RunnerApp(root)
    root.mainloop()
    return 0
