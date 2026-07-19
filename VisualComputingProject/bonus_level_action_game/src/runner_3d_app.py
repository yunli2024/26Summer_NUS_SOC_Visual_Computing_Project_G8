"""Ursina entry point with YOLO Pose gesture input for the 3D runner."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import cv2

from . import config
from .pose_normalizer import swap_left_right_keypoints
from .pose_types import PoseFrame
from .runner_actions import Action, GestureRecognizer


class PoseInput:
    def __init__(self):
        self.detector = None
        self.tracker = None
        self.pose_filter = None
        self.recognizer = GestureRecognizer()
        self.capture = None
        self.running = False
        self.thread = None
        self.model_thread = None
        self.model_load_requested = False
        self.frame_index = 0
        self.start_time = time.perf_counter()
        self.queue: queue.Queue[Action] = queue.Queue(maxsize=4)
        self.warning = ""
        self.preview_lock = threading.Lock()
        self.preview_rgb = None
        self.preview_id = 0

    def start(self):
        self.capture = self._open_capture()
        if self.capture is None or not self.capture.isOpened():
            self.warning = "Camera/video unavailable; keyboard only"
            return
        if config.RUNNER_SHOW_CV_PREVIEW:
            cv2.namedWindow("Pose Camera Preview", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Pose Camera Preview", 480, 270)
            cv2.moveWindow("Pose Camera Preview", 40, 80)
        self.running = True
        self.warning = "" if self.detector is not None else "Camera preview starting; pose model loads next"
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def load_pose_model(self):
        self._load_pose_model()

    def _load_pose_model(self):
        try:
            from .dancer_tracker import DancerTracker
            from .pose_detector import PoseDetector
            from .pose_filter import PoseFilter

            detector = PoseDetector()
            tracker = DancerTracker()
            pose_filter = PoseFilter(alpha=config.SMOOTHING_ALPHA)
        except Exception as exc:
            self.detector = None
            self.tracker = None
            self.pose_filter = None
            self.warning = f"Pose model unavailable; camera preview only: {exc}"
            return
        self.detector = detector
        self.tracker = tracker
        self.pose_filter = pose_filter
        self.warning = ""

    def _open_capture(self):
        video_path = str(config.RUNNER_VIDEO_PATH).strip()
        if video_path:
            path = Path(video_path)
            if not path.exists():
                self.warning = f"Video not found: {path}"
                return None
            return cv2.VideoCapture(str(path))
        candidates = [config.RUNNER_CAMERA_INDEX]
        candidates.extend(i for i in range(4) if i != config.RUNNER_CAMERA_INDEX)
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
        for index in candidates:
            for backend in backends:
                cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.RUNNER_CAMERA_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.RUNNER_CAMERA_HEIGHT)
                    self.warning = "" if index == config.RUNNER_CAMERA_INDEX else f"Using camera index {index}"
                    return cap
                cap.release()
        return None

    def _loop(self):
        while self.running and self.capture is not None and self.capture.isOpened():
            ok, frame = self.capture.read()
            if not ok:
                if str(config.RUNNER_VIDEO_PATH).strip():
                    self.warning = "Video ended; keyboard only"
                    self.running = False
                    break
                time.sleep(0.02)
                continue
            if config.MIRROR_WEBCAM and not str(config.RUNNER_VIDEO_PATH).strip():
                frame = cv2.flip(frame, 1)
            timestamp = time.perf_counter() - self.start_time
            self.frame_index += 1
            pose_frame = PoseFrame.empty(timestamp, self.frame_index, "runner3d", 0)
            if self.detector is None:
                self._update_preview_frame(frame, pose_frame)
                self._maybe_start_pose_model()
            else:
                pose_frame = self._process_frame(frame, timestamp)
                self._update_preview_frame(frame, pose_frame)
            action = self.recognizer.update(pose_frame)
            if action is not Action.NONE:
                self._push_action(action)

    def _maybe_start_pose_model(self):
        if self.detector is not None or self.model_load_requested or self.frame_index < 8:
            return
        self.model_load_requested = True
        self.warning = "Loading pose model; camera preview active"
        self.model_thread = threading.Thread(target=self._load_pose_model, daemon=True)
        self.model_thread.start()

    def _process_frame(self, frame, timestamp: float) -> PoseFrame:
        if self.detector is None:
            pose_frame = PoseFrame.empty(timestamp, self.frame_index, "runner3d", 0)
            pose_frame.infer_ms = 0.0
            return pose_frame
        infer_frame = self._resize_for_inference(frame)
        scale_x = frame.shape[1] / infer_frame.shape[1]
        scale_y = frame.shape[0] / infer_frame.shape[0]
        try:
            people, infer_ms = self.detector.detect(infer_frame)
        except Exception as exc:
            self.warning = f"Pose inference failed: {exc}"
            return PoseFrame.empty(timestamp, self.frame_index, "runner3d", 0)
        for person in people:
            person.keypoints[:, 0] *= scale_x
            person.keypoints[:, 1] *= scale_y
            x1, y1, x2, y2 = person.bbox
            person.bbox = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
        if self.tracker is None or self.pose_filter is None:
            return PoseFrame.empty(timestamp, self.frame_index, "runner3d", len(people))
        selected_index, selected = self.tracker.select(people, frame.shape)
        selected = self.pose_filter.apply(selected)
        if selected is None:
            pose_frame = PoseFrame.empty(timestamp, self.frame_index, "runner3d", len(people))
            pose_frame.infer_ms = infer_ms
            return pose_frame
        if config.MIRROR_WEBCAM and not str(config.RUNNER_VIDEO_PATH).strip() and config.SWAP_LEFT_RIGHT:
            selected.keypoints, selected.valid_mask, selected.confidences = swap_left_right_keypoints(
                selected.keypoints, selected.valid_mask, selected.confidences
            )
        return PoseFrame(
            timestamp=timestamp,
            frame_index=self.frame_index,
            source="runner3d",
            bbox=selected.bbox,
            keypoints=selected.keypoints,
            confidences=selected.confidences,
            valid_mask=selected.valid_mask,
            pose_confidence=selected.pose_confidence,
            people_count=len(people),
            selected_index=-1 if selected_index is None else selected_index,
            infer_ms=infer_ms,
        )

    def _resize_for_inference(self, frame):
        width = frame.shape[1]
        if width <= config.RUNNER_INFER_WIDTH:
            return frame
        scale = config.RUNNER_INFER_WIDTH / width
        return cv2.resize(frame, (config.RUNNER_INFER_WIDTH, max(1, int(frame.shape[0] * scale))))

    def _update_preview_frame(self, frame, pose_frame: PoseFrame):
        preview = frame.copy()
        if pose_frame.keypoints is not None and pose_frame.valid_mask is not None:
            for a, b in config.SKELETON:
                if pose_frame.valid_mask[a] and pose_frame.valid_mask[b]:
                    p1 = tuple(pose_frame.keypoints[a].astype(int))
                    p2 = tuple(pose_frame.keypoints[b].astype(int))
                    cv2.line(preview, p1, p2, (0, 220, 90), 2)
            for idx, point in enumerate(pose_frame.keypoints):
                if pose_frame.valid_mask[idx]:
                    x, y = point.astype(int)
                    cv2.circle(preview, (x, y), 4, (40, 40, 255), -1)
        if pose_frame.bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in pose_frame.bbox]
            cv2.rectangle(preview, (x1, y1), (x2, y2), (80, 255, 80), 2)
        status = self.recognizer.status or "Calibrating"
        cv2.putText(preview, status[:36], (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        preview = cv2.resize(preview, (360, 202))
        rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        with self.preview_lock:
            self.preview_rgb = rgb
            self.preview_id += 1

    def _push_action(self, action: Action):
        try:
            self.queue.put_nowait(action)
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait(action)

    def drain_actions(self) -> list[Action]:
        actions: list[Action] = []
        while True:
            try:
                actions.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return actions

    def get_preview(self):
        with self.preview_lock:
            if self.preview_rgb is None:
                return self.preview_id, None
            return self.preview_id, self.preview_rgb.copy()

    def show_cv_preview(self, rgb_frame):
        if not config.RUNNER_SHOW_CV_PREVIEW or rgb_frame is None:
            return
        cv2.imshow("Pose Camera Preview", cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
        cv2.moveWindow("Pose Camera Preview", 40, 80)
        cv2.waitKey(1)

    def stop(self):
        self.running = False
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        cv2.destroyAllWindows()


def run_app() -> int:
    pose_input = PoseInput()
    pose_input.load_pose_model()

    from ursina import Entity, Ursina, application, color, window
    from .runner_3d_game import Runner3DGame

    app = Ursina(title=config.RUNNER_3D_WINDOW_TITLE, borderless=False)
    window.color = color.rgb32(12, 16, 22)
    pose_input.start()
    game = Runner3DGame()
    last_preview_id = -1
    game.update_preview(None)

    def update():
        nonlocal last_preview_id
        for action in pose_input.drain_actions():
            game.handle_action(action)
        preview_id, preview = pose_input.get_preview()
        if preview_id != last_preview_id:
            game.update_preview(preview)
            pose_input.show_cv_preview(preview)
            last_preview_id = preview_id
        game.update(pose_input.recognizer.status, pose_input.recognizer.calibrated, pose_input.warning)

    def input(key):
        if key in {"a", "left arrow"}:
            game.handle_action(Action.LEFT)
        elif key in {"d", "right arrow"}:
            game.handle_action(Action.RIGHT)
        elif key in {"w", "up arrow", "space"}:
            game.handle_action(Action.JUMP)
        elif key in {"s", "down arrow"}:
            game.handle_action(Action.SLIDE)
        elif key == "c":
            pose_input.recognizer.reset_calibration()
        elif key == "r":
            game.reset()
        elif key == "escape":
            pose_input.stop()
            application.quit()

    controller = Entity(name="runner3d_controller", eternal=True)
    controller.update = update
    controller.input = input
    try:
        app.run()
    finally:
        pose_input.stop()
    return 0
