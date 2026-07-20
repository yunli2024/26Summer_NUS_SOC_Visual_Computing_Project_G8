"""Realtime expression prediction demo for Expert Level."""

from __future__ import annotations

from collections import Counter, deque
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image
import torch

try:
    from . import config
    from .face_geometry import face_area, face_center, face_inside, face_iou
    from .head_pose import HeadPoseSmoother, classify_head_pose, estimate_head_pose
    from .image_preprocessing import image_stats, preprocess_realtime_image
    from .landmark_extractor import LandmarkExtractor
    from .roi_inference import clamp_box, inference_device, roi_boxes_from_landmarks, roi_eval_transform
    from .roi_cnn_model import build_roi_cnn_model
    from .video_effects import TRACK_COLORS, draw_debug_hud, draw_face_label, draw_landmarks
except ImportError:
    import config
    from face_geometry import face_area, face_center, face_inside, face_iou
    from head_pose import HeadPoseSmoother, classify_head_pose, estimate_head_pose
    from image_preprocessing import image_stats, preprocess_realtime_image
    from landmark_extractor import LandmarkExtractor
    from roi_inference import clamp_box, inference_device, roi_boxes_from_landmarks, roi_eval_transform
    from roi_cnn_model import build_roi_cnn_model
    from video_effects import TRACK_COLORS, draw_debug_hud, draw_face_label, draw_landmarks


def majority_vote(history: deque[str]) -> str:
    if not history:
        return "unknown"
    return Counter(history).most_common(1)[0][0]


class RealtimeROICNNPredictor:
    """Realtime wrapper for the trained full-face + ROI CNN model."""

    def __init__(self, model_file=config.ROI_CNN_DEMO_MODEL_FILE, variant: str = config.ROI_CNN_DEMO_VARIANT) -> None:
        if not model_file.exists():
            raise FileNotFoundError(f"ROI CNN model not found: {model_file}")
        self.model_file = model_file
        self.variant = variant
        self.device = inference_device()
        checkpoint = torch.load(model_file, map_location=self.device)
        checkpoint_variant = checkpoint.get("metadata", {}).get("variant", variant)
        self.model = build_roi_cnn_model(checkpoint_variant).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transform = roi_eval_transform()
        self.classes = list(config.CLASSES)

    def _crop_tensor(self, gray: np.ndarray, box_xyxy: np.ndarray) -> torch.Tensor:
        height, width = gray.shape[:2]
        x1, y1, x2, y2 = clamp_box(box_xyxy, width, height)
        crop = gray[int(y1) : int(y2), int(x1) : int(x2)]
        if crop.size == 0:
            raise ValueError("empty ROI crop")
        image = Image.fromarray(crop).convert("L")
        return self.transform(image)

    def predict_from_frame(
        self,
        gray: np.ndarray,
        landmarks: np.ndarray,
        face_xywh: np.ndarray,
        pose: tuple[float, float, float] | None = None,
    ):
        x, y, w, h = [float(value) for value in face_xywh]
        face_box = np.asarray([x, y, x + w, y + h], dtype=np.float32)
        eye_box, mouth_box = roi_boxes_from_landmarks(landmarks, gray.shape)
        batch = {
            "face": self._crop_tensor(gray, face_box).unsqueeze(0).to(self.device),
            "eye_brow": self._crop_tensor(gray, eye_box).unsqueeze(0).to(self.device),
            "nose_mouth": self._crop_tensor(gray, mouth_box).unsqueeze(0).to(self.device),
        }
        with torch.no_grad():
            logits = self.model(batch)["main"]
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()[0].astype(np.float32)
        order = np.argsort(probabilities)[::-1]
        top_index = int(order[0])
        confidence = float(probabilities[top_index])
        second = float(probabilities[int(order[1])]) if probabilities.size > 1 else 0.0
        margin = confidence - second
        return self.classes[top_index], confidence, margin, probabilities, self.classes


class RealtimeROIEnsemblePredictor:
    """Fuse the original ROI-CNN and robust ROI-CNN by weighted probabilities."""

    def __init__(
        self,
        base_model_file=config.ROI_CNN_DEMO_MODEL_FILE,
        robust_model_file=config.ROBUST_ROI_CNN_MODEL_FILE,
        base_weight: float = config.ROI_ENSEMBLE_BASE_WEIGHT,
        robust_weight: float = config.ROI_ENSEMBLE_ROBUST_WEIGHT,
    ) -> None:
        if not base_model_file.exists():
            raise FileNotFoundError(f"Base ROI CNN model not found: {base_model_file}")
        if not robust_model_file.exists():
            raise FileNotFoundError(f"Robust ROI CNN model not found: {robust_model_file}")
        total = max(float(base_weight + robust_weight), 1e-6)
        self.base_weight = float(base_weight) / total
        self.robust_weight = float(robust_weight) / total
        self.base = RealtimeROICNNPredictor(model_file=base_model_file)
        self.robust = RealtimeROICNNPredictor(model_file=robust_model_file)
        self.classes = self.base.classes

    def predict_from_frame(
        self,
        gray: np.ndarray,
        landmarks: np.ndarray,
        face_xywh: np.ndarray,
        pose: tuple[float, float, float] | None = None,
    ):
        _base_label, _base_confidence, _base_margin, base_probabilities, classes = self.base.predict_from_frame(
            gray,
            landmarks,
            face_xywh,
        )
        _robust_label, _robust_confidence, _robust_margin, robust_probabilities, _classes = self.robust.predict_from_frame(
            gray,
            landmarks,
            face_xywh,
        )
        probabilities = self.base_weight * base_probabilities + self.robust_weight * robust_probabilities
        probabilities = probabilities.astype(np.float32, copy=False)
        order = np.argsort(probabilities)[::-1]
        top_index = int(order[0])
        confidence = float(probabilities[top_index])
        second = float(probabilities[int(order[1])]) if probabilities.size > 1 else 0.0
        margin = confidence - second
        return self.classes[top_index], confidence, margin, probabilities, classes


class PredictionSmoother:
    """Stabilize realtime labels with probability EWMA and asymmetric transitions."""

    def __init__(
        self,
        history_size: int = config.PREDICTION_HISTORY_SIZE,
        min_confidence: float = config.MIN_PREDICTION_CONFIDENCE,
        min_margin: float = config.MIN_PREDICTION_MARGIN,
        min_stable_frames: int = config.MIN_STABLE_FRAMES,
        uncertain_label: str = config.UNCERTAIN_LABEL,
        ema_alpha: float = config.REALTIME_PROBABILITY_EMA_ALPHA,
    ) -> None:
        self.history: deque[str] = deque(maxlen=history_size)
        self.probability_history: deque[np.ndarray] = deque(maxlen=history_size)
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.min_stable_frames = min_stable_frames
        self.ema_alpha = float(ema_alpha)
        self.uncertain_label = uncertain_label
        self.stable_label = uncertain_label
        self.candidate_label = uncertain_label
        self.candidate_frames = 0
        self.ema_probabilities: np.ndarray | None = None

    def reset(self) -> None:
        self.history.clear()
        self.probability_history.clear()
        self.stable_label = self.uncertain_label
        self.candidate_label = self.uncertain_label
        self.candidate_frames = 0
        self.ema_probabilities = None

    def apply_runtime_config(self, payload: dict | None) -> None:
        if not payload:
            return
        self.ema_alpha = float(payload.get("ema_alpha", self.ema_alpha))
        self.min_confidence = float(payload.get("min_confidence", self.min_confidence))
        self.min_margin = float(payload.get("min_margin", self.min_margin))

    def _smoothed_label(self, classes: list[str] | None) -> tuple[str, str]:
        if not classes or self.ema_probabilities is None:
            return majority_vote(self.history), "label vote"
        averaged = self.ema_probabilities
        order = np.argsort(averaged)[::-1]
        top_index = int(order[0])
        averaged_confidence = float(averaged[top_index])
        second = float(averaged[int(order[1])]) if averaged.size > 1 else 0.0
        averaged_margin = averaged_confidence - second

        if averaged_confidence < self.min_confidence:
            return self.uncertain_label, f"ema confidence {averaged_confidence:.2f} < {self.min_confidence:.2f}"
        if averaged_margin < self.min_margin:
            return self.uncertain_label, f"ema margin {averaged_margin:.2f} < {self.min_margin:.2f}"
        return classes[top_index], f"ema confidence {averaged_confidence:.2f}, margin {averaged_margin:.2f}"

    def _required_frames(self, next_label: str) -> int:
        if self.stable_label == "neutral" and next_label not in {"neutral", self.uncertain_label}:
            return config.REALTIME_NEUTRAL_TO_EXPRESSION_FRAMES
        if self.stable_label not in {"neutral", self.uncertain_label} and next_label == "neutral":
            return config.REALTIME_EXPRESSION_TO_NEUTRAL_FRAMES
        if self.stable_label == self.uncertain_label:
            return config.REALTIME_NEUTRAL_TO_EXPRESSION_FRAMES
        return config.REALTIME_EXPRESSION_SWITCH_FRAMES

    def update(
        self,
        raw_label: str,
        confidence,
        margin=None,
        probabilities: np.ndarray | None = None,
        classes: list[str] | None = None,
    ) -> tuple[str, str, str]:
        status_parts = []
        if confidence is not None and confidence < self.min_confidence:
            accepted_label = self.uncertain_label
            status_parts.append(f"low confidence {confidence:.2f} < {self.min_confidence:.2f}")
        elif margin is not None and margin < self.min_margin:
            accepted_label = self.uncertain_label
            status_parts.append(f"ambiguous margin {margin:.2f} < {self.min_margin:.2f}")
        else:
            accepted_label = raw_label
            status_parts.append("tracking")

        self.history.append(accepted_label)
        if probabilities is not None:
            current = np.asarray(probabilities, dtype=np.float32)
            self.probability_history.append(current)
            if self.ema_probabilities is None or self.ema_probabilities.shape != current.shape:
                self.ema_probabilities = current.copy()
            else:
                self.ema_probabilities = self.ema_alpha * current + (1.0 - self.ema_alpha) * self.ema_probabilities
        voted_label, average_status = self._smoothed_label(classes)
        status_parts.append(average_status)

        if voted_label == self.stable_label:
            self.candidate_label = voted_label
            self.candidate_frames = 0
            return self.stable_label, voted_label, "; ".join(status_parts)

        if voted_label == self.candidate_label:
            self.candidate_frames += 1
        else:
            self.candidate_label = voted_label
            self.candidate_frames = 1

        required_frames = self._required_frames(voted_label)
        if self.candidate_frames >= required_frames:
            self.stable_label = self.candidate_label
            self.candidate_frames = 0
            status_parts.append("stable label updated")
        else:
            status_parts.append(f"waiting {self.candidate_frames}/{required_frames}")

        return self.stable_label, voted_label, "; ".join(status_parts)

    def queue_text(self) -> str:
        return " ".join(self.history)


class DisplayLandmarkSmoother:
    """Extra visual-only smoothing so landmark drawing does not flicker."""

    def __init__(
        self,
        alpha: float = config.REALTIME_DISPLAY_LANDMARK_ALPHA,
        jaw_alpha: float = config.REALTIME_DISPLAY_JAW_ALPHA,
    ) -> None:
        self.alpha = alpha
        self.jaw_alpha = jaw_alpha
        self.smoothed: np.ndarray | None = None

    def reset(self) -> None:
        self.smoothed = None

    def update(self, landmarks: np.ndarray) -> np.ndarray:
        current = landmarks.astype(np.float32, copy=False)
        if self.smoothed is None or self.smoothed.shape != current.shape:
            self.smoothed = current.copy()
            return self.smoothed
        alpha = np.full((68, 1), self.alpha, dtype=np.float32)
        alpha[0:17] = self.jaw_alpha
        self.smoothed = alpha * current + (1.0 - alpha) * self.smoothed
        return self.smoothed


class LandmarkSmoother:
    """Region-aware smoothing for realtime 68-point landmarks."""

    def __init__(self) -> None:
        self.smoothed: np.ndarray | None = None
        self.previous_face: np.ndarray | None = None
        self.previous_eye_angle: float | None = None

    def reset(self) -> None:
        self.smoothed = None
        self.previous_face = None
        self.previous_eye_angle = None

    def _eye_angle(self, landmarks: np.ndarray) -> float:
        left_eye = landmarks[36:42].mean(axis=0)
        right_eye = landmarks[42:48].mean(axis=0)
        vector = right_eye - left_eye
        return float(np.degrees(np.arctan2(vector[1], vector[0])))

    def _should_reset(self, landmarks: np.ndarray, face: np.ndarray) -> bool:
        if self.smoothed is None or self.previous_face is None or self.previous_eye_angle is None:
            return True
        old_center = face_center(self.previous_face)
        new_center = face_center(face)
        old_width = max(float(self.previous_face[2]), 1.0)
        center_shift = float(np.linalg.norm(new_center - old_center)) / old_width
        old_area = max(face_area(self.previous_face), 1.0)
        new_area = max(face_area(face), 1.0)
        area_delta = abs(new_area - old_area) / old_area
        eye_angle = self._eye_angle(landmarks)
        angle_delta = abs(eye_angle - self.previous_eye_angle)
        return (
            center_shift > config.REALTIME_LANDMARK_RESET_CENTER_SHIFT
            or area_delta > config.REALTIME_LANDMARK_RESET_AREA_RATIO
            or angle_delta > config.REALTIME_LANDMARK_RESET_EYE_ANGLE_DEGREES
        )

    def _constrain_jawline(self, landmarks: np.ndarray, face: np.ndarray) -> np.ndarray:
        constrained = landmarks.copy()
        x, y, w, h = [float(value) for value in face]
        min_x = x - w * config.REALTIME_JAW_BOUND_MARGIN_X
        max_x = x + w * (1.0 + config.REALTIME_JAW_BOUND_MARGIN_X)
        min_y = y + h * config.REALTIME_JAW_BOUND_TOP_RATIO
        max_y = y + h * config.REALTIME_JAW_BOUND_BOTTOM_RATIO
        constrained[0:17, 0] = np.clip(constrained[0:17, 0], min_x, max_x)
        constrained[0:17, 1] = np.clip(constrained[0:17, 1], min_y, max_y)
        return constrained

    def update(self, landmarks: np.ndarray, face: np.ndarray) -> np.ndarray:
        current = landmarks.astype(np.float32, copy=False)
        face = np.asarray(face, dtype=np.float32)
        current = self._constrain_jawline(current, face)
        if self._should_reset(current, face):
            self.smoothed = current.copy()
        else:
            face_scale = max(float(face[2]), float(face[3]), 1.0)
            max_delta = np.full((68, 1), face_scale * config.REALTIME_LANDMARK_MAX_POINT_JUMP_RATIO, dtype=np.float32)
            max_delta[0:17] = face_scale * config.REALTIME_LANDMARK_JAW_MAX_POINT_JUMP_RATIO
            delta = current - self.smoothed
            distance = np.linalg.norm(delta, axis=1, keepdims=True)
            scale = np.minimum(1.0, max_delta / np.maximum(distance, 1e-6))
            current = self.smoothed + delta * scale
            alpha = np.full((68, 1), config.REALTIME_LANDMARK_ALPHA_BROW_NOSE, dtype=np.float32)
            alpha[0:17] = config.REALTIME_LANDMARK_ALPHA_JAW
            alpha[36:48] = config.REALTIME_LANDMARK_ALPHA_EYES_MOUTH
            alpha[48:68] = config.REALTIME_LANDMARK_ALPHA_EYES_MOUTH
            self.smoothed = alpha * current + (1.0 - alpha) * self.smoothed
            self.smoothed = self._constrain_jawline(self.smoothed, face)
        self.previous_face = face.copy()
        self.previous_eye_angle = self._eye_angle(current)
        return self.smoothed


def pose_adjusted_display_landmarks(
    landmarks: np.ndarray,
    face: np.ndarray,
    head_pose: tuple[float, float, float] | None,
) -> np.ndarray:
    """Return drawing-only landmarks with a pose-aware jawline.

    LBF often keeps a frontal jaw template while the head turns. The expression
    pipeline still receives the normal smoothed landmarks; only the displayed
    jawline is narrowed so the contour does not float outside the visible face.
    """

    display = landmarks.astype(np.float32, copy=True)
    if head_pose is None:
        return display
    yaw, _pitch, _roll = head_pose
    yaw_abs = abs(float(yaw))
    if yaw_abs <= config.REALTIME_DISPLAY_POSE_CONTOUR_START_YAW:
        return display

    start = config.REALTIME_DISPLAY_POSE_CONTOUR_START_YAW
    full = max(config.REALTIME_DISPLAY_POSE_CONTOUR_FULL_YAW, start + 1e-6)
    strength = min(1.0, max(0.0, (yaw_abs - start) / (full - start)))
    shrink = 1.0 - config.REALTIME_DISPLAY_POSE_CONTOUR_MAX_SHRINK * strength
    x, y, w, h = [float(value) for value in face]
    anchor_x = float(np.mean(display[[30, 33, 51, 57], 0]))
    jaw = display[0:17].copy()
    jaw[:, 0] = anchor_x + (jaw[:, 0] - anchor_x) * shrink
    jaw[:, 1] = jaw[:, 1] - h * config.REALTIME_DISPLAY_POSE_CONTOUR_Y_LIFT * strength
    jaw[:, 0] = np.clip(jaw[:, 0], x - w * config.REALTIME_JAW_BOUND_MARGIN_X, x + w * (1.0 + config.REALTIME_JAW_BOUND_MARGIN_X))
    jaw[:, 1] = np.clip(jaw[:, 1], y + h * config.REALTIME_JAW_BOUND_TOP_RATIO, y + h * config.REALTIME_JAW_BOUND_BOTTOM_RATIO)
    display[0:17] = jaw
    return display


@dataclass
class FaceTrack:
    track_id: int
    face: np.ndarray
    display_face: np.ndarray | None = None
    prediction_smoother: PredictionSmoother = field(default_factory=PredictionSmoother)
    display_landmark_smoother: DisplayLandmarkSmoother = field(default_factory=DisplayLandmarkSmoother)
    landmark_smoother: LandmarkSmoother = field(default_factory=LandmarkSmoother)
    pose_smoother: HeadPoseSmoother = field(default_factory=HeadPoseSmoother)
    frames_since_prediction: int = config.REALTIME_PREDICT_EVERY_N_FRAMES
    missed_frames: int = 0
    raw_expression: str = config.UNCERTAIN_LABEL
    voted_expression: str = config.UNCERTAIN_LABEL
    stable_expression: str = config.UNCERTAIN_LABEL
    confidence: float | None = None
    margin: float | None = None
    head_pose: tuple[float, float, float] | None = None
    pose_label: str = ""
    status: str = "new"

    def __post_init__(self) -> None:
        self.face = np.asarray(self.face, dtype=np.float32)
        self.display_face = self.face.copy()

    def update_face(self, face: np.ndarray) -> None:
        self.face = np.asarray(face, dtype=np.float32)
        if self.display_face is None:
            self.display_face = self.face.copy()
        else:
            alpha = config.REALTIME_BOX_SMOOTHING_ALPHA
            self.display_face = alpha * self.face + (1.0 - alpha) * self.display_face


def filter_realtime_detections(detections: list[dict], tracks: dict[int, FaceTrack], frame_shape) -> list[dict]:
    height, width = frame_shape[:2]
    frame_area = float(width * height)
    filtered = []
    for detection in detections:
        face = detection["face"]
        x, y, w, h = [float(v) for v in face]
        if w <= 0 or h <= 0:
            continue
        area = w * h
        aspect = w / h
        if area / frame_area < config.REALTIME_MIN_FACE_AREA_RATIO:
            continue
        if not (config.REALTIME_MIN_FACE_ASPECT <= aspect <= config.REALTIME_MAX_FACE_ASPECT):
            continue
        reject_inner = False
        for track in tracks.values():
            reference = track.display_face if track.display_face is not None else track.face
            if face_inside(face, reference) and area < face_area(reference) * config.REALTIME_REJECT_INNER_FACE_AREA_RATIO:
                reject_inner = True
                break
        if reject_inner:
            continue
        filtered.append(detection)
    return filtered


def search_faces_from_tracks(tracks: dict[int, FaceTrack]) -> list[np.ndarray]:
    """Return stable face boxes used as local Haar search regions."""

    search_faces = []
    for track in tracks.values():
        if track.missed_frames > config.REALTIME_MAX_CACHED_FRAMES:
            continue
        reference = track.display_face if track.display_face is not None else track.face
        search_faces.append(np.asarray(reference, dtype=np.int32))
    return search_faces


def match_tracks(tracks: dict[int, FaceTrack], detections: list[dict], next_track_id: int):
    assignments = []
    used_tracks = set()
    for detection in detections:
        center = face_center(detection["face"])
        best_id = None
        best_score = float("inf")
        for track_id, track in tracks.items():
            if track_id in used_tracks:
                continue
            reference = track.display_face if track.display_face is not None else track.face
            distance = float(np.linalg.norm(center - face_center(reference)))
            iou = face_iou(detection["face"], reference)
            if distance > config.REALTIME_MAX_MATCH_DISTANCE and iou < config.REALTIME_MIN_MATCH_IOU:
                continue
            size_penalty = abs(face_area(detection["face"]) - face_area(reference)) / max(face_area(reference), 1.0)
            score = distance - iou * 80.0 + size_penalty * 25.0
            if score < best_score:
                best_score = score
                best_id = track_id
        if best_id is not None:
            track = tracks[best_id]
            used_tracks.add(best_id)
        else:
            track = FaceTrack(track_id=next_track_id, face=detection["face"])
            tracks[next_track_id] = track
            used_tracks.add(next_track_id)
            next_track_id += 1
        assignments.append((track, detection))

    assigned_ids = {track.track_id for track, _detection in assignments}
    for track_id in list(tracks):
        if track_id not in assigned_ids:
            tracks[track_id].missed_frames += 1
            tracks[track_id].status = "CACHED"
            if tracks[track_id].missed_frames > config.REALTIME_MAX_CACHED_FRAMES:
                del tracks[track_id]
    visible_tracks = [track for track in tracks.values() if track.missed_frames <= config.REALTIME_MAX_CACHED_FRAMES]
    return assignments, visible_tracks, next_track_id


def run_demo(
    camera_index: int = config.CAMERA_INDEX,
    model_source: str = "roi-ensemble",
    debug_hud: bool = False,
    preprocess_mode: str = config.REALTIME_PREPROCESS_MODE,
    max_faces: int = config.REALTIME_DEFAULT_MAX_FACES,
    show_pose: bool = config.REALTIME_SHOW_POSE_LABEL,
) -> int:
    if model_source != "roi-ensemble":
        print(f"Unsupported model source: {model_source}. Only roi-ensemble is retained.")
        return 1

    required_checkpoints = [config.ROI_CNN_DEMO_MODEL_FILE, config.ROBUST_ROI_CNN_MODEL_FILE]
    missing_files = [path for path in required_checkpoints if not path.exists()]
    if missing_files:
        print("ROI ensemble is missing required checkpoint(s):")
        for path in missing_files:
            print(f"- {path}")
        return 1

    roi_predictor = RealtimeROIEnsemblePredictor()
    model_reason = (
        f"base={config.ROI_ENSEMBLE_BASE_WEIGHT:.2f}, "
        f"robust={config.ROI_ENSEMBLE_ROBUST_WEIGHT:.2f}"
    )
    extractor = LandmarkExtractor()
    tracks: dict[int, FaceTrack] = {}
    next_track_id = 1
    frame_index = 0
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Cannot open camera index {camera_index}.")
        return 1

    print(
        f"ROI Ensemble realtime demo started ({model_reason}), "
        f"preprocess={preprocess_mode}. Press q to quit."
    )
    last_time = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed.")
                break

            frame_index += 1
            raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            preprocessed_gray = preprocess_realtime_image(frame, preprocess_mode)
            brightness_mean, contrast_std = image_stats(preprocessed_gray)
            search_faces = search_faces_from_tracks(tracks)
            force_full_frame = not search_faces or frame_index % config.REALTIME_FULL_DETECT_INTERVAL == 0
            raw_detections = extractor.extract_realtime_all(
                frame,
                max_faces=max_faces,
                preprocess_mode=preprocess_mode,
                search_faces=search_faces,
                force_full_frame=force_full_frame,
            )
            detections = filter_realtime_detections(raw_detections, tracks, frame.shape)
            assignments, visible_tracks, next_track_id = match_tracks(tracks, detections, next_track_id)
            source_counts = Counter(detection.get("detection_source", "FULL") for detection in raw_detections)
            debug_lines = [
                f"faces: {len(assignments)} / raw {len(raw_detections)} source={dict(source_counts)}",
                f"roi search: {len(search_faces)} full={force_full_frame}",
                f"preprocess: {preprocess_mode} mean={brightness_mean:.1f} std={contrast_std:.1f}",
            ]
            for index, (track, detection) in enumerate(assignments):
                face = detection["face"]
                display_face = detection.get("display_face", face)
                landmarks = detection["landmarks"]
                track.update_face(display_face)
                track.missed_frames = 0
                color = TRACK_COLORS[index % len(TRACK_COLORS)]
                if not detection["success"]:
                    track.status = "TRACKED"
                    debug_lines.append(f"id {track.track_id}: {track.status}")
                    continue
                smoothed_landmarks = track.landmark_smoother.update(landmarks, face)
                raw_pose = estimate_head_pose(smoothed_landmarks, frame.shape)
                track.head_pose = track.pose_smoother.update(raw_pose)
                if track.head_pose is not None:
                    yaw, pitch, _roll = track.head_pose
                    track.pose_label = classify_head_pose(yaw, pitch)
                display_landmarks = pose_adjusted_display_landmarks(smoothed_landmarks, display_face, track.head_pose)
                display_landmarks = track.display_landmark_smoother.update(display_landmarks)
                try:
                    track.frames_since_prediction += 1
                    should_predict = track.frames_since_prediction >= config.REALTIME_PREDICT_EVERY_N_FRAMES
                    if should_predict:
                        (
                            track.raw_expression,
                            track.confidence,
                            track.margin,
                            probabilities,
                            classes,
                        ) = roi_predictor.predict_from_frame(raw_gray, smoothed_landmarks, face, track.head_pose)
                        track.stable_expression, track.voted_expression, track.status = track.prediction_smoother.update(
                            track.raw_expression,
                            track.confidence,
                            track.margin,
                            probabilities,
                            classes,
                        )
                        track.frames_since_prediction = 0
                    else:
                        track.stable_expression = track.prediction_smoother.stable_label
                    track.status = "TRACKED"
                except ValueError as exc:
                    track.stable_expression = track.prediction_smoother.stable_label
                    track.status = f"feature skipped: {exc}"
                draw_landmarks(frame, display_landmarks, color)
                debug_lines.append(
                    f"id {track.track_id}: raw={track.raw_expression} vote={track.voted_expression} "
                    f"stable={track.stable_expression} conf={track.confidence} margin={track.margin} "
                    f"pose={track.head_pose} {track.pose_label} {track.status} "
                    f"detect={detection.get('detection_source', 'FULL')}"
                )
            now = time.perf_counter()
            fps = 1.0 / max(now - last_time, 1e-6)
            last_time = now

            for index, track in enumerate(visible_tracks):
                color = TRACK_COLORS[index % len(TRACK_COLORS)]
                display_face = track.display_face if track.display_face is not None else track.face
                status = "CACHED" if track.missed_frames > 0 else "TRACKED"
                pose_label = track.pose_label if show_pose else ""
                draw_face_label(
                    frame,
                    display_face,
                    track.stable_expression,
                    track.confidence,
                    fps,
                    color,
                    status,
                    pose_label,
                )
            if debug_hud:
                draw_debug_hud(frame, debug_lines)
            cv2.imshow(config.DEMO_WINDOW_NAME, frame)
            if cv2.getWindowProperty(config.DEMO_WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if cv2.waitKey(1) & 0xFF == ord(config.QUIT_KEY):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0
