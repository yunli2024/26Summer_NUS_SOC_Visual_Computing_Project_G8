"""Improved LBF landmark detector with exponential smoothing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from . import config
    from .head_pose import HeadPoseSmoother, classify_head_pose, estimate_head_pose
except ImportError:
    import config
    from head_pose import HeadPoseSmoother, classify_head_pose, estimate_head_pose


class SmoothedLandmarkDetector:
    """Detect 68 landmarks and smooth them when the target is stable."""

    def __init__(self, model_path: Path = config.LBF_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"LBF model file not found: {self.model_path}")
        if not hasattr(cv2, "face"):
            raise RuntimeError("cv2.face is not available. Install opencv-contrib-python.")
        self.facemark = cv2.face.createFacemarkLBF()
        try:
            self.facemark.loadModel(str(self.model_path))
        except cv2.error as exc:
            raise RuntimeError(f"Failed to load LBF model: {self.model_path}") from exc
        self.previous_tracks = []
        self.landmark_smoothers = []
        self.display_smoothers = []
        self.pose_smoothers = []
        self.last_valid_faces = np.empty((0, 4), dtype=np.int32)
        self.display_faces = []
        self.last_pose_labels = []

    def reset(self) -> None:
        self.previous_tracks = []
        self.landmark_smoothers = []
        self.display_smoothers = []
        self.pose_smoothers = []
        self.last_valid_faces = np.empty((0, 4), dtype=np.int32)
        self.display_faces = []
        self.last_pose_labels = []

    def fit(self, gray_frame, faces, detection_status: str):
        if detection_status == "CACHED":
            return False, [], "Cached face box only; landmarks not updated"
        if detection_status == "LOST":
            self.reset()
            return False, [], "Face lost; smoothing reset"
        if len(faces) == 0:
            self.reset()
            return False, [], "No face detected"
        fit_faces = np.asarray([_expand_face_box(face, gray_frame.shape) for face in faces], dtype=np.int32)
        try:
            ok, landmarks = self.facemark.fit(gray_frame, fit_faces)
        except cv2.error as exc:
            return False, [], f"Landmark fitting failed: {exc}"
        if not ok or landmarks is None or len(landmarks) == 0:
            return False, [], "Landmark fitting failed"

        current = [np.asarray(face_points, dtype=np.float32).reshape(-1, 2) for face_points in landmarks]
        smoothed = []
        next_tracks = []
        next_landmark_smoothers = []
        next_display_smoothers = []
        self._next_display_faces = []
        valid_faces = []
        pose_labels = []
        next_pose_smoothers = []
        rejection_reasons = []
        for idx, points in enumerate(current):
            face = tuple(int(v) for v in faces[idx])
            valid, reason, pose = _validate_landmarks(points, fit_faces[idx], gray_frame.shape)
            if not valid:
                rejection_reasons.append(reason)
                continue
            landmark_smoother = self._landmark_smoother_for(face)
            smoothed_points = landmark_smoother.update(points, np.asarray(face, dtype=np.float32))
            display_face = self._display_face_for(face)
            valid_faces.append(display_face)
            pose_smoother = self._pose_smoother_for(face)
            smoothed_pose = pose_smoother.update(pose)
            display_points = _pose_adjusted_display_landmarks(
                smoothed_points,
                np.asarray(face, dtype=np.float32),
                smoothed_pose,
            )
            display_smoother = self._display_smoother_for(face)
            display_points = display_smoother.update(display_points)
            smoothed.append(display_points)
            next_tracks.append((face, smoothed_points))
            next_landmark_smoothers.append((face, landmark_smoother))
            next_display_smoothers.append((face, display_smoother))
            next_pose_smoothers.append((face, pose_smoother))
            pose_labels.append(_pose_label(smoothed_pose))
        self.previous_tracks = next_tracks
        self.landmark_smoothers = next_landmark_smoothers
        self.display_smoothers = next_display_smoothers
        self.pose_smoothers = next_pose_smoothers
        self.display_faces = self._next_display_faces
        self._next_display_faces = []
        self.last_valid_faces = np.asarray(valid_faces, dtype=np.int32)
        self.last_pose_labels = pose_labels
        if not smoothed:
            reason = rejection_reasons[0] if rejection_reasons else "no_valid_landmarks"
            return False, [], f"Rejected obvious non-face candidate: {reason}"
        return True, smoothed, "Validated 68-point landmarks detected"

    def _match_previous(self, face):
        best_iou = 0.0
        best_points = None
        for previous_face, previous_points in self.previous_tracks:
            overlap = _face_iou(previous_face, face)
            if overlap > best_iou:
                best_iou = overlap
                best_points = previous_points
        if best_iou < config.FACE_CHANGE_IOU_THRESHOLD:
            return None
        return best_points

    def _pose_smoother_for(self, face):
        return _match_helper(face, self.pose_smoothers, HeadPoseSmoother)

    def _landmark_smoother_for(self, face):
        return _match_helper(face, self.landmark_smoothers, LandmarkSmoother)

    def _display_smoother_for(self, face):
        return _match_helper(face, self.display_smoothers, DisplayLandmarkSmoother)

    def _display_face_for(self, face):
        previous_face = _match_face(face, self.display_faces)
        face_array = np.asarray(face, dtype=np.float32)
        if previous_face is None:
            display_face = face_array
        else:
            alpha = config.BOX_SMOOTHING_ALPHA
            display_face = alpha * face_array + (1.0 - alpha) * previous_face
        self._next_display_faces.append((tuple(int(v) for v in face), display_face))
        return tuple(int(round(v)) for v in display_face)


class DisplayLandmarkSmoother:
    """Extra visual-only smoothing so landmark drawing does not flicker."""

    def __init__(
        self,
        alpha: float = config.DISPLAY_LANDMARK_ALPHA,
        jaw_alpha: float = config.DISPLAY_JAW_ALPHA,
    ) -> None:
        self.alpha = alpha
        self.jaw_alpha = jaw_alpha
        self.smoothed: np.ndarray | None = None

    def update(self, landmarks: np.ndarray) -> np.ndarray:
        current = landmarks.astype(np.float32, copy=False)
        if self.smoothed is None or self.smoothed.shape != current.shape:
            self.smoothed = current.copy()
            return self.smoothed
        alpha = np.full((68, 1), self.alpha, dtype=np.float32)
        alpha[0:17] = self.jaw_alpha
        if _is_open_mouth(current):
            alpha[48:68] = config.DISPLAY_OPEN_MOUTH_ALPHA
        self.smoothed = alpha * current + (1.0 - alpha) * self.smoothed
        return self.smoothed


class LandmarkSmoother:
    """Expert-style region-aware smoothing for realtime 68-point landmarks."""

    def __init__(self) -> None:
        self.smoothed: np.ndarray | None = None
        self.previous_face: np.ndarray | None = None
        self.previous_eye_angle: float | None = None

    def _eye_angle(self, landmarks: np.ndarray) -> float:
        left_eye = landmarks[36:42].mean(axis=0)
        right_eye = landmarks[42:48].mean(axis=0)
        vector = right_eye - left_eye
        return float(np.degrees(np.arctan2(vector[1], vector[0])))

    def _should_reset(self, landmarks: np.ndarray, face: np.ndarray) -> bool:
        if self.smoothed is None or self.previous_face is None or self.previous_eye_angle is None:
            return True
        old_center = _face_center(self.previous_face)
        new_center = _face_center(face)
        old_width = max(float(self.previous_face[2]), 1.0)
        center_shift = float(np.linalg.norm(new_center - old_center)) / old_width
        old_area = max(_face_area(self.previous_face), 1.0)
        new_area = max(_face_area(face), 1.0)
        area_delta = abs(new_area - old_area) / old_area
        eye_angle = self._eye_angle(landmarks)
        angle_delta = abs(eye_angle - self.previous_eye_angle)
        return (
            center_shift > config.LANDMARK_RESET_CENTER_SHIFT
            or area_delta > config.LANDMARK_RESET_AREA_RATIO
            or angle_delta > config.LANDMARK_RESET_EYE_ANGLE_DEGREES
        )

    def _constrain_jawline(self, landmarks: np.ndarray, face: np.ndarray) -> np.ndarray:
        constrained = landmarks.copy()
        x, y, w, h = [float(value) for value in face]
        min_x = x - w * config.LANDMARK_JAW_BOUND_MARGIN_X
        max_x = x + w * (1.0 + config.LANDMARK_JAW_BOUND_MARGIN_X)
        min_y = y + h * config.LANDMARK_JAW_BOUND_TOP_RATIO
        max_y = y + h * config.LANDMARK_JAW_BOUND_BOTTOM_RATIO
        interior_center_x = float(np.mean(constrained[[27, 30, 33, 51, 57], 0]))
        left_eye = constrained[36:42].mean(axis=0)
        right_eye = constrained[42:48].mean(axis=0)
        eye_distance = float(np.linalg.norm(right_eye - left_eye))
        interior_half_width = max(
            eye_distance * config.LANDMARK_JAW_EYE_HALF_WIDTH_RATIO,
            w * config.LANDMARK_JAW_MIN_HALF_WIDTH_RATIO,
        )
        min_x = max(min_x, interior_center_x - interior_half_width)
        max_x = min(max_x, interior_center_x + interior_half_width)
        constrained[0:17, 0] = np.clip(constrained[0:17, 0], min_x, max_x)
        constrained[0:17, 1] = np.clip(constrained[0:17, 1], min_y, max_y)
        return constrained

    def update(self, landmarks: np.ndarray, face: np.ndarray) -> np.ndarray:
        current = landmarks.astype(np.float32, copy=False)
        face = np.asarray(face, dtype=np.float32)
        current = self._constrain_jawline(current, face)
        current = _stabilize_open_mouth_lower_lip(current, face)
        if self._should_reset(current, face):
            self.smoothed = current.copy()
        else:
            face_scale = max(float(face[2]), float(face[3]), 1.0)
            max_delta = np.full((68, 1), face_scale * config.LANDMARK_MAX_POINT_JUMP_RATIO, dtype=np.float32)
            max_delta[0:17] = face_scale * config.LANDMARK_JAW_MAX_POINT_JUMP_RATIO
            if _is_open_mouth(current):
                max_delta[48:68] = face_scale * config.LANDMARK_MOUTH_MAX_POINT_JUMP_RATIO
            delta = current - self.smoothed
            distance = np.linalg.norm(delta, axis=1, keepdims=True)
            scale = np.minimum(1.0, max_delta / np.maximum(distance, 1e-6))
            current = self.smoothed + delta * scale
            alpha = np.full((68, 1), config.LANDMARK_ALPHA_BROW_NOSE, dtype=np.float32)
            alpha[0:17] = config.LANDMARK_ALPHA_JAW
            alpha[36:48] = config.LANDMARK_ALPHA_EYES_MOUTH
            alpha[48:68] = config.LANDMARK_ALPHA_EYES_MOUTH
            if _is_open_mouth(current):
                alpha[48:68] = config.LANDMARK_ALPHA_OPEN_MOUTH
            self.smoothed = alpha * current + (1.0 - alpha) * self.smoothed
            self.smoothed = self._constrain_jawline(self.smoothed, face)
            self.smoothed = _stabilize_open_mouth_lower_lip(self.smoothed, face)
        self.previous_face = face.copy()
        self.previous_eye_angle = self._eye_angle(current)
        return self.smoothed


def _match_helper(face, smoother_pairs, factory):
    best_iou = 0.0
    best_smoother = None
    for previous_face, smoother in smoother_pairs:
        overlap = _face_iou(previous_face, face)
        if overlap > best_iou:
            best_iou = overlap
            best_smoother = smoother
    if best_iou < config.FACE_CHANGE_IOU_THRESHOLD or best_smoother is None:
        return factory()
    return best_smoother


def _match_face(face, face_pairs):
    best_iou = 0.0
    best_face = None
    for previous_key, previous_face in face_pairs:
        overlap = _face_iou(previous_key, face)
        if overlap > best_iou:
            best_iou = overlap
            best_face = previous_face
    if best_iou < config.FACE_CHANGE_IOU_THRESHOLD:
        return None
    return best_face


def _face_center(face) -> np.ndarray:
    x, y, w, h = [float(value) for value in face]
    return np.asarray([x + w / 2.0, y + h / 2.0], dtype=np.float32)


def _face_area(face) -> float:
    _x, _y, w, h = [float(value) for value in face]
    return max(w, 0.0) * max(h, 0.0)


def _pose_adjusted_display_landmarks(landmarks: np.ndarray, face: np.ndarray, head_pose) -> np.ndarray:
    display = landmarks.astype(np.float32, copy=True)
    if head_pose is None:
        return display
    yaw, _pitch, _roll = head_pose
    yaw_abs = abs(float(yaw))
    if yaw_abs <= config.DISPLAY_POSE_CONTOUR_START_YAW:
        return display

    start = config.DISPLAY_POSE_CONTOUR_START_YAW
    full = max(config.DISPLAY_POSE_CONTOUR_FULL_YAW, start + 1e-6)
    strength = min(1.0, max(0.0, (yaw_abs - start) / (full - start)))
    shrink = 1.0 - config.DISPLAY_POSE_CONTOUR_MAX_SHRINK * strength
    x, y, w, h = [float(value) for value in face]
    anchor_x = float(np.mean(display[[30, 33, 51, 57], 0]))
    jaw = display[0:17].copy()
    jaw[:, 0] = anchor_x + (jaw[:, 0] - anchor_x) * shrink
    jaw[:, 1] = jaw[:, 1] - h * config.DISPLAY_POSE_CONTOUR_Y_LIFT * strength
    jaw[:, 0] = np.clip(
        jaw[:, 0],
        x - w * config.LANDMARK_JAW_BOUND_MARGIN_X,
        x + w * (1.0 + config.LANDMARK_JAW_BOUND_MARGIN_X),
    )
    jaw[:, 1] = np.clip(
        jaw[:, 1],
        y + h * config.LANDMARK_JAW_BOUND_TOP_RATIO,
        y + h * config.LANDMARK_JAW_BOUND_BOTTOM_RATIO,
    )
    display[0:17] = jaw
    return display


def _is_open_mouth(landmarks: np.ndarray, face=None) -> bool:
    if face is None:
        face_h = max(float(np.ptp(landmarks[:, 1])), 1.0)
    else:
        face_h = max(float(face[3]), 1.0)
    outer_open = float(landmarks[57, 1] - landmarks[51, 1])
    inner_open = float(landmarks[66, 1] - landmarks[62, 1])
    return max(outer_open, inner_open) / face_h >= config.MOUTH_OPEN_RATIO


def _stabilize_open_mouth_lower_lip(landmarks: np.ndarray, face: np.ndarray) -> np.ndarray:
    if not _is_open_mouth(landmarks, face):
        return landmarks
    adjusted = landmarks.copy()
    face_h = max(float(face[3]), 1.0)
    upper_lip_y = float(np.mean(adjusted[[50, 51, 52, 61, 62, 63], 1]))
    min_lower_y = upper_lip_y + face_h * config.MOUTH_OPEN_LOWER_LIP_MIN_GAP_RATIO
    lower_lip_indices = [55, 56, 57, 58, 59, 65, 66, 67]
    adjusted[lower_lip_indices, 1] = np.maximum(adjusted[lower_lip_indices, 1], min_lower_y)
    return adjusted


def _face_iou(face_a, face_b) -> float:
    ax, ay, aw, ah = face_a
    bx, by, bw, bh = face_b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _expand_face_box(face, image_shape):
    h, w = image_shape[:2]
    x, y, fw, fh = [float(value) for value in face]
    new_x = x - fw * config.LBF_EXPAND_X
    new_y = y - fh * config.LBF_EXPAND_TOP
    new_w = fw * (1.0 + 2.0 * config.LBF_EXPAND_X)
    new_h = fh * (1.0 + config.LBF_EXPAND_TOP + config.LBF_EXPAND_BOTTOM)
    x1 = max(0, int(round(new_x)))
    y1 = max(0, int(round(new_y)))
    x2 = min(w, int(round(new_x + new_w)))
    y2 = min(h, int(round(new_y + new_h)))
    if x2 <= x1 or y2 <= y1:
        return tuple(int(v) for v in face)
    return x1, y1, x2 - x1, y2 - y1


def _validate_landmarks(points, face, image_shape):
    if len(points) != 68 or not np.isfinite(points).all():
        return False, "bad_count_or_nan", None

    x, y, w, h = [float(value) for value in face]
    if w <= 0 or h <= 0:
        return False, "bad_face_box", None

    margin = config.LANDMARK_MAX_OUTSIDE_MARGIN
    left = x - w * margin
    right = x + w * (1.0 + margin)
    top = y - h * margin
    bottom = y + h * (1.0 + margin)
    inside_x = (points[:, 0] >= left) & (points[:, 0] <= right)
    inside_y = (points[:, 1] >= top) & (points[:, 1] <= bottom)
    inside_ratio = float(np.mean(inside_x & inside_y))
    if inside_ratio < config.LANDMARK_MIN_INSIDE_RATIO:
        return False, "landmarks_outside_face", None

    spread_x = float(np.ptp(points[:, 0]))
    spread_y = float(np.ptp(points[:, 1]))
    if spread_x < w * config.LANDMARK_MIN_FACE_SPREAD_RATIO:
        return False, "landmark_width_too_small", None
    if spread_y < h * config.LANDMARK_MIN_FACE_SPREAD_RATIO:
        return False, "landmark_height_too_small", None

    pose = _pose_with_2d_yaw_fallback(points, face, estimate_head_pose(points, image_shape))
    return True, "ok", pose


def _pose_label(pose):
    if pose is None:
        return "Pose: unknown"
    yaw, pitch, _roll = pose
    return f"{classify_head_pose(yaw, pitch)} yaw={yaw:.0f} pitch={pitch:.0f}"


def _pose_with_2d_yaw_fallback(points, face, pose):
    if not config.POSE_USE_2D_YAW_FALLBACK:
        return pose

    x, y, w, h = [float(value) for value in face]
    if w <= 0 or h <= 0:
        return pose

    nose_tip = points[30]
    box_center_x = x + w / 2.0
    yaw_2d = (float(nose_tip[0]) - box_center_x) / max(w, 1.0) * config.POSE_2D_YAW_SCALE
    yaw_2d = float(np.clip(yaw_2d, -config.POSE_2D_MAX_ABS_YAW, config.POSE_2D_MAX_ABS_YAW))

    if pose is None:
        return yaw_2d, 0.0, 0.0

    yaw, pitch, roll = pose
    if abs(float(pitch)) > config.POSE_MAX_REASONABLE_ABS_PITCH:
        pitch = 0.0
    if abs(yaw_2d) > abs(float(yaw)) + 8.0:
        yaw = yaw_2d
    return float(yaw), float(pitch), float(roll)
