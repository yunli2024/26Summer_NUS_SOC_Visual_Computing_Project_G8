"""Configuration for the Bonus Level pose scoring demo."""

from __future__ import annotations

from pathlib import Path


BONUS_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BONUS_DIR.parent

MODEL_PATH = PROJECT_DIR / "resources" / "pose_models" / "yolov8n-pose.pt"
DEFAULT_VIDEO_PATH = PROJECT_DIR / "resources" / "videos" / "dance_example_1.mp4"
TIKTOK_DATASET_DIR = BONUS_DIR / "data" / "input" / "reference_videos" / "tiktok"
RUNTIME_OUTPUT_DIR = BONUS_DIR / "outputs" / "runtime"

DISPLAY_SIZE = (640, 384)
YOLO_CONF_THRESHOLD = 0.30
KEYPOINT_CONF_THRESHOLD = 0.30
MIN_VALID_KEYPOINTS = 6

TRACK_WEIGHTS = {
    "area": 0.40,
    "center": 0.25,
    "continuity": 0.25,
    "confidence": 0.10,
}

EMA_ALPHA = 0.55
MAX_MISSING_FRAMES = 3

ALIGNMENT_WINDOW_SECONDS = 0.80
REFERENCE_BUFFER_SECONDS = 12.0
USER_BUFFER_FRAME_COUNT = 30
MOTION_WINDOW_SECONDS = 0.40
SCORE_EMA_ALPHA = 0.35
OFFICIAL_SCORE_INTERVAL = 1.00

POSE_SCORE_WEIGHTS = {
    "position": 0.35,
    "angle": 0.65,
}
FINAL_SCORE_WEIGHTS = {
    "pose": 0.55,
    "motion": 0.45,
}
POSITION_SIGMA = 0.55
ANGLE_SIGMA_DEGREES = 32.0
MOTION_VECTOR_SIGMA = 0.22
MOTION_NOISE_FLOOR = 0.04
MOTION_ACTIVE_THRESHOLD = 0.07
MOTION_HOLD_EXIT_THRESHOLD = 0.10
MOTION_HOLD_CONFIRM_SAMPLES = 2
MOTION_HOLD_EMA_ALPHA = 0.35
MIN_COMMON_BODY_KEYPOINTS = 4
ALLOW_MIRROR_MATCH = True

FEEDBACK_THRESHOLDS = {
    "perfect": 85.0,
    "super": 70.0,
    "good": 55.0,
}
GRADE_POINTS = {
    "Perfect": 1000,
    "Super": 700,
    "Good": 400,
    "Miss": 0,
    "Move!": 0,
}

KEYPOINT_ERROR_THRESHOLD = 0.34
ANGLE_ERROR_THRESHOLD = 42.0

FEEDBACK_FLASH_MS = 450
FEEDBACK_FLASH_COLORS = {
    "Perfect": "#ffd166",
    "Super": "#9bdbff",
    "Good": "#8de38d",
    "Miss": "#ff8f8f",
    "Move!": "#ff8f8f",
    "Hold": "#fff08d",
    "Sync": "#dddddd",
}

SOUND_ENABLED = True

MIRROR_WEBCAM = True
SWAP_LEFT_RIGHT = True

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

SKELETON = [
    (0, 5),
    (0, 6),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

LEFT_RIGHT_PAIRS = [
    (1, 2),
    (3, 4),
    (5, 6),
    (7, 8),
    (9, 10),
    (11, 12),
    (13, 14),
    (15, 16),
]

ANGLE_TRIPLES = {
    "left_elbow": (5, 7, 9),
    "right_elbow": (6, 8, 10),
    "left_shoulder": (7, 5, 11),
    "right_shoulder": (8, 6, 12),
    "left_hip": (5, 11, 13),
    "right_hip": (6, 12, 14),
    "left_knee": (11, 13, 15),
    "right_knee": (12, 14, 16),
}

BODY_JOINTS = tuple(range(5, 17))
