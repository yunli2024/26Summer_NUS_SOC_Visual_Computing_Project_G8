"""Improved Beginner Level configuration."""

from pathlib import Path


IMPROVED_DIR = Path(__file__).resolve().parents[1]
BEGINNER_DIR = IMPROVED_DIR.parent
PROJECT_ROOT = BEGINNER_DIR.parent

HAAR_CASCADE_PATH = PROJECT_ROOT / "resources" / "face_models" / "haarcascade_frontalface_default.xml"
LBF_MODEL_PATH = PROJECT_ROOT / "resources" / "face_models" / "lbfmodel.yaml"

CAMERA_INDEX = 0
WINDOW_NAME = "Improved Beginner Face Landmarks"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
QUIT_KEY = "q"

# More conservative Haar settings to reduce nose/mouth false positives.
FACE_SCALE_FACTOR = 1.05
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (80, 80)
FACE_MAX_SIZE = (420, 420)

USE_CLAHE = False
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

PADDING_RATIO = 0.15
MAX_FACE_HOLD_FRAMES = 3

LANDMARK_SMOOTHING_ALPHA = 0.35
FACE_CHANGE_IOU_THRESHOLD = 0.25
FACE_MIN_AREA_RATIO = 0.015
FACE_MAX_AREA_RATIO = 0.75
FACE_MIN_ASPECT_RATIO = 0.65
FACE_MAX_ASPECT_RATIO = 1.45
SUDDEN_SHRINK_AREA_RATIO = 0.35
INNER_FALSE_POSITIVE_IOU = 0.90
SINGLE_FACE_MODE = True
