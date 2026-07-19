"""Configuration values for Part II face landmark detection.

All paths are built with pathlib so the project can move to another folder
without changing Windows-specific absolute paths.
"""

from pathlib import Path


BEGINNER_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BEGINNER_DIR.parent

HAAR_CASCADE_PATH = PROJECT_ROOT / "resources" / "face_models" / "haarcascade_frontalface_default.xml"
LBF_MODEL_PATH = PROJECT_ROOT / "resources" / "face_models" / "lbfmodel.yaml"

OUTPUTS_DIR = BEGINNER_DIR / "outputs"
SCREENSHOTS_DIR = OUTPUTS_DIR / "screenshots"
ROBUSTNESS_RESULTS_DIR = OUTPUTS_DIR / "robustness_results"
LOGS_DIR = OUTPUTS_DIR / "logs"

CAMERA_INDEX = 0
WINDOW_NAME = "Part II Face Landmarks"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (60, 60)

USE_CLAHE = False
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)

QUIT_KEY = "q"
