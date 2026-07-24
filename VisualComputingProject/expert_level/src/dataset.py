"""Dataset inspection and preparation for Expert Level."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import cv2

try:
    from . import config
except ImportError:
    import config


RESOURCE_DATASET_DIR = config.RESOURCES_DIR / "expression_data" / "facial_expression_dataset"


def ensure_expert_dirs() -> None:
    for path in [
        config.RAW_DATA_DIR,
        config.MODELS_DIR,
        config.RESULTS_DIR,
        config.OUTPUTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def is_valid_path(path: Path | str) -> bool:
    path = Path(path)
    if "__MACOSX" in path.parts:
        return False
    if path.name == ".DS_Store":
        return False
    if path.name.startswith("._"):
        return False
    return True


def dataset_root() -> Path:
    raw_nested = config.RAW_DATA_DIR / "facial_expression_dataset"
    if raw_nested.exists():
        return raw_nested
    if RESOURCE_DATASET_DIR.exists():
        return RESOURCE_DATASET_DIR
    return config.RAW_DATA_DIR


def prepare_dataset_if_needed() -> Path:
    """Prepare raw data only when Expert raw data and resource folder are absent."""
    ensure_expert_dirs()
    root = dataset_root()
    if (root / "train").exists() and (root / "test").exists():
        return root
    if not config.EXPRESSION_ZIP.exists():
        raise FileNotFoundError(f"Dataset zip not found: {config.EXPRESSION_ZIP}")
    with zipfile.ZipFile(config.EXPRESSION_ZIP) as zf:
        for member in zf.infolist():
            if member.is_dir() or not is_valid_path(member.filename):
                continue
            target = config.RAW_DATA_DIR / member.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
    return dataset_root()


def image_paths(split: str, label: str) -> list[Path]:
    label_dir = dataset_root() / split / label
    if not label_dir.exists():
        return []
    return sorted(
        path
        for path in label_dir.iterdir()
        if path.is_file() and path.suffix.lower() in config.IMAGE_EXTENSIONS and is_valid_path(path)
    )


def readable_image_paths(split: str, label: str, limit: int | None = None) -> tuple[list[Path], list[tuple[Path, str]]]:
    good: list[Path] = []
    bad: list[tuple[Path, str]] = []
    for path in image_paths(split, label):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            bad.append((path, "bad_image"))
            continue
        good.append(path)
        if limit is not None and len(good) >= limit:
            break
    return good, bad


def inspect_dataset() -> dict:
    prepare_dataset_if_needed()
    counts = {split: {} for split in config.SPLITS}
    for split in config.SPLITS:
        for label in config.CLASSES:
            counts[split][label] = len(image_paths(split, label))
    return counts


def print_counts(counts: dict) -> None:
    for split in config.SPLITS:
        print(f"{split}:")
        for label in config.CLASSES:
            print(f"  {label}: {counts[split].get(label, 0)}")
