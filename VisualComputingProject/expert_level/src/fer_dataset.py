from __future__ import annotations

import json
import random
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np

from keypoint_features import EMOTION_CLASSES


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class ImageRecord:
    split: str
    label: str
    path: str


def is_real_image_member(name: str) -> bool:
    normalized = name.replace("\\", "/")
    parts = normalized.split("/")
    if "__MACOSX" in parts or any(part.startswith("._") for part in parts):
        return False
    return normalized.lower().endswith(IMAGE_EXTENSIONS)


def parse_record(name: str) -> Optional[ImageRecord]:
    if not is_real_image_member(name):
        return None
    parts = name.replace("\\", "/").split("/")
    for idx, part in enumerate(parts[:-2]):
        if part in {"train", "test"}:
            label = parts[idx + 1]
            if label in EMOTION_CLASSES:
                return ImageRecord(split=part, label=label, path=name)
    return None


class FerZipDataset:
    def __init__(self, zip_path: Path) -> None:
        if not zip_path.exists():
            raise FileNotFoundError(f"Dataset zip not found: {zip_path}")
        self.zip_path = zip_path
        self.records = self._load_records()

    def _load_records(self) -> List[ImageRecord]:
        records: List[ImageRecord] = []
        with zipfile.ZipFile(self.zip_path) as zf:
            for name in zf.namelist():
                record = parse_record(name)
                if record is not None:
                    records.append(record)
        return records

    def inventory(self) -> Dict[str, Dict[str, int]]:
        counts: Dict[str, Counter] = {"train": Counter(), "test": Counter()}
        for record in self.records:
            counts[record.split][record.label] += 1
        return {
            split: {label: int(counter.get(label, 0)) for label in EMOTION_CLASSES}
            for split, counter in counts.items()
        }

    def validate(self) -> None:
        inventory = self.inventory()
        missing = []
        for split in ("train", "test"):
            for label in EMOTION_CLASSES:
                if inventory[split].get(label, 0) <= 0:
                    missing.append(f"{split}/{label}")
        if missing:
            raise RuntimeError(f"Dataset is incomplete. Missing classes: {', '.join(missing)}")

    def sample_records(
        self,
        split: str,
        *,
        max_per_class: int = 0,
        seed: int = 42,
    ) -> List[ImageRecord]:
        by_label: Dict[str, List[ImageRecord]] = defaultdict(list)
        for record in self.records:
            if record.split == split:
                by_label[record.label].append(record)

        rng = random.Random(seed)
        selected: List[ImageRecord] = []
        for label in EMOTION_CLASSES:
            records = sorted(by_label[label], key=lambda item: item.path)
            if max_per_class > 0 and len(records) > max_per_class:
                records = rng.sample(records, max_per_class)
                records = sorted(records, key=lambda item: item.path)
            selected.extend(records)
        return selected

    def read_image(self, record: ImageRecord) -> np.ndarray:
        with zipfile.ZipFile(self.zip_path) as zf:
            data = zf.read(record.path)
        buffer = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to decode image: {record.path}")
        return image

    def write_inventory(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "zip_path": str(self.zip_path),
            "classes": list(EMOTION_CLASSES),
            "counts": self.inventory(),
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
