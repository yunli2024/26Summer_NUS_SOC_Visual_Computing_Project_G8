"""Validate project assets before a demo, training run, or release check."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Asset:
    label: str
    relative_path: str
    minimum_bytes: int
    profiles: frozenset[str]
    zip_container: bool = False


ASSETS = (
    Asset(
        "Haar face detector",
        "resources/face_models/haarcascade_frontalface_default.xml",
        500_000,
        frozenset({"tracked", "demo", "training"}),
    ),
    Asset(
        "YuNet face detector",
        "resources/face_models/face_detection_yunet_2023mar.onnx",
        100_000,
        frozenset({"tracked", "demo", "training"}),
    ),
    Asset(
        "LBF landmark model (local)",
        "resources/face_models/lbfmodel.yaml",
        10_000_000,
        frozenset({"demo", "training"}),
    ),
    Asset(
        "YOLOv8 nano pose model",
        "resources/pose_models/yolov8n-pose.pt",
        5_000_000,
        frozenset({"tracked", "demo", "training"}),
        zip_container=True,
    ),
    Asset(
        "Dance reference video",
        "resources/videos/dance_example_1.mp4",
        1_000_000,
        frozenset({"tracked", "demo", "training"}),
    ),
    Asset(
        "Expression classifier",
        "expert_level/artifacts_svm_geometry/expression_classifier.joblib",
        1_000_000,
        frozenset({"tracked", "demo"}),
    ),
    Asset(
        "Dance pose cache",
        "bonus_level/task2_results/dance_example_1/pose_cache.npz",
        100_000,
        frozenset({"tracked", "demo"}),
        zip_container=True,
    ),
    Asset(
        "FER-style dataset archive (local)",
        "resources/expression_data/facial_expression_dataset.zip",
        1_000_000,
        frozenset({"training"}),
        zip_container=True,
    ),
)


def validate_asset(asset: Asset) -> dict[str, object]:
    path = PROJECT_DIR / asset.relative_path
    result: dict[str, object] = {
        "label": asset.label,
        "path": asset.relative_path,
        "status": "ok",
    }
    if not path.is_file():
        result["status"] = "missing"
        result["detail"] = "file not found"
        return result

    size = path.stat().st_size
    result["bytes"] = size
    if size < asset.minimum_bytes:
        result["status"] = "invalid"
        result["detail"] = f"expected at least {asset.minimum_bytes:,} bytes"
        return result

    if asset.zip_container:
        try:
            if not zipfile.is_zipfile(path):
                raise zipfile.BadZipFile("not a ZIP-compatible container")
            with zipfile.ZipFile(path) as archive:
                corrupt_member = archive.testzip()
            if corrupt_member is not None:
                raise zipfile.BadZipFile(f"corrupt member: {corrupt_member}")
        except (OSError, zipfile.BadZipFile) as error:
            result["status"] = "invalid"
            result["detail"] = str(error)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("tracked", "demo", "training"),
        default="demo",
        help="tracked=CI/release assets; demo=all live-demo assets; training=training inputs",
    )
    parser.add_argument("--json", type=Path, help="also write the complete result as JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = [asset for asset in ASSETS if args.profile in asset.profiles]
    results = [validate_asset(asset) for asset in selected]
    failures = [result for result in results if result["status"] != "ok"]

    print(f"Asset profile: {args.profile}")
    for result in results:
        size = (
            f"{int(result['bytes']) / (1024 * 1024):7.2f} MiB"
            if "bytes" in result
            else "       -"
        )
        detail = f" ({result['detail']})" if "detail" in result else ""
        print(
            f"  {str(result['status']).upper():7} {size}  "
            f"{result['path']}{detail}"
        )

    payload = {
        "profile": args.profile,
        "project_root": str(PROJECT_DIR),
        "passed": not failures,
        "assets": results,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON report: {args.json}")

    if failures:
        print(
            "\nAsset check failed. Restore the listed files before running this profile.",
            file=sys.stderr,
        )
        return 1
    print(f"Asset check passed: {len(results)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
