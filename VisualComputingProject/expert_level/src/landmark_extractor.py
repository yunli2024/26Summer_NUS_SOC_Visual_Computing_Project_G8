"""Haar + LBF landmark extraction for FER images."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from . import config
    from .face_geometry import expand_face_box, face_area, face_iou
    from .image_preprocessing import preprocess_realtime_image
except ImportError:
    import config
    from face_geometry import expand_face_box, face_area, face_iou
    from image_preprocessing import preprocess_realtime_image


class LandmarkExtractor:
    """Extract 68 facial landmarks from small FER images."""

    def __init__(self, haar_path: Path = config.HAAR_CASCADE_PATH, lbf_path: Path = config.LBF_MODEL_PATH) -> None:
        if not haar_path.exists():
            raise FileNotFoundError(f"Haar cascade not found: {haar_path}")
        if not lbf_path.exists():
            raise FileNotFoundError(f"LBF model not found: {lbf_path}")
        if not hasattr(cv2, "face"):
            raise RuntimeError("cv2.face is unavailable; opencv-contrib-python is required.")
        self.face_detector = cv2.CascadeClassifier(str(haar_path))
        if self.face_detector.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {haar_path}")
        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel(str(lbf_path))

    def _dedupe_faces(self, faces: list[dict]) -> list[dict]:
        selected: list[dict] = []
        for candidate in sorted(faces, key=lambda item: face_area(item["face"]), reverse=True):
            if any(face_iou(candidate["face"], existing["face"]) > config.REALTIME_DETECTION_DEDUPE_IOU for existing in selected):
                continue
            selected.append(candidate)
        return selected

    def preprocess(self, image, upscale: int = config.FER_UPSCALE, mode: str = "clahe"):
        if image is None:
            return None
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if upscale > 1:
            gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        if mode == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        return gray

    def extract(self, image):
        if image is None:
            return False, None, None, "bad_image"

        lbf_failures = []
        for strategy in config.FER_DETECTION_STRATEGIES:
            gray = self.preprocess(
                image,
                upscale=strategy["upscale"],
                mode=strategy["preprocess"],
            )
            faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=strategy["scaleFactor"],
                minNeighbors=strategy["minNeighbors"],
                minSize=strategy["minSize"],
            )
            if len(faces) == 0:
                continue
            faces = np.asarray(sorted(faces, key=lambda f: f[2] * f[3], reverse=True), dtype=np.int32)
            for face_box in faces[:3]:
                face = np.asarray([face_box], dtype=np.int32)
                try:
                    ok, landmarks = self.facemark.fit(gray, face)
                except cv2.error:
                    lbf_failures.append(f"lbf_fit_error:{strategy['name']}")
                    continue
                if not ok or landmarks is None or len(landmarks) == 0:
                    lbf_failures.append(f"lbf_fit_failed:{strategy['name']}")
                    continue
                points = np.asarray(landmarks[0], dtype=np.float32).reshape(-1, 2)
                if len(points) != 68:
                    lbf_failures.append(f"unexpected_landmark_count_{len(points)}:{strategy['name']}")
                    continue
                return True, points, face_box, f"ok:{strategy['name']}"

        if lbf_failures:
            return False, None, None, lbf_failures[0]
        return False, None, None, "haar_no_face_all_strategies"

    def extract_original(self, image):
        """Extract landmarks and map them back to the original image size."""
        ok, points, face_box, message = self.extract(image)
        if not ok:
            return False, None, None, message
        strategy_name = message.split(":", 1)[1] if message.startswith("ok:") else ""
        strategy = next(
            (item for item in config.FER_DETECTION_STRATEGIES if item["name"] == strategy_name),
            {"upscale": 1},
        )
        upscale = float(strategy.get("upscale", 1))
        if upscale <= 0:
            upscale = 1.0
        mapped_points = points / upscale
        x, y, w, h = np.asarray(face_box, dtype=np.float32) / upscale
        height, width = image.shape[:2]
        x = max(0, min(float(x), width - 1))
        y = max(0, min(float(y), height - 1))
        w = max(1, min(float(w), width - x))
        h = max(1, min(float(h), height - y))
        mapped_face = np.asarray([round(x), round(y), round(w), round(h)], dtype=np.int32)
        return True, mapped_points.astype(np.float32), mapped_face, message

    def extract_realtime(self, frame, preprocess_mode: str | None = None):
        """Extract landmarks in the original camera coordinate system.

        Unlike FER extraction, this does not upscale the image. Returned face
        boxes and landmarks can be drawn directly on the original frame.
        """
        results = self.extract_realtime_all(frame, preprocess_mode=preprocess_mode)
        if not results:
            return False, None, None, "realtime_no_face"
        first = results[0]
        if not first["success"]:
            return False, first.get("landmarks"), first.get("face"), first.get("message", "realtime_lbf_fit_failed")
        return True, first["landmarks"], first["face"], first["message"]

    def _detect_realtime_faces(self, gray, search_faces=None, force_full_frame: bool = True) -> list[dict]:
        height, width = gray.shape[:2]
        candidates: list[dict] = []

        if config.REALTIME_USE_ROI_TRACKING and search_faces:
            for search_face in search_faces:
                roi_box = expand_face_box(
                    search_face,
                    width,
                    height,
                    expand_x=config.REALTIME_ROI_EXPAND_X,
                    expand_top=config.REALTIME_ROI_EXPAND_TOP,
                    expand_bottom=config.REALTIME_ROI_EXPAND_BOTTOM,
                )
                rx, ry, rw, rh = [int(value) for value in roi_box]
                if rw <= 0 or rh <= 0:
                    continue
                roi_gray = gray[ry : ry + rh, rx : rx + rw]
                min_w = max(24, int(float(search_face[2]) * config.REALTIME_ROI_MIN_FACE_SCALE))
                min_h = max(24, int(float(search_face[3]) * config.REALTIME_ROI_MIN_FACE_SCALE))
                roi_faces = self.face_detector.detectMultiScale(
                    roi_gray,
                    scaleFactor=config.REALTIME_FACE_SCALE_FACTOR,
                    minNeighbors=config.REALTIME_FACE_MIN_NEIGHBORS,
                    minSize=(min_w, min_h),
                )
                for face in sorted(roi_faces, key=lambda item: item[2] * item[3], reverse=True)[
                    : config.REALTIME_ROI_MAX_RESULTS_PER_TRACK
                ]:
                    x, y, w, h = [int(value) for value in face]
                    candidates.append(
                        {
                            "face": np.asarray([x + rx, y + ry, w, h], dtype=np.int32),
                            "detection_source": "ROI",
                        }
                    )

        if force_full_frame or not candidates:
            full_faces = self.face_detector.detectMultiScale(
                gray,
                scaleFactor=config.REALTIME_FACE_SCALE_FACTOR,
                minNeighbors=config.REALTIME_FACE_MIN_NEIGHBORS,
                minSize=config.REALTIME_FACE_MIN_SIZE,
            )
            for face in full_faces:
                candidates.append({"face": np.asarray(face, dtype=np.int32), "detection_source": "FULL"})

        return self._dedupe_faces(candidates)

    def extract_realtime_all(
        self,
        frame,
        max_faces: int = 5,
        preprocess_mode: str | None = None,
        search_faces=None,
        force_full_frame: bool = True,
    ):
        """Extract realtime landmarks for multiple faces in original frame coordinates."""
        if frame is None:
            return []
        height, width = frame.shape[:2]
        gray = preprocess_realtime_image(frame, preprocess_mode or config.REALTIME_PREPROCESS_MODE)
        detected_faces = self._detect_realtime_faces(
            gray,
            search_faces=search_faces,
            force_full_frame=force_full_frame,
        )
        if len(detected_faces) == 0:
            return []
        results = []
        for detected in detected_faces[:max_faces]:
            face_box = detected["face"]
            detection_source = detected.get("detection_source", "FULL")
            display_face_box = expand_face_box(
                face_box,
                width,
                height,
                expand_x=config.REALTIME_FACE_EXPAND_X,
                expand_top=config.REALTIME_FACE_EXPAND_TOP,
                expand_bottom=config.REALTIME_FACE_EXPAND_BOTTOM,
            )
            fit_face_box = expand_face_box(
                face_box,
                width,
                height,
                expand_x=config.REALTIME_LBF_EXPAND_X,
                expand_top=config.REALTIME_LBF_EXPAND_TOP,
                expand_bottom=config.REALTIME_LBF_EXPAND_BOTTOM,
            )
            face = np.asarray([fit_face_box], dtype=np.int32)
            try:
                ok, landmarks = self.facemark.fit(gray, face)
            except cv2.error:
                results.append(
                    {
                        "success": False,
                        "landmarks": None,
                        "face": face_box,
                        "display_face": display_face_box,
                        "fit_face": fit_face_box,
                        "raw_face": face_box,
                        "detection_source": detection_source,
                        "message": "realtime_lbf_fit_error",
                    }
                )
                continue
            if not ok or landmarks is None or len(landmarks) == 0:
                results.append(
                    {
                        "success": False,
                        "landmarks": None,
                        "face": face_box,
                        "display_face": display_face_box,
                        "fit_face": fit_face_box,
                        "raw_face": face_box,
                        "detection_source": detection_source,
                        "message": "realtime_lbf_fit_failed",
                    }
                )
                continue
            points = np.asarray(landmarks[0], dtype=np.float32).reshape(-1, 2)
            if len(points) != 68:
                results.append(
                    {
                        "success": False,
                        "landmarks": None,
                        "face": face_box,
                        "display_face": display_face_box,
                        "fit_face": fit_face_box,
                        "raw_face": face_box,
                        "detection_source": detection_source,
                        "message": f"realtime_unexpected_landmark_count_{len(points)}",
                    }
                )
                continue
            results.append(
                {
                    "success": True,
                    "landmarks": points,
                    "face": face_box,
                    "display_face": display_face_box,
                    "fit_face": fit_face_box,
                    "raw_face": face_box,
                    "detection_source": detection_source,
                    "message": "ok:realtime_original_coords",
                }
            )
        return results
