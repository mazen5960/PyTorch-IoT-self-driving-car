import os
import time
from typing import Dict, List, Tuple


MODEL_PATH = os.path.join(os.path.dirname(__file__), "efficientdet.tflite")
HALT_CLASSES = {"person", "stop sign", "car"}
    

class VisionHalt:
    #tries mediapipe first, falls back to opencv though. we js kept mediapipe

    def __init__(
        self,
        score_threshold: float = 0.5,
        backend: str = "auto",
        model_path: str = MODEL_PATH,
        camera_size: Tuple[int, int] = (640, 480),
        max_results: int = 5,
        halt_classes=None,
    ):
        if backend not in {"auto", "mediapipe", "opencv"}:
            raise ValueError("backend must be one of: auto, mediapipe, opencv")

        self.score_threshold = float(score_threshold)
        self.halt_classes = {c.lower() for c in (halt_classes or HALT_CLASSES)}
        self.model_path = model_path
        self.max_results = int(max_results)
        self.backend_name = None
        self.last_detections: List[Dict] = []
        self._init_errors: List[str] = []

        from picamera2 import Picamera2

        self.cam = Picamera2()
        self.cam.configure(
            self.cam.create_video_configuration(
                main={"format": "RGB888", "size": camera_size}
            )
        )
        self.cam.start()
        time.sleep(0.2)

        if backend in {"auto", "mediapipe"}:
            try:
                self._setup_mediapipe()
                self.backend_name = "mediapipe"
            except Exception as e:
                self._init_errors.append(f"mediapipe: {e}")
                if backend == "mediapipe":
                    raise

        if self.backend_name is None and backend in {"auto", "opencv"}:
            try:
                self._setup_opencv()
                self.backend_name = "opencv"
            except Exception as e:
                self._init_errors.append(f"opencv: {e}")
                if backend == "opencv":
                    raise

        if self.backend_name is None:
            err = "; ".join(self._init_errors) if self._init_errors else "unknown init error"
            raise RuntimeError(f"no vision backend available ({err})")

    def _setup_mediapipe(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"model not found: {self.model_path}")

        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._mp_ts_ms = 0

        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.ObjectDetectorOptions(
            base_options=base_options,
            score_threshold=self.score_threshold,
            max_results=self.max_results,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._mp_detector = vision.ObjectDetector.create_from_options(options)

    def _setup_opencv(self):
        import cv2

        self._cv2 = cv2
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _detect_with_mediapipe(self, frame) -> List[Dict]:
        now_ms = int(time.monotonic() * 1000)
        self._mp_ts_ms = now_ms if now_ms > self._mp_ts_ms else self._mp_ts_ms + 1

        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame)
        result = self._mp_detector.detect_for_video(mp_image, self._mp_ts_ms)

        out = []
        for det in result.detections:
            if not det.categories:
                continue
            cat = det.categories[0]
            name = (cat.category_name or "unknown").lower()
            score = float(cat.score)
            bbox = (0, 0, 0, 0)
            if det.bounding_box is not None:
                bb = det.bounding_box
                bbox = (int(bb.origin_x), int(bb.origin_y), int(bb.width), int(bb.height))
            out.append({"name": name, "score": score, "bbox": bbox})
        return out

    def _detect_stop_signs_opencv(self, frame_bgr) -> List[Dict]:
        cv2 = self._cv2

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        lower1 = (0, 80, 60)
        upper1 = (10, 255, 255)
        lower2 = (170, 80, 60)
        upper2 = (180, 255, 255)
        mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = frame_bgr.shape[:2]
        frame_area = float(max(1, h * w))
        out = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 350:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) < 6 or len(approx) > 10:
                continue

            x, y, bw, bh = cv2.boundingRect(approx)
            if bw <= 0 or bh <= 0:
                continue
            aspect = bw / float(bh)
            if aspect < 0.65 or aspect > 1.35:
                continue

            bbox_area = float(bw * bh)
            fill_ratio = area / max(1.0, bbox_area)
            if fill_ratio < 0.25 or fill_ratio > 0.95:
                continue

            red_pixels = cv2.countNonZero(mask[y : y + bh, x : x + bw])
            red_ratio = red_pixels / max(1.0, bbox_area)
            if red_ratio < 0.12:
                continue

            score = min(0.99, 0.4 + (area / (0.15 * frame_area)))
            out.append({"name": "stop sign", "score": float(score), "bbox": (x, y, bw, bh)})
        return out

    def _detect_with_opencv(self, frame) -> List[Dict]:
        cv2 = self._cv2
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        out: List[Dict] = []
        rects, weights = self._hog.detectMultiScale(
            frame_bgr, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        for (x, y, w, h), wt in zip(rects, weights):
            score = float(wt)
            if score < self.score_threshold:
                continue
            out.append({"name": "person", "score": score, "bbox": (int(x), int(y), int(w), int(h))})

        out.extend(self._detect_stop_signs_opencv(frame_bgr))
        return out

    def detect_once(self) -> List[Dict]:
        frame = self.cam.capture_array()
        if self.backend_name == "mediapipe":
            self.last_detections = self._detect_with_mediapipe(frame)
        elif self.backend_name == "opencv":
            self.last_detections = self._detect_with_opencv(frame)
        else:
            self.last_detections = []
        return self.last_detections

    def check(self) -> bool:
        detections = self.detect_once()
        for det in detections:
            name = str(det.get("name", "")).lower()
            score = float(det.get("score", 0.0))
            if name in self.halt_classes and score >= self.score_threshold:
                print(f"HALT[{self.backend_name}]: {name} (score={score:.2f})")
                return True
        return False

    def close(self):
        try:
            self.cam.stop()
        except Exception:
            pass
