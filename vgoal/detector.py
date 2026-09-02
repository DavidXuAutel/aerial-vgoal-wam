"""Target detector frontend wrapping lightweight YOLO and mock detectors.

Provides a unified interface: detect(rgb_image) -> Optional[DetectionResult]
where DetectionResult contains [u_min, v_min, u_max, v_max], confidence, and class_name.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np

from vgoal.prompt_classes import classes_from_visual_prompt


@dataclass
class DetectionResult:
    """Detection output for a single object candidate."""
    bbox: np.ndarray      # [u_min, v_min, u_max, v_max] in float32 pixel coordinates
    confidence: float     # Detection confidence in [0.0, 1.0]
    class_id: int         # Class ID (e.g. COCO class index)
    class_name: str       # Human-readable class name (e.g. "car", "person", "landing_pad")

    @property
    def center(self) -> np.ndarray:
        """Center point (u, v) in pixel coordinates."""
        return np.array([
            (self.bbox[0] + self.bbox[2]) * 0.5,
            (self.bbox[1] + self.bbox[3]) * 0.5,
        ], dtype=np.float32)

    @property
    def area(self) -> float:
        """Bounding box pixel area."""
        w = max(0.0, float(self.bbox[2] - self.bbox[0]))
        h = max(0.0, float(self.bbox[3] - self.bbox[1]))
        return w * h


class BaseDetector:
    """Base interface for visual object-goal detectors."""

    def detect(self, rgb: np.ndarray) -> Optional[DetectionResult]:
        """Detect highest-confidence target object in the given RGB image."""
        raise NotImplementedError


class MockDetector(BaseDetector):
    """Deterministic mock detector for offline tests, unit tests, and headless verification."""

    def __init__(
        self,
        target_bbox: Optional[Sequence[float]] = None,
        confidence: float = 0.95,
        class_name: str = "target_object",
    ) -> None:
        self.target_bbox = np.asarray(target_bbox, dtype=np.float32) if target_bbox is not None else None
        self.confidence = float(confidence)
        self.class_name = str(class_name)

    def set_target(self, bbox: Optional[Sequence[float]], confidence: float = 0.95) -> None:
        """Update mock target position and confidence."""
        self.target_bbox = np.asarray(bbox, dtype=np.float32) if bbox is not None else None
        self.confidence = float(confidence)

    def detect(self, rgb: np.ndarray) -> Optional[DetectionResult]:
        if self.target_bbox is None or self.confidence <= 0.0:
            return None
        return DetectionResult(
            bbox=self.target_bbox.copy(),
            confidence=self.confidence,
            class_id=0,
            class_name=self.class_name,
        )


class YOLOTargetDetector(BaseDetector):
    """Lightweight YOLO detector (YOLOv8n / YOLOv10n / custom landing-pad weights)."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        *,
        target_classes: Optional[Sequence[Union[str, int]]] = None,
        conf_threshold: float = 0.4,
        imgsz: int = 640,
        device: str = "cpu",
    ) -> None:
        self.model_path = str(model_path)
        self.target_classes = set(target_classes) if target_classes is not None else None
        self.conf_threshold = float(conf_threshold)
        self.imgsz = int(imgsz)
        self.device = str(device)
        self._model = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for YOLOTargetDetector. Install via `pip install ultralytics`"
            ) from e

    def detect_all(self, rgb: np.ndarray) -> List[DetectionResult]:
        """Detect all matching target candidates in the image."""
        self._lazy_load()
        # ultralytics handles RGB numpy arrays [H, W, 3] directly
        results = self._model(
            rgb,
            conf=self.conf_threshold,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: List[DetectionResult] = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        names = r.names or {}
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy().astype(int)

        for bbox, conf, cls_id in zip(boxes, confs, classes):
            cls_name = names.get(int(cls_id), str(cls_id))
            # Filter by target class if specified
            if self.target_classes is not None:
                if cls_name not in self.target_classes and int(cls_id) not in self.target_classes:
                    continue
            detections.append(
                DetectionResult(
                    bbox=np.asarray(bbox, dtype=np.float32),
                    confidence=float(conf),
                    class_id=int(cls_id),
                    class_name=str(cls_name),
                )
            )

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def detect(self, rgb: np.ndarray) -> Optional[DetectionResult]:
        """Detect the single best (highest confidence) target object."""
        all_matches = self.detect_all(rgb)
        return all_matches[0] if len(all_matches) > 0 else None


class OpenVocabPromptDetector(BaseDetector):
    """Prompt-conditioned detector reusing YOLO / YOLO-World.

    * If ``inner`` is provided (tests), detect delegates to it.
    * If ``model_path`` contains ``\"world\"``, load ultralytics YOLO-World and
      ``set_classes`` from the visual prompt.
    * Otherwise wrap :class:`YOLOTargetDetector` with COCO class name filter.
    """

    def __init__(
        self,
        visual_prompt: str,
        *,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.4,
        imgsz: int = 640,
        device: str = "cpu",
        inner: Optional[BaseDetector] = None,
    ) -> None:
        self.visual_prompt = str(visual_prompt)
        self.model_path = str(model_path)
        self.conf_threshold = float(conf_threshold)
        self.imgsz = int(imgsz)
        self.device = str(device)
        self._inner = inner
        self._world_model = None
        self._yolo_wrap: Optional[YOLOTargetDetector] = None
        if self._inner is None:
            self._rebuild_backend()

    def set_visual_prompt(self, prompt: str) -> None:
        self.visual_prompt = str(prompt)
        if self._inner is None:
            self._rebuild_backend()

    def _classes(self) -> List[str]:
        return classes_from_visual_prompt(self.visual_prompt)

    def _rebuild_backend(self) -> None:
        classes = self._classes()
        if "world" in self.model_path.lower():
            self._yolo_wrap = None
            self._world_model = None  # lazy
            self._world_classes = classes
        else:
            self._world_model = None
            self._yolo_wrap = YOLOTargetDetector(
                model_path=self.model_path,
                target_classes=classes or None,
                conf_threshold=self.conf_threshold,
                imgsz=self.imgsz,
                device=self.device,
            )

    def _lazy_world(self) -> None:
        if self._world_model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for YOLO-World. Install via `pip install ultralytics`"
            ) from e
        self._world_model = YOLO(self.model_path)
        classes = getattr(self, "_world_classes", self._classes())
        if classes and hasattr(self._world_model, "set_classes"):
            self._world_model.set_classes(classes)

    def detect(self, rgb: np.ndarray) -> Optional[DetectionResult]:
        if self._inner is not None:
            return self._inner.detect(rgb)
        if "world" in self.model_path.lower():
            self._lazy_world()
            results = self._world_model(
                rgb,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                return None
            r = results[0]
            names = r.names or {}
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            best_i = int(np.argmax(confs))
            cls_id = int(classes[best_i])
            return DetectionResult(
                bbox=np.asarray(boxes[best_i], dtype=np.float32),
                confidence=float(confs[best_i]),
                class_id=cls_id,
                class_name=str(names.get(cls_id, str(cls_id))),
            )
        assert self._yolo_wrap is not None
        return self._yolo_wrap.detect(rgb)

    def detect_all(self, rgb: np.ndarray) -> List[DetectionResult]:
        """Return all prompt-class hits (needed for nearest-instance selection)."""
        if self._inner is not None:
            inner_all = getattr(self._inner, "detect_all", None)
            if callable(inner_all):
                return list(inner_all(rgb) or [])
            one = self._inner.detect(rgb)
            return [one] if one is not None else []
        if "world" in self.model_path.lower():
            self._lazy_world()
            results = self._world_model(
                rgb,
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                return []
            r = results[0]
            names = r.names or {}
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)
            out: List[DetectionResult] = []
            for i in range(len(boxes)):
                cls_id = int(classes[i])
                out.append(
                    DetectionResult(
                        bbox=np.asarray(boxes[i], dtype=np.float32),
                        confidence=float(confs[i]),
                        class_id=cls_id,
                        class_name=str(names.get(cls_id, str(cls_id))),
                    )
                )
            out.sort(key=lambda d: d.confidence, reverse=True)
            return out
        assert self._yolo_wrap is not None
        return self._yolo_wrap.detect_all(rgb)
